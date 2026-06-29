import optuna
import numpy as np
import time
import os
import torch
import gc

from copy import deepcopy
from tqdm import tqdm


import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aux.datos import GestorDatos
from aux.funciones_guardado import guardar_pipeline_state
from config import CONFIG as GLOBAL_CONFIG, get_fase_config
CONFIG = deepcopy(GLOBAL_CONFIG)
_f2_cfg = get_fase_config("FASE_2")
for k, v in _f2_cfg.items():
    CONFIG[k] = v
from motor_ga import entrenar_ventana

# --- PARÁMETROS DE LA FASE 2 (Sincronizados con FASE_2 de config.json) ---
N_TRIALS = CONFIG.get('N_TRIALS', 10)          
N_VENTANAS_REP = CONFIG.get('N_VENTANAS_REP', 3)  # Fijo: 3 ventanas ancladas cubre inicio/medio/reciente
N_MODELOS_OBJETIVO = CONFIG.get('n_comite', 1)

RESTART_STUDY = False

_DIR_SCRIPT = os.path.dirname(os.path.abspath(__file__))


X_global, R_global, idx_c_global, idx_m_global, idx_l_global = None, None, None, None, None

def get_datos_globales():
    global X_global, R_global, idx_c_global, idx_m_global, idx_l_global
    if X_global is None:
        fecha_fin = CONFIG.get('fecha_fin')
        if not fecha_fin:
            fecha_fin = time.strftime("%Y-%m-%d")
            CONFIG['fecha_fin'] = fecha_fin
        
        gestor = GestorDatos(CONFIG['tickers'], CONFIG['ticker_cash'], CONFIG['ticker_macro'], CONFIG['fecha_inicio'], fecha_fin)
        X_global, R_global, idx_c_global, idx_m_global, idx_l_global = gestor.obtener_datos_listos(escalar_global=False)
    return X_global, R_global, idx_c_global, idx_m_global, idx_l_global
def _calcular_score_r2_target(rets_ia, target_anual=0.12):
    """
    Score R2 Target (v8.0).
    Mide qué tan fielmente sigue la curva de capital la línea ideal del
    `target_anual` (por defecto 12% anual). Rango natural: (-∞, 1].
    Se clampea a [-1, 1] para dar gradiente útil a Optuna sin explotar
    el espacio de búsqueda en ventanas muy negativas.
    También retorna el MDD para que el agregador pueda penalizarlo.
    """
    rets_ia = np.array(rets_ia, dtype=np.float64)
    n = len(rets_ia)
    if n < 5:
        return 0.0, 1.0  # (r2, mdd)

    cap = np.cumprod(1.0 + rets_ia)

    # MDD para el factor de penalización
    max_cap = np.maximum.accumulate(cap)
    mdd = float(np.max(np.abs((max_cap - cap) / (max_cap + 1e-9))))

    # Línea objetivo en espacio logarítmico (parte de log(1)=0)
    m_target = np.log(1.0 + target_anual) / 252.0
    t_idx    = np.arange(n, dtype=np.float64)
    log_cap  = np.log(cap + 1e-9)
    y_target = t_idx * m_target

    # R² contra la línea ideal
    ss_res = float(np.sum((log_cap - y_target) ** 2))
    ss_tot = float(np.sum((log_cap - log_cap.mean()) ** 2)) + 1e-9
    r2 = float(np.clip(1.0 - ss_res / ss_tot, -1.0, 1.0))

    return r2, mdd



def objective(trial):
    X, R, idx_c, idx_m, idx_l = get_datos_globales()
    config_trial = get_fase_config("FASE_2")
    
    # 2. PARÁMETROS A OPTIMIZAR
    # ═══════════════════════════════════════════════════════

    # Arquitectura multi-capa (Saneada v6.8: Más conservadora para evitar overfitting)
    n1     = trial.suggest_int("n1", 16, 64, step=8)
    n2_raw = trial.suggest_int("n2", 0,  32, step=8)
    n2 = min(n2_raw, n1 - 8) if n2_raw > 0 else 0
    config_trial['ocultas'] = [x for x in [n1, n2] if x > 0]

    # Train size: cuánto histórico ve el modelo por ventana (1 a 3 años)
    # Más años → ve más regímenes → más robusto, pero más lento
    config_trial['train_size'] = trial.suggest_int("train_size", 252, 756, step=63)

    # Step size fijo a 63 días
    config_trial['step_size'] = 63

    # Umbral de rebalanceo
    config_trial['umbral_rebalanceo'] = trial.suggest_float("umbral_rebalanceo", 0.05, 0.20)

    # Feature Dropout Rate (Opción E): calibra diversidad del comité
    config_trial['feature_dropout_rate'] = trial.suggest_float("feature_dropout_rate", 0.10, 0.40)

    # 4. EVALUACIÓN
    config_trial['n_comite'] = N_MODELOS_OBJETIVO
    config_trial['silent'] = True 
    config_trial['fundadores_preminados'] = None
    config_trial['n_ventanas'] = N_VENTANAS_REP
    
    # Ventanas fijas ancladas al espacio OOS: idénticas en todos los trials
    idx_limite = len(X)
    espacio_oos_total = idx_limite - config_trial['train_size'] - config_trial['step_size']
    if espacio_oos_total < config_trial['step_size'] * 6:
        raise optuna.exceptions.TrialPruned()

    # 6 puntos fijos: 10%, 25%, 40%, 60%, 75%, 90% del espacio OOS disponible.
    # Cubren: crisis 2008, bull 2013, lateral 2015, bajista 2018, COVID 2020, post-rally 2022.
    n_ventanas_usar = 6
    anclas_raw = [int(espacio_oos_total * p) for p in [0.10, 0.25, 0.40, 0.60, 0.75, 0.90]]
    # Redondear al step_size más cercano
    anclas = [max(0, round(a / config_trial['step_size']) * config_trial['step_size']) for a in anclas_raw]
    # Garantizar que las 6 anclas son distintas
    anclas = sorted(set(anclas))
    if len(anclas) < 6:
        raise optuna.exceptions.TrialPruned()
    ventanas_seleccionadas = anclas[:6]
    ventanas_args = [(idx_v, i, X, R, idx_c, idx_m, idx_l, config_trial) for idx_v, i in enumerate(ventanas_seleccionadas)]


    
    # Barra de progreso del trial (Persistente en posición 1)
    pbar_sub = tqdm(total=n_ventanas_usar, desc=f"  └─ Trial {trial.number}", leave=False, position=1, bar_format='{l_bar}{bar:20}{r_bar}{bar:-10b}')

    try:
        resultados = []
        # ALPHA v6.9: Ejecución en serie para estabilidad MPS y evitar overhead de Pool(spawn)
        for idx_v, v_args in enumerate(ventanas_args):
            # Clonar config para evitar colisiones de pbar si existieran
            v_list = list(v_args)
            v_config = v_list[7].copy()
            v_config['pbar_sub'] = None
            v_list[7] = v_config
            
            res = entrenar_ventana(tuple(v_list))
            resultados.append(res)
            
            # Limpieza inmediata de GPU tras cada ventana del trial
            if torch.backends.mps.is_available():
                torch.mps.synchronize()
                torch.mps.empty_cache()
            gc.collect()

            # Actualizar barra
            pbar_sub.set_description(f"  └─ Trial {trial.number} | Window {idx_v+1}/{n_ventanas_usar}")
            pbar_sub.update(1)

            # --- PRUNING INTERMEDIO v8.0: R2 Target ---
            if len(resultados) >= 1:
                scores_temp = [_calcular_score_r2_target(np.array(r['ia_rets']))[0] for r in resultados]
                score_ultima = _calcular_score_r2_target(np.array(resultados[-1]['ia_rets']))[0]
                trial.report(score_ultima, idx_v)
                if trial.should_prune():
                    pbar_sub.close()
                    tqdm.write(f" [!] Trial {trial.number} Podado (R2 Temp: {float(np.mean(scores_temp)):.4f})")
                    raise optuna.exceptions.TrialPruned()

        # JUEZ v8.0 — R2 Target (fidelidad a curva ideal del 12% anual)
        r2_scores = []
        mdd_scores = []
        for res in resultados:
            r2, mdd = _calcular_score_r2_target(np.array(res['ia_rets']))
            r2_scores.append(r2)
            mdd_scores.append(mdd)

        if len(r2_scores) < 6:
            raise optuna.exceptions.TrialPruned()

        r2_arr   = np.array(r2_scores)
        r2_medio = float(np.mean(r2_arr))
        r2_vol   = float(np.std(r2_arr, ddof=1))

        # Penalizar variabilidad entre regímenes (consistencia estructural)
        score_consistente = r2_medio - 0.5 * r2_vol

        # Factor suelo: penaliza multiplicativamente si alguna ventana tiene MDD > 40%
        max_mdd = float(np.max(mdd_scores))
        factor_suelo = 1.0 if max_mdd <= 0.40 else max(0.05, 1.0 - (max_mdd - 0.40) * 2.0)

        return float(score_consistente * factor_suelo)
    except Exception as e:
        if not isinstance(e, optuna.exceptions.TrialPruned):
            tqdm.write(f" [!] Trial {trial.number} Inviable: {e}")
        raise optuna.exceptions.TrialPruned()
    finally:
        pbar_sub.close()

class _TqdmCallback:
    def __init__(self, pbar):
        self.pbar = pbar
        self.mejor_score = -1e10
    def __call__(self, study, trial):
        if trial.value is not None:
            if trial.value > self.mejor_score:
                self.mejor_score = trial.value
            
            # Imprimir info detallada al completar un trial sin ser podado
            if trial.state.name == "COMPLETE":
                n1 = trial.params.get("n1", 16)
                n2_raw = trial.params.get("n2", 0)
                n2 = min(n2_raw, n1 - 8) if n2_raw > 0 else 0
                ocultas_str = str([x for x in [n1, n2] if x > 0])
                dropout    = trial.params.get('feature_dropout_rate', 0.25)
                train_size = trial.params.get('train_size', 252)
                tqdm.write(f" [✔] Trial {trial.number} | R2: {trial.value:+.4f} | Umbral: {trial.params.get('umbral_rebalanceo', 0.0):.3f} | Ocultas: {ocultas_str} | Train: {train_size}d | Dropout: {dropout:.2f}")
                
        self.pbar.set_description(f"🚀 FASE 2 | Trial {trial.number} | Max R2 Target: {self.mejor_score:+.4f}")
        self.pbar.update(1)



def run_fase_2():
    print("="*60)
    print(" 🚀 FASE 2: OPTIMIZACIÓN DE HIPERPARÁMETROS 🚀")
    print("="*60)
    get_datos_globales()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    estudio_db = "sqlite:///optuna_trading.db"
    study_name = "estudio_fase_2"
    
    if RESTART_STUDY:
        try: optuna.delete_study(study_name=study_name, storage=estudio_db)
        except: pass
    
    # Barra principal (Posición 0)
    pbar_main = tqdm(total=N_TRIALS, unit="trial", position=0)
    callback = _TqdmCallback(pbar_main)
    
    sampler = optuna.samplers.TPESampler(n_startup_trials=7)
    pruner  = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=1)

    study = optuna.create_study(study_name=study_name, storage=estudio_db, direction="maximize", sampler=sampler, pruner=pruner, load_if_exists=True)
    try:
        study.optimize(objective, n_trials=N_TRIALS, callbacks=[callback])
    except KeyboardInterrupt:
        pass
    finally:
        pbar_main.close()
        
    if len(study.trials) > 0:
        best_par = study.best_params
        best_val = study.best_value
        print(f"\n🏆 MEJOR CONFIGURACIÓN FASE 2: {best_val:+.4f}")
        for k, v in best_par.items(): print(f"  -> {k}: {v}")
        # Reconstruir ocultas a partir de n1/n2
        n1 = best_par.get("n1", 16)
        n2_raw = best_par.get("n2", 0)
        n2 = min(n2_raw, n1 - 8) if n2_raw > 0 else 0
        best_par["ocultas"] = [x for x in [n1, n2] if x > 0]

        # Eliminar claves de arquitectura cruda (no van al config)
        for k in ["n1", "n2", "n3", "neuronas_capa"]:
            best_par.pop(k, None)

        # Asegurar que feature_dropout_rate queda en el state
        if "feature_dropout_rate" not in best_par:
            best_par["feature_dropout_rate"] = 0.25

        guardar_pipeline_state(best_par, "fase_2")
        print(f"[*] Parámetros optimizados de fase_2 guardados en config.json")


if __name__ == "__main__":
    run_fase_2()
