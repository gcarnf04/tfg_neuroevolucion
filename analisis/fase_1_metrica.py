import optuna
import numpy as np
import time
import os
import torch
import gc
import json
from copy import deepcopy
from tqdm import tqdm

import importlib
import threading
import sys
from functools import partial
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from motor_ga import evaluar_poblacion_mps, obtener_predicciones_mps, ModeloComiteTorch

from modelo.evolucion_comite import EvolucionGenetica

from aux.datos import GestorDatos
from aux.funciones_guardado import guardar_pipeline_state
from config import CONFIG as GLOBAL_CONFIG, get_fase_config
CONFIG = deepcopy(GLOBAL_CONFIG)
_f1_cfg = get_fase_config("FASE_1")
for k, v in _f1_cfg.items():
    CONFIG[k] = v

# --- PARÁMETROS DE LA FASE 1 (dinámicos desde config.json) ---
N_TRIALS = CONFIG.get('N_TRIALS', 500)
N_FINAL_MODELS = CONFIG.get('N_FINAL_MODELS', 10000)
RESTART_STUDY = True

_DIR_SCRIPT = os.path.dirname(os.path.abspath(__file__))
_DIR_LABORATORIO = os.path.dirname(_DIR_SCRIPT) if 'analisis' in _DIR_SCRIPT else _DIR_SCRIPT

# --- Utilidades Matemáticas en GPU/MPS ---
def spearman_correlation_torch(x, y):
    """Calcula la Correlación de Spearman directamente en GPU/MPS."""
    rank_x = x.argsort().argsort().float()
    rank_y = y.argsort().argsort().float()
    centered_x = rank_x - rank_x.mean()
    centered_y = rank_y - rank_y.mean()
    cov = (centered_x * centered_y).sum()
    norm = torch.sqrt((centered_x**2).sum()) * torch.sqrt((centered_y**2).sum())
    return cov / (norm + 1e-9)

X_global, R_global, idx_c_global, idx_m_global, idx_l_global = None, None, None, None, None
DATASET_FASE_1 = None
_dataset_lock = threading.Lock()

def get_datos_globales():
    global X_global, R_global, idx_c_global, idx_m_global, idx_l_global
    if X_global is None:
        print("[*] Preparando datos globales...")
        fecha_fin = CONFIG.get('fecha_fin')
        if not fecha_fin:
            fecha_fin = time.strftime("%Y-%m-%d")
            CONFIG['fecha_fin'] = fecha_fin
            print(f"[*] Fecha de fin dinámica establecida a hoy: {fecha_fin}")
        else:
            print(f"[*] Usando fecha de fin persistente: {fecha_fin}")
            
        gestor = GestorDatos(CONFIG['tickers'], CONFIG['ticker_cash'], CONFIG['ticker_macro'], CONFIG['fecha_inicio'], fecha_fin)
        X_global, R_global, idx_c_global, idx_m_global, idx_l_global = gestor.obtener_datos_listos(escalar_global=False)
    return X_global, R_global, idx_c_global, idx_m_global, idx_l_global

def preparar_dataset_fase_1(config_base, local_device, idx_c, idx_m, idx_l, n_params, ocultas):
    global DATASET_FASE_1
    if DATASET_FASE_1 is not None:
        return DATASET_FASE_1
    with _dataset_lock:
        if DATASET_FASE_1 is not None:
            return DATASET_FASE_1
    
        X, R, _, _, _ = get_datos_globales()
        _f1 = get_fase_config("FASE_1")
        # ALPHA v6.15: Forzamos que las ventanas OOS coincidan con el step_size real
        step_size = config_base.get('step_size', 63)
        dias_por_ventana = step_size
        n_ventanas = _f1.get('N_VENTANAS', 15) # Aumentamos ventanas para tener más datos de 63 días
        train_size = config_base.get('train_size', 252)
        
        def eval_local(population, X_t, R_t, is_val=False, batch_size=5000, silent=True):
            rets_list, turnover_med_list, l2_pen_list = [], [], []
            config_local = deepcopy(config_base)
            config_local['ocultas'] = ocultas
            
            # Solo mostramos barra si no es silent
            iterator = range(0, len(population), batch_size)
            if not silent:
                iterator = tqdm(iterator, desc="🚀 Eval MPS", leave=False)
                
            for i in iterator:
                chunk = population[i:i+batch_size]
                r_net, t_med, l2_p = evaluar_poblacion_mps(chunk, X_t, R_t, config_local, local_device, None, idx_c, idx_m, idx_l, return_raw=True)
                rets_list.append(r_net.cpu()); l2_pen_list.append(l2_p.cpu())
                if not is_val: turnover_med_list.append(t_med.cpu())
                
            if local_device.type == 'mps': torch.mps.empty_cache(); gc.collect()
            return (torch.cat(rets_list, dim=0).to(local_device), 
                    torch.cat(turnover_med_list, dim=0).to(local_device) if not is_val else None, 
                    torch.cat(l2_pen_list, dim=0).to(local_device))

        # --- ALPHA v6.6: POOL DIVERSO ---
        print(f"[*] Generando pool diverso de {N_FINAL_MODELS} modelos...")
        dias_val_total = n_ventanas * dias_por_ventana
        n_tercio = N_FINAL_MODELS // 3
        
        # 1. Elite (Evolución dirigida)
        idx_fin_is_pool = len(R) - dias_val_total
        X_pool_t = torch.tensor(X[:idx_fin_is_pool], dtype=torch.float32, device=local_device)
        R_pool_t = torch.tensor(R[:idx_fin_is_pool], dtype=torch.float32, device=local_device)
        
        n_p = n_params if n_params else 100
        pop_elite = (torch.rand((n_tercio * 3, n_p), device=local_device) * 2.0) - 1.0
        
        pbar_elite = tqdm(range(10), desc="🧬 Evolucionando Elite")
        for _ in pbar_elite:
            rets_g, _, _ = eval_local(pop_elite, X_pool_t[-504:], R_pool_t[-504:], batch_size=2000, silent=True)
            # Fitness alineado con paradigma (v6.8): Calmar con target saturado
            n_dias = rets_g.shape[1]
            ret_anual_g = (torch.clamp(torch.prod(1.0 + rets_g, dim=1), min=1e-6) ** (252.0 / n_dias)) - 1.0
            cap_g = torch.cumprod(1.0 + rets_g, dim=1)
            max_cap_g, _ = torch.cummax(cap_g, dim=1)
            mdd_g = torch.max(torch.abs((max_cap_g - cap_g) / (max_cap_g + 1e-9)), dim=1)[0]
            target_f1 = config_base.get('target_return_anual', 0.12)
            exceso_g = torch.clamp(ret_anual_g - target_f1, min=0.0)
            ret_util_g = torch.minimum(ret_anual_g, torch.tensor(target_f1, device=ret_anual_g.device)) + exceso_g * 0.1
            fits = ret_util_g / (mdd_g + 1e-9)
            
            _, top_idx = torch.topk(fits, n_tercio)
            elite_sub = pop_elite[top_idx]
            hijos = elite_sub + torch.randn_like(elite_sub) * 0.05
            pop_elite = torch.cat([elite_sub, hijos, 
                                   (torch.rand((n_tercio, n_p), device=local_device)*2-1)], dim=0)

        rets_g_final, _, _ = eval_local(pop_elite, X_pool_t[-504:], R_pool_t[-504:], batch_size=2000, silent=True)
        # Fitness final alineado con paradigma
        ret_anual_g_f = (torch.clamp(torch.prod(1.0 + rets_g_final, dim=1), min=1e-6) ** (252.0 / rets_g_final.shape[1])) - 1.0
        cap_g_f = torch.cumprod(1.0 + rets_g_final, dim=1)
        max_cap_g_f, _ = torch.cummax(cap_g_f, dim=1)
        mdd_g_f = torch.max(torch.abs((max_cap_g_f - cap_g_f) / (max_cap_g_f + 1e-9)), dim=1)[0]
        exceso_g_f = torch.clamp(ret_anual_g_f - target_f1, min=0.0)
        ret_util_g_f = torch.minimum(ret_anual_g_f, torch.tensor(target_f1, device=ret_anual_g_f.device)) + exceso_g_f * 0.1
        fits_final = ret_util_g_f / (mdd_g_f + 1e-9)
        _, top_idx_final = torch.topk(fits_final, n_tercio)
        pop_elite = pop_elite[top_idx_final]
            
        # 2. Random
        pop_rand = (torch.rand((n_tercio, n_p), device=local_device) * 2.0) - 1.0
        
        # 3. Anti-Elite (Selección de los peores)
        print("[*] Seleccionando Anti-Elite...")
        pop_anti_pool = (torch.rand((n_tercio * 5, n_p), device=local_device) * 2.0) - 1.0
        rets_a, _, _ = eval_local(pop_anti_pool, X_pool_t[-504:], R_pool_t[-504:], batch_size=2000, silent=False)
        # Fitness alineado con paradigma (v6.8): Calmar con target saturado para anti-elite
        n_dias_a = rets_a.shape[1]
        ret_anual_a = (torch.clamp(torch.prod(1.0 + rets_a, dim=1), min=1e-6) ** (252.0 / n_dias_a)) - 1.0
        cap_a = torch.cumprod(1.0 + rets_a, dim=1)
        max_cap_a, _ = torch.cummax(cap_a, dim=1)
        mdd_a = torch.max(torch.abs((max_cap_a - cap_a) / (max_cap_a + 1e-9)), dim=1)[0]
        exceso_a = torch.clamp(ret_anual_a - target_f1, min=0.0)
        ret_util_a = torch.minimum(ret_anual_a, torch.tensor(target_f1, device=ret_anual_a.device)) + exceso_a * 0.1
        fits_a = ret_util_a / (mdd_a + 1e-9)
        _, bot_idx = torch.topk(fits_a, n_tercio, largest=False)
        pop_anti = pop_anti_pool[bot_idx]
        
        pop_total = torch.cat([pop_elite, pop_rand, pop_anti], dim=0)
        l2_global = torch.norm(pop_total, p=2, dim=1) / (pop_total.shape[1]**0.5)

        datasets_ventanas = []
        pbar_windows = tqdm(range(n_ventanas), desc="🗄️ Preparando Ventanas IS/OOS")
        for i in pbar_windows:
            idx_v_start = len(R) - (dias_val_total - (i * dias_por_ventana))
            idx_v_end   = idx_v_start + dias_por_ventana
            idx_is_start = idx_v_start - train_size
            
            # Normalización Local (IS -> OOS)
            X_is_raw = X[idx_is_start:idx_v_start]
            mean_is = np.mean(X_is_raw, axis=0)
            std_is  = np.std(X_is_raw, axis=0) + 1e-9
            
            X_is_norm = (X_is_raw - mean_is) / std_is
            X_oos_norm = (X[idx_v_start:idx_v_end] - mean_is) / std_is
            
            X_is_t = torch.tensor(X_is_norm, dtype=torch.float32, device=local_device)
            R_is_t = torch.tensor(R[idx_is_start:idx_v_start], dtype=torch.float32, device=local_device)
            X_oos_t = torch.tensor(X_oos_norm, dtype=torch.float32, device=local_device)
            R_oos_t = torch.tensor(R[idx_v_start:idx_v_end], dtype=torch.float32, device=local_device)
            
            # Evaluación Simétrica (Silent para no ensuciar el pbar_windows)
            rets_is, turn_is, _ = eval_local(pop_total, X_is_t, R_is_t, is_val=False, batch_size=2000, silent=True)
            rets_oos, turn_oos, _ = eval_local(pop_total, X_oos_t, R_oos_t, is_val=False, batch_size=2000, silent=True)
            
            datasets_ventanas.append({
                'rets_is': rets_is, 'rets_spy_is': R_is_t[1:, 0], 'turnover_is': turn_is,
                'rets_oos': rets_oos, 'rets_spy_oos': R_oos_t[1:, 0], 'turnover_oos': turn_oos,
                'l2_penalty': l2_global
            })
            
        print(f"[✔] Dataset V6.6 listo ({len(pop_total)} modelos x {n_ventanas} ventanas).")
        DATASET_FASE_1 = datasets_ventanas
        return DATASET_FASE_1

def objective(trial, neuronas=None):
    X, R, idx_c, idx_m, idx_l = get_datos_globales()
    config_trial = deepcopy(CONFIG)

    # ── Optimización 4D: Objetivo Bajo MDD + Retorno Consistente (v7.0) ──
    # Objetivo: maximizar retorno real con mínimo drawdown y volatilidad bajista.
    # Linealidad ya no es dominante — sirve solo para suavizar la curva moderadamente.
    wlin = trial.suggest_float("w_linealidad", 0.0, 0.35)
    wr   = trial.suggest_float("w_ret", 0.1, 0.5)
    wm   = trial.suggest_float("w_mdd", 0.2, 0.6)   # MDD prioritario
    wso  = trial.suggest_float("w_sortino", 0.0, 0.25)  # Penaliza vol bajista
    ws   = trial.suggest_float("w_sharpe_dif", 0.0, 0.30)  # Sharpe diferencial activo
    
    total = wr + wm + ws + wso + wlin + 1e-9
    config_trial['w_ret'] = wr / total
    config_trial['w_mdd'] = wm / total
    config_trial['w_sharpe_dif'] = ws / total
    config_trial['w_sortino'] = wso / total
    config_trial['w_linealidad'] = wlin / total
    
    config_trial['w_l2'] = trial.suggest_float("w_l2", 0.0, 0.1)
    config_trial['w_turnover'] = trial.suggest_float("w_turnover", 0.0, 0.5)

    # Pesos legacy silenciados
    for k in ['w_cobardia', 'w_dominancia', 'w_decorrel', 'w_oportunidad']: config_trial[k] = 0.0

    metrica_name = config_trial['metrica']
    metrica_modulo = importlib.import_module(f"modelo.{metrica_name}")
    calc_fitness_fn = getattr(metrica_modulo, 'calcular_fitness_torch')
    local_device = torch.device("mps") if torch.backends.mps.is_available() else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    
    ocultas = neuronas if neuronas else config_trial.get('ocultas', [64, 8])
    motor_temp = EvolucionGenetica(idx_c, idx_m, idx_l, len(CONFIG['tickers'])+1, 1, local_device, ocultas=ocultas)
    
    ds_list = preparar_dataset_fase_1(config_trial, local_device, idx_c, idx_m, idx_l, motor_temp.n_params, ocultas)
    
    # v7.0: Objetivo directo — Calmar OOS de los modelos que quedan top IS
    # "¿Los pesos de métrica seleccionan modelos que realmente son rentables y seguros OOS?"
    calmar_ventanas = []
    for ds in ds_list:
        # 1. Fitness IS con los pesos del trial → ranking IS
        fits_is = calc_fitness_fn(
            ds['rets_is'], ds['rets_spy_is'],
            w_ret=config_trial['w_ret'], w_mdd=config_trial['w_mdd'],
            w_sharpe_dif=config_trial['w_sharpe_dif'], w_sortino=config_trial['w_sortino'],
            w_linealidad=config_trial.get('w_linealidad', 0.0),
            w_l2=config_trial['w_l2'], w_turnover=config_trial['w_turnover'],
            l2_penalty=ds['l2_penalty'], turnover_medio=ds['turnover_is']
        )

        # 2. Seleccionar top 10% según fitness IS
        n_total = fits_is.shape[0]
        k = max(10, n_total // 10)
        top_idx = torch.topk(fits_is, k).indices

        # 3. Medir el Calmar REAL de esos modelos en OOS (sin función de fitness intermedia)
        rets_oos_top = ds['rets_oos'][top_idx]  # (K, T_oos)
        T_oos = rets_oos_top.shape[1]
        if T_oos < 5:
            continue

        # Ajuste Transaccional OOS (10 bps por unidad de turnover diario)
        friccion_diaria = ds['turnover_oos'][top_idx] * config_trial.get('base_spread', 0.0005)
        rets_oos_top_ajustado = rets_oos_top - friccion_diaria.unsqueeze(1)

        ret_acum = torch.prod(1.0 + rets_oos_top_ajustado, dim=1)
        ret_anual = (torch.clamp(ret_acum, min=1e-6) ** (252.0 / T_oos)) - 1.0
        
        # Saturacion suave concava a partir del 15% de retorno
        umbral_sat = 0.15
        ret_util = torch.where(
            ret_anual <= umbral_sat,
            ret_anual,
            umbral_sat + umbral_sat * torch.log(1.0 + torch.clamp(ret_anual - umbral_sat, min=0.0) / umbral_sat)
        )

        cap         = torch.cumprod(1.0 + rets_oos_top_ajustado, dim=1)
        max_cap, _  = torch.cummax(cap, dim=1)
        mdd         = torch.max(torch.abs((max_cap - cap) / (max_cap + 1e-9)), dim=1)[0]

        calmar = torch.where(
            ret_util >= 0,
            ret_util / (mdd + 1e-9),
            ret_util * (1.0 + mdd)
        )
        
        # Calmar continuo con saturacion suave superior y sin clampado estricto
        calmar_util = torch.where(
            calmar <= 3.0,
            torch.where(calmar < -5.0, -5.0 + 0.5 * (calmar + 5.0), calmar),
            3.0 + 2.0 * torch.log(1.0 + torch.clamp(calmar - 3.0, min=0.0) / 2.0)
        )
        
        calmar_ventanas.append(float(calmar_util.mean().cpu()))

    if not calmar_ventanas: return -1.0
    calmar_arr = np.array(calmar_ventanas)
    # Objetivo: Calmar medio alto con poca varianza entre regímenes (consistencia real)
    score_final = float(calmar_arr.mean()) - 0.5 * float(calmar_arr.std())
    return score_final

class _TqdmCallback:
    def __init__(self, pbar):
        self.pbar = pbar
        self.mejor_score = -1e10
    def __call__(self, study, trial):
        if trial.value is not None and trial.value > self.mejor_score:
            self.mejor_score = trial.value
        params = trial.params
        desc = (f"T{trial.number:02d} | Score: {trial.value:+.4f} | ★: {self.mejor_score:+.4f} | "
                f"w_ret={params.get('w_ret', 0):.2f} w_mdd={params.get('w_mdd', 0):.2f}")
        self.pbar.set_description(desc)
        self.pbar.update(1)

def run_optimization():
    print("="*60)
    print(" 🚀 FASE 1: BÚSQUEDA DE PESOS ÓPTIMOS DE MÉTRICA 🚀")
    print("="*60)
    get_datos_globales()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    # Arquitectura base desde OPTIMIZABLES (punto de partida de la búsqueda)
    arch_base = CONFIG.get('ocultas', [32, 16])
    arquitecturas = [arch_base]
    mejores_resultados = {}
    absoluto_best_params = {}
    absoluto_best_score = -1.0
    estudio_db = "sqlite:///optuna_trading.db"
    
    for neuronas in arquitecturas:
        arch_name = "_".join(map(str, neuronas)) if isinstance(neuronas, list) else str(neuronas)
        print(f"\n[!] Probando arquitectura: {arch_name} neuronas")
        
        # Reset de caché para nueva arquitectura
        global DATASET_FASE_1
        DATASET_FASE_1 = None
        
        study_name = f"estudio_fase_1_{arch_name}n"
        
        try: optuna.delete_study(study_name=study_name, storage=estudio_db)
        except: pass
        
        sampler = optuna.samplers.TPESampler(n_startup_trials=20)
        study = optuna.create_study(study_name=study_name, storage=estudio_db, direction="maximize", sampler=sampler)
        pbar = tqdm(total=N_TRIALS, unit="trial", leave=False)
        callback = _TqdmCallback(pbar)
        
        try:
            # Inyectar la arquitectura actual en el objective
            obj_func = partial(objective, neuronas=neuronas)
            study.optimize(obj_func, n_trials=N_TRIALS, callbacks=[callback])
        except KeyboardInterrupt:
            break
        finally:
            pbar.close()
            
        if len(study.trials) > 0:
            score = study.best_value
            mejores_resultados[tuple(neuronas) if isinstance(neuronas, list) else neuronas] = score
            print(f"[*] Fin {arch_name}N -> Max Score: {score:+.4f}")
            
            if score > absoluto_best_score:
                absoluto_best_score = score
                absoluto_best_params = study.best_params
                if isinstance(neuronas, list):
                    absoluto_best_params['ocultas'] = neuronas

    
    print("\n🏆 RESUMEN DE ARQUITECTURAS:")
    for n, s in mejores_resultados.items():
        print(f"  -> {n} Neuronas: {s:+.4f}")
        
    if absoluto_best_score > -1.0:
        best_arch = absoluto_best_params.get('ocultas', absoluto_best_params.get('neuronas_capa'))
        print(f"\n⭐ GANADOR: {best_arch} neuronas con Score {absoluto_best_score:+.4f}")
        
        config_final = deepcopy(CONFIG)
        config_final.update(absoluto_best_params)
        
        bp = absoluto_best_params
        wr  = bp.get('w_ret', 0.2)
        wm  = bp.get('w_mdd', 0.5)
        ws  = bp.get('w_sharpe_dif', 0.0)
        wso = bp.get('w_sortino', 0.1)
        wlin = bp.get('w_linealidad', 0.2)
        total = wr + wm + ws + wso + wlin + 1e-9
        
        config_final['w_ret']        = round(wr  / total, 4)
        config_final['w_mdd']        = round(wm  / total, 4)
        config_final['w_sharpe_dif'] = round(ws  / total, 4)
        config_final['w_linealidad'] = round(wlin / total, 4)
        config_final['w_sortino']    = round(wso / total, 4)
        config_final['w_l2']         = round(bp.get('w_l2', 0.05), 4)
        config_final['w_turnover']   = round(bp.get('w_turnover', 0.3), 4)
        
        config_final['w_cobardia'] = 0.0
        config_final['w_dominancia'] = 0.0
        config_final['w_decorrel'] = 0.0
        config_final['w_oportunidad'] = 0.0
        
        # Eliminar las claves _raw para no contaminar el state
        for k in ['w_ret_raw', 'w_mdd_raw', 'w_cob_raw', 'w_dom_raw', 'w_dec_raw', 'w_shd_raw', 'w_opp_raw']:
            config_final.pop(k, None)
            
        print(f"[*] Parámetros de fitness a guardar: "
              f"w_ret={config_final.get('w_ret'):.3f}, "
              f"w_mdd={config_final.get('w_mdd'):.3f}, "
              f"w_sharpe_dif={config_final.get('w_sharpe_dif'):.3f}, "
              f"w_sortino={config_final.get('w_sortino'):.3f}, "
              f"w_linealidad={config_final.get('w_linealidad', 0.0):.3f}")

        # Eliminar claves _raw si existiesen de ejecuciones anteriores
        for k in ['w_ret_raw', 'w_mdd_raw', 'w_cob_raw', 'w_dom_raw', 'w_dec_raw', 'w_shd_raw', 'w_opp_raw']:
            config_final.pop(k, None)
            
        # Forzar excelencia individual para asegurar que la Fase 2 optimice sobre la misma base
        config_final['n_mejores'] = 1
        
        guardar_pipeline_state(config_final, "fase_1")

if __name__ == "__main__":
    run_optimization()
