import torch
import gc
import numpy as np
import os
import time
import json
import pandas as pd
from copy import deepcopy
from tqdm import tqdm
from scipy import stats

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aux.datos import GestorDatos
from aux.funciones_guardado import guardar_pipeline_state
from config import CONFIG, get_fase_config
from motor_ga import entrenar_ventana
from modelo.modelo_comite import ModeloComite
from modelo.inferencia import ensamblar_y_predecir, simular_trading_vectorizado, simular_trading_batch
from aux.slippage import calcular_slippage_dinamico

_DIR_SCRIPT = os.path.dirname(os.path.abspath(__file__))
_DIR_LABORATORIO = os.path.dirname(_DIR_SCRIPT) if 'analisis' in _DIR_SCRIPT else _DIR_SCRIPT

# ── CONFIGURACIÓN BOOTSTRAP (Sincronizado con CONFIG) ────────
# Los valores se leen dinámicamente dentro de run_fase_3 para evitar stale data.

def _detectar_regimen_ventana(R_tr, R_oos, config):
    """
    Clasifica el régimen de mercado de la ventana OOS.
    Returns: dict con 'tipo' (bull/bear/lateral/crisis), 
             'spy_ret_anual_tr', 'spy_vol_tr', 'spy_ret_oos'
    """
    spy_tr  = R_tr[:, 0]
    spy_oos = R_oos[1:, 0]
    
    ret_tr_anual = (np.prod(1 + spy_tr) ** (252 / max(len(spy_tr), 1))) - 1
    vol_tr       = np.std(spy_tr) * np.sqrt(252)
    ret_oos      = np.prod(1 + spy_oos) - 1
    
    if ret_oos > 0.05: tipo = "bull_fuerte"
    elif ret_oos > 0.01: tipo = "bull_moderado"
    elif ret_oos > -0.03: tipo = "lateral"
    elif ret_oos > -0.10: tipo = "bear_moderado"
    else: tipo = "crisis"
    
    return {
        'tipo': tipo,
        'spy_ret_anual_tr': round(float(ret_tr_anual), 4),
        'spy_vol_anual_tr': round(float(vol_tr), 4),
        'spy_ret_oos':      round(float(ret_oos), 4),
    }

def _calcular_entropia_prediccion(logits_pool, idx_seleccionados):
    logits_comite = logits_pool[idx_seleccionados].mean(axis=0)  
    exp_l = np.exp(logits_comite - logits_comite.max(axis=1, keepdims=True))
    probs = exp_l / exp_l.sum(axis=1, keepdims=True)
    # Entropía media diaria normalizada (0=máxima concentración, 1=máxima diversificación)
    n_activos = probs.shape[1]
    entropia = -np.sum(probs * np.log(probs + 1e-9), axis=1) / np.log(n_activos)
    return float(entropia.mean())


# ═══════════════════════════════════════════════════════════════════
# UTILS: DIAGNÓSTICO Y MÉTRICAS
# ═══════════════════════════════════════════════════════════════════

def calcular_fechas_ventana(idx_tr_start, config, fechas_index):
    T_size = config['train_size']
    S_size = config['step_size']
    d_tr_start = fechas_index[idx_tr_start].strftime("%Y-%m-%d")
    idx_oos_start = idx_tr_start + T_size
    idx_oos_end   = min(idx_oos_start + S_size - 1, len(fechas_index) - 1)
    d_oos_start = fechas_index[idx_oos_start].strftime("%Y-%m-%d")
    d_oos_end   = fechas_index[idx_oos_end].strftime("%Y-%m-%d")
    return d_tr_start, None, d_oos_start, d_oos_end

def calcular_metricas_oos(rets, rets_spy):
    rets = np.array(rets)
    rets_spy = np.array(rets_spy)
    if len(rets) == 0: return 0, 0, 0, 0, 0, 0, 0, 0
    
    ret_acu = np.prod(1 + rets) - 1
    ret_acu_spy = np.prod(1 + rets_spy) - 1
    
    # Retorno Anualizado (asumiendo 252 días de trading)
    n_days = len(rets)
    ret_anualizado = (max(1.0 + ret_acu, 1e-6) ** (252.0 / n_days)) - 1.0
    
    # Cálculo de Maximum Drawdown
    c_ia = np.cumprod(1.0 + rets)
    m_cap = np.maximum.accumulate(c_ia)
    mdd = np.max(np.abs((m_cap - c_ia) / (m_cap + 1e-9)))
    
    alpha_absoluto = ret_acu - ret_acu_spy
    alpha_pen = alpha_absoluto
    if mdd > 0.25:
        alpha_pen -= (mdd - 0.25) * 5.0
        
    # --- NUEVAS MÉTRICAS INSTITUCIONALES ---
    volatilidad = np.std(rets) * np.sqrt(252)
    sharpe = (np.mean(rets) * 252) / (volatilidad + 1e-9)
    
    rets_negativos = rets[rets < 0]
    vol_negativa = np.std(rets_negativos) * np.sqrt(252) if len(rets_negativos) > 0 else 1e-9
    sortino = (np.mean(rets) * 252) / (vol_negativa + 1e-9)
    
    calmar = ret_anualizado / (mdd + 1e-9)
    
    exceso_rets = rets - rets_spy
    tracking_error = np.std(exceso_rets) * np.sqrt(252)
    ir = (np.mean(exceso_rets) * 252) / (tracking_error + 1e-9)
    
    # Retorno Mensual Compuesto (asumiendo 21 días de trading por mes)
    ret_mensual = (max(1.0 + ret_acu, 1e-6) ** (21.0 / n_days)) - 1.0
    
    return ret_acu, mdd, alpha_pen, sharpe, sortino, calmar, ir, ret_mensual

def calcular_stats_montecarlo(rets_ia, rets_spy, iteraciones=1000):
    """
    P-Value via bootstrap de bloques vectorizado (v8.0 — sin loop Python).
    Testa si el Calmar es estadísticamente > 1 preservando autocorrelación mensual.
    """
    rets_ia = np.array(rets_ia)
    n = len(rets_ia)
    if n == 0: return 0.0, 0.0, 1.0

    ret_anual_ia = (max(np.prod(1 + rets_ia), 1e-9) ** (252.0 / n)) - 1.0
    vol_ia       = np.std(rets_ia) * np.sqrt(252) + 1e-9
    sharpe_real  = np.mean(rets_ia) * 252 / vol_ia

    # Bootstrap de bloques vectorizado — bloques de ~21 días para preservar autocorrelación
    tam_bloque    = 21
    n_bloques     = max(1, n // tam_bloque)
    bloques_inicio = np.arange(n_bloques) * tam_bloque
    idx_bloques   = np.random.randint(0, n_bloques, size=(iteraciones, n_bloques))

    # Construir matriz de índices (iteraciones, n_bloques * tam_bloque) → truncar a n
    block_starts = bloques_inicio[idx_bloques]                            # (iter, n_bloques)
    t_range      = np.arange(tam_bloque)
    idx_3d       = block_starts[:, :, None] + t_range[None, None, :]     # (iter, n_bloques, tam)
    idx_flat     = np.clip(idx_3d.reshape(iteraciones, -1), 0, n - 1)[:, :n]  # (iter, n)

    all_perms    = rets_ia[idx_flat]                                      # (iter, n)

    # Calmar vectorizado sobre todas las permutaciones a la vez
    caps         = np.cumprod(1.0 + all_perms, axis=1)
    max_caps     = np.maximum.accumulate(caps, axis=1)
    mdds         = np.max(np.abs((max_caps - caps) / (max_caps + 1e-9)), axis=1)
    ret_finals   = (np.clip(caps[:, -1], 1e-9, None) ** (252.0 / n)) - 1.0
    sharpe_sims  = ret_finals / (mdds + 1e-9)

    # Calmar Real
    cap_real   = np.cumprod(1 + rets_ia)
    max_real   = np.maximum.accumulate(cap_real)
    mdd_real   = np.max(np.abs((max_real - cap_real) / (max_real + 1e-9)))
    calmar_real = ret_anual_ia / (mdd_real + 1e-9)

    p_value = float(np.mean(sharpe_sims >= calmar_real))
    return ret_anual_ia, sharpe_real, p_value

@torch.no_grad()
def _precomputar_logits_pool_mps(adns_pob, X_oos_norm, config, idx_c, idx_m, idx_l):
    """
    Alpha v4.5: Calcula los logits de los modelos del pool usando MPS.
    Procesa los 30 modelos de golpe para máximo rendimiento.
    Retorna (N_pool, T, A) en NumPy.
    """
    from motor_ga import obtener_predicciones_mps
    local_device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    
    adns_t = torch.tensor(np.array(adns_pob), dtype=torch.float32, device=local_device)
    X_t    = torch.tensor(X_oos_norm, dtype=torch.float32, device=local_device)
    
    # obtener_predicciones_mps devuelve (N_pool, T, A) - pesos softmax
    pesos_mps = obtener_predicciones_mps(adns_t, X_t, config, local_device, idx_c, idx_m, idx_l)
    
    # ALPHA v6.16: Devolvemos los pesos reales (ya normalizados por ReLU+Linear)
    # No usamos logs porque ahora promediamos pesos directamente (Democracia Aritmética)
    return pesos_mps.cpu().numpy()


def _seleccionar_comite_por_diversidad(logits_pool, n):
    """
    Selección greedy de N modelos con máxima diversidad entre sí.
    Evita que el comité esté formado por clones del mismo régimen de mercado.
    
    logits_pool: (N_pool, T, A) — logits de todos los modelos del pool
    n: tamaño del comité a seleccionar
    
    Returns: índices de los N modelos seleccionados
    """
    N_pool = logits_pool.shape[0]
    if n >= N_pool:
        return list(range(N_pool))
    
    # Aplanar para calcular distancias: (N_pool, T*A)
    pool_flat = logits_pool.reshape(N_pool, -1)
    
    # --- OPTIMIZACIÓN v5.5: Vectorización Matricial de Distancias ---
    # ||a-b||² = ||a||² + ||b||² - 2a·b
    norms = np.einsum('ij,ij->i', pool_flat, pool_flat)
    dists = norms[:, None] + norms[None, :] - 2 * (pool_flat @ pool_flat.T)
    dists = np.maximum(dists, 0) # Estabilidad numérica

    seleccionados = [int(np.argmin(dists.sum(axis=1)))]
    
    for _ in range(n - 1):
        dist_al_comite = dists[:, seleccionados].min(axis=1)
        dist_al_comite[seleccionados] = -1.0
        # Añadir el más alejado (mayor diversidad)
        seleccionados.append(int(np.argmax(dist_al_comite)))
    
    return seleccionados


from multiprocessing import Pool, set_start_method
from functools import partial

def _procesar_ventana_f3(v_arg, X, R, idx_c, idx_m, idx_l, config, tamanos_comite, n_bootstraps, fechas_index):
    """Procesa una ventana completa de Fase 3. Diseñada para ejecución en proceso hijo."""
    # ALPHA v6.3: Prioridad baja para no bloquear el ordenador del usuario
    try:
        os.nice(10)
    except:
        pass
        
    idx_v, i_tr_l = v_arg

    
    _f3_cfg = get_fase_config("FASE_3")
    techo_exploracion = config.get('TECHO_EXPLORACION', _f3_cfg.get('TECHO_EXPLORACION', 1))
    config_w = config.copy()
    for k, v in _f3_cfg.items():
        config_w[k] = v
        
    config_w['n_comite'] = techo_exploracion
    config_w['n_mejores'] = techo_exploracion

    d_tr_s, _, d_oos_s, d_oos_e = calcular_fechas_ventana(i_tr_l, config_w, fechas_index)
    
    config_w_run = config_w.copy()
    config_w_run['silent'] = True
    # ALPHA v6.10: Pool fijo independiente de TECHO_EXPLORACION para velocidad constante
    # Ajustamos pool_fijo para que sea al menos el máximo de los comités que se quieren explorar
    max_comite_req = max(tamanos_comite) if len(tamanos_comite) > 0 else 1
    pool_fijo = max(config_w.get('pool_fijo_fundadores', 1), max_comite_req)
    config_w_run['n_fundadores'] = pool_fijo
    config_w_run['n_comite'] = pool_fijo
    config_w_run['n_mejores'] = pool_fijo

    res = entrenar_ventana((idx_v, i_tr_l, X, R, idx_c, idx_m, idx_l, config_w_run))
    
    # Feedback inmediato tras el entrenamiento — el bootstrap puede tardar con n_bootstraps alto
    print(f"\r   [W{idx_v:02d}] Entrenamiento OK (fit={res.get('max_fitness', 0):.3f}) — calculando bootstrap...", 
          end='', flush=True)



    f_tr  = i_tr_l + config['train_size']
    f_oos = f_tr   + config['step_size']
    X_oos, R_oos = X[f_tr:f_oos], R[f_tr:f_oos]
    X_oos_norm = (X_oos - res['stats_norm'][0]) / res['stats_norm'][1]

    spy_rets_oos = R_oos[1:, 0]
    spy_ret_total = float(np.prod(1 + spy_rets_oos) - 1)

    logits_pool = _precomputar_logits_pool_mps(
        res['pesos_ensemble'], X_oos_norm, config_w, idx_c, idx_m, idx_l
    )
    adns_pob  = res['pesos_ensemble']
    max_fit   = float(res['max_fitness'])
    diag_is   = res.get('diagnostico', {})

    # Slippage dinámico calculado internamente si corresponde

    config_oos = config.copy()
    config_oos['umbral_rebalanceo'] = config.get('umbral_rebalanceo', 0.10)

    n_pob_total   = len(adns_pob)
    max_n_comite  = min(max(tamanos_comite), n_pob_total)
    
    # Instanciar una sola vez fuera del bucle de tamaños para eficiencia
    inf_model = ModeloComite(
        idx_c, idx_m, idx_l,
        len(config['tickers']) + 1,
        ocultas=config.get('ocultas', [56])
    )

    # Vectorizado: una sola llamada en lugar de 10.000 iteraciones Python
    replace = n_pob_total < max_n_comite
    if replace:
        master_idx_matrix = np.random.randint(0, n_pob_total, size=(n_bootstraps, max_n_comite))
    else:
        # Sin reemplazo: permutar columnas — equivalente estadístico y mucho más rápido
        base = np.tile(np.arange(n_pob_total), (n_bootstraps, 1))
        idx_sort = np.random.rand(n_bootstraps, n_pob_total).argsort(axis=1)
        master_idx_matrix = base[np.arange(n_bootstraps)[:, None], idx_sort][:, :max_n_comite]


    # Resultados por tamaño de comité
    rets_por_n    = {}
    pesos_por_n   = {}
    prob_por_n    = {}
    alpha_wins    = {}
    # No serializamos retornos de bootstrap para optimización de RAM


    # === PASO 1 v8.0: retornos reales para todos los N (barato, O(N)) ===
    for n in tamanos_comite:
        if n > n_pob_total:
            continue
        adns_comite = adns_pob[:n]
        # ALPHA v6.16: Democracia de Pesos (Promedio Aritmético)
        p_top, _ = ensamblar_y_predecir(
            adns_comite, None, config_oos, inf_model,
            all_opinions_raw=logits_pool[:n]
        )
        rets_top       = simular_trading_vectorizado(p_top, R_oos, config_oos)
        rets_por_n[n]  = rets_top.tolist()
        pesos_por_n[n] = p_top[:-1].tolist()
        ret_top        = float(np.prod(1 + rets_top) - 1)
        alpha_wins[n]  = 1 if (ret_top - spy_ret_total) > 0 else 0

    # === PASO 2 v8.0: identificar top-2 N por Calmar real (sin bootstrap) ===
    def _calmar_rets(rets_list):
        rets = np.array(rets_list)
        if len(rets) < 2: return -1e9
        cap   = np.cumprod(1.0 + rets)
        max_c = np.maximum.accumulate(cap)
        mdd   = float(np.max(np.abs((max_c - cap) / (max_c + 1e-9))))
        ret_an = float(max(cap[-1], 1e-9) ** (252.0 / len(rets)) - 1.0)
        return ret_an / (mdd + 1e-9)

    ns_disponibles    = [n for n in tamanos_comite if n in rets_por_n]
    n_candidatos      = min(2, len(ns_disponibles))
    ns_para_bootstrap = set(
        sorted(ns_disponibles, key=lambda n: _calmar_rets(rets_por_n[n]), reverse=True)[:n_candidatos]
    )

    # === PASO 3 v8.0: bootstrap SOLO para los top-2 N (Opción C) ===
    for n in tamanos_comite:
        if n not in rets_por_n or n not in ns_para_bootstrap:
            prob_por_n[n] = 0.0
            continue

        # BOOTSTRAP: Promedio aritmético de los pesos de los modelos aleatorios
        idx_matrix    = master_idx_matrix[:, :n]
        p_batch       = logits_pool[idx_matrix].mean(axis=1).astype(np.float32)
        all_rets_boot = simular_trading_batch(p_batch, R_oos, config_oos)

        n_oos_dias      = all_rets_boot.shape[1]
        ret_anual_boots = (np.clip(np.prod(1 + all_rets_boot, axis=1), 1e-6, None) ** (252.0 / n_oos_dias)) - 1.0
        caps_boot       = np.cumprod(1 + all_rets_boot, axis=1)
        max_caps_boot   = np.maximum.accumulate(caps_boot, axis=1)
        mdd_boots       = np.max(np.abs((max_caps_boot - caps_boot) / (max_caps_boot + 1e-9)), axis=1)
        calmar_boots    = ret_anual_boots / (mdd_boots + 1e-9)
        prob_por_n[n]   = float(np.sum(calmar_boots > 1.0)) / n_bootstraps

    fechas_v = fechas_index[f_tr+1:f_oos].tolist()

    # ── EVOLUCIÓN POR GENERACIÓN (opcional) ─────────────────────────────────
    evolucion_generaciones = []
    historia = []
    if config.get('track_gen_evolution', False):
        historia = res.get('historia_generaciones', [])
        n_oos_dias = len(R_oos) - 1  # días OOS reales
        for g_idx, fit_IS, adn_np in historia:
            try:
                p_gen, _ = ensamblar_y_predecir(
                    [adn_np], X_oos_norm, config_oos, inf_model
                )
                rets_gen = simular_trading_vectorizado(p_gen, R_oos, config_oos)
                ret_an = float(max(np.prod(1 + rets_gen), 1e-9) ** (252.0 / max(len(rets_gen), 1)) - 1.0)
            except Exception:
                ret_an = float('nan')
            evolucion_generaciones.append({
                'gen': g_idx,
                'fit_IS': round(float(fit_IS), 5),
                'ret_anual_OOS': round(ret_an, 5),
            })

    print(f"\r   [W{idx_v:02d}] ✓ Completado                                              ", flush=True)
    return {

        'idx_v':                  idx_v,
        'spy_rets':               spy_rets_oos.tolist(),
        'fechas':                 fechas_v,
        'rets_por_n':             rets_por_n,
        # Eliminado rets_boot_por_n para ahorrar memoria
        'pesos_por_n':            pesos_por_n,
        'prob_por_n':             prob_por_n,
        'alpha_wins':             alpha_wins,
        'spy_ret_total':          spy_ret_total,
        'max_fit':                max_fit,
        'diag_is':                diag_is,
        'd_oos_s':                d_oos_s,
        'd_tr_s':                 d_tr_s,
        'd_oos_e':                d_oos_e,
        'evolucion_generaciones': evolucion_generaciones,
        'historia_generaciones_raw': historia,
    }


# ═══════════════════════════════════════════════════════════════════
# EXECUTION: FASE 3 WALK-FORWARD (BOOTSTRAP v3.7)
# ═══════════════════════════════════════════════════════════════════

def run_fase_3():
    print("\n" + "="*70)
    print(" 🚀 FASE 3: EVALUACIÓN WALK-FORWARD (Bootstrap v3.7) 🚀")
    print("="*70)
    
    config = deepcopy(CONFIG)
    _fase3_cfg = get_fase_config("FASE_3")
    for k, v in _fase3_cfg.items():
        config[k] = v
    techo_exploracion = _fase3_cfg.get('TECHO_EXPLORACION', 1)
    n_bootstraps      = _fase3_cfg.get('N_BOOTSTRAPS', 100)
    
    diagnostico_ventanas = []
    run_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    
    # Opción B v8.0: configurable desde FASE_3.TAMANOS_COMITE (evita explorar N redundantes)
    _tamanos_default = _fase3_cfg.get('TAMANOS_COMITE', [1, 5, 15, 20, 30])
    tamanos_comite   = sorted([n for n in _tamanos_default if n <= techo_exploracion])
    
    fecha_fin_wf = config.get('fecha_fin')
    if not fecha_fin_wf:
        fecha_fin_wf = time.strftime("%Y-%m-%d")
        config['fecha_fin'] = fecha_fin_wf
    gestor = GestorDatos(config['tickers'], config['ticker_cash'], config['ticker_macro'], config['fecha_inicio'], fecha_fin_wf)
    X, R, idx_c, idx_m, idx_l = gestor.obtener_datos_listos(escalar_global=False)
    print(f"\n[DEBUG WF] len(X)={len(X)}, train_size={config['train_size']}, step_size={config['step_size']}")
    fechas_index = gestor.features.index
    
    idx_limite = len(X)
    
    # Solo índices — X y R se cargan en cada worker para evitar serialización masiva
    ventanas_args = []
    idx_v = 0
    for i in range(0, idx_limite - config['train_size'] - config['step_size'] + 1, config['step_size']):
        ventanas_args.append((idx_v, i))
        idx_v += 1

    
    print(f"[*] Iniciando Simulación: {len(ventanas_args)} ventanas detectadas.")
    print(f"[*] Robustez: {techo_exploracion} modelos/vew + {n_bootstraps} muestreos aleatorios por N.")
    
    curvas_ia = {n: [] for n in tamanos_comite}
    prob_alpha_total = {n: [] for n in tamanos_comite}  
    alpha_wins_top = {n: 0 for n in tamanos_comite}
    historia_pesos = {n: [] for n in tamanos_comite}
    fechas_oos = []
    curva_spy = []
    evolucion_gens_raw = []  # lista de listas [{gen, fit_IS, ret_anual_OOS}, ...]

    print(f"[*] Procesando {len(ventanas_args)} ventanas en serie para estabilidad MPS...")
    print(f"[*] Nota: Se realiza limpieza de memoria GPU/RAM tras cada ventana.")

    resultados_raw = []
    for v_arg in tqdm(ventanas_args, desc="Walk-Forward"):
        res_v = _procesar_ventana_f3(
            v_arg,
            X=X,
            R=R,
            idx_c=idx_c,
            idx_m=idx_m,
            idx_l=idx_l,
            config=config,
            tamanos_comite=tamanos_comite,
            n_bootstraps=n_bootstraps,
            fechas_index=fechas_index
        )
        resultados_raw.append(res_v)

        
        # Limpieza inmediata de GPU/RAM tras cada ventana
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()
        
        # ALPHA v6.5: Feedback en tiempo real. Procesamos y mostramos cada ventana nada más terminar.
        idx_v      = res_v['idx_v']
        d_oos_s    = res_v['d_oos_s']
        max_fit    = res_v['max_fit']
        diag_is    = res_v['diag_is']

        curva_spy.extend(res_v['spy_rets'])
        fechas_oos.extend(res_v['fechas'])

        competidores_ventana = []
        for n in tamanos_comite:
            if n not in res_v['rets_por_n']:
                continue
            curvas_ia[n].extend(res_v['rets_por_n'][n])
            historia_pesos[n].append(np.array(res_v['pesos_por_n'][n]))
            prob_alpha_total[n].append(res_v['prob_por_n'][n])
            alpha_wins_top[n] += res_v['alpha_wins'].get(n, 0)

            ret_top_abs = float(np.prod(1 + np.array(res_v['rets_por_n'][n])) - 1)
            competidores_ventana.append((n, ret_top_abs, res_v['prob_por_n'][n]))

        best_top_n  = sorted(competidores_ventana, key=lambda x: x[1], reverse=True)[0]
        best_prob_n = sorted(competidores_ventana, key=lambda x: x[2], reverse=True)[0]
        gens_ejec   = diag_is.get('gen_final', config['generaciones'])
        tqdm.write(
            f" [W{idx_v:02d}] {d_oos_s} | Gens: {gens_ejec:3d} | Fit: {max_fit:5.3f} | "
            f"Ret.Abs (N={best_top_n[0]}): {best_top_n[1]*100:+5.1f}% | "
            f"P(Calmar>1): {best_prob_n[2]*100:4.1f}% (N={best_prob_n[0]})"
        )

        diagnostico_ventanas.append({
            'ventana_idx':       idx_v,
            'fecha_tr_inicio':   res_v['d_tr_s'],
            'fecha_oos_inicio':  d_oos_s,
            'fecha_oos_fin':     res_v['d_oos_e'],
            'is_max_fitness':    round(max_fit, 4),
            'is_n_fundadores':   diag_is.get('n_fundadores', 0),
            'oos_spy_ret':       round(res_v['spy_ret_total'], 4),
            'competidores': [
                {'n': n_v, 'ret_abs': round(float(a), 4), 'prob_calmar_gt1': round(float(p), 4)}
                for n_v, a, p in competidores_ventana
            ],
        })

        # Acumular datos de evolución por generación
        if res_v.get('evolucion_generaciones'):
            evolucion_gens_raw.append(res_v['evolucion_generaciones'])


    # 4. Tabla de Resultados Finales
    import sys
    from io import StringIO
    import datetime

    results_out = StringIO()
    class Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                if isinstance(obj, str): f.write(obj)
                else: f.write(str(obj))
                f.flush()
        def flush(self):
            for f in self.files: f.flush()

    original_stdout = sys.stdout
    sys.stdout = Tee(sys.stdout, results_out)
    
    try:
        print("\n" + "🏆"*5 + " RESULTADOS WALK-FORWARD — RETORNO ABSOLUTO " + "🏆"*5)
        print(f"{'Comité':<15} | {'Ret. Acum':<10} | {'Ret. Anual':<11} | {'Ret. Mens':<10} | {'Max DD':<8} | {'P-Value':<9}")
        print("-" * 80)

        metrics_by_n = {}
        results_summary = []

        for n in tamanos_comite:
            ret_acu, mdd, _, _, _, _, _, ret_mensual = calcular_metricas_oos(curvas_ia[n], curva_spy)
            ret_anual_ia, _, p_val = calcular_stats_montecarlo(curvas_ia[n], curva_spy)

            print(f"N={n:2d} modelos    | {ret_acu*100:9.2f}% | {ret_anual_ia*100:10.2f}% | {ret_mensual*100:9.2f}% | {mdd*100:7.2f}% | {p_val:9.4f}")

            current_metrics = {
                'Ret_Acum': ret_acu, 'Ret_Anual': ret_anual_ia, 'Max_DD': mdd,
                'Ret_Mens': ret_mensual, 'P_Value': p_val
            }
            metrics_by_n[n] = current_metrics
            results_summary.append((n, {'ret': ret_acu, 'mdd': mdd}))

        # ── MATRIZ DE ROBUSTEZ INSTITUCIONAL (P-Values via t-test Unilateral) ──
        # Colectar retornos anualizados independientes por ventana
        ret_anual_por_ventana_por_n = {n: [] for n in tamanos_comite}
        for res_v in resultados_raw:
            for n in tamanos_comite:
                if n not in res_v['rets_por_n']: continue
                rets = np.array(res_v['rets_por_n'][n])
                if len(rets) < 5: continue
                ret_anual = (max(np.prod(1 + rets), 1e-9) ** (252.0 / len(rets))) - 1.0
                ret_anual_por_ventana_por_n[n].append(ret_anual)

        objetivos = np.arange(0.01, 0.16, 0.01) # 1% al 15%
        robustez_por_n = {}
        
        # Calcular robustez para cada tamaño de comité
        for n in tamanos_comite:
            retornos_ventanas = ret_anual_por_ventana_por_n[n]
            n_v = len(retornos_ventanas)
            if not retornos_ventanas or n_v < 2:
                robustez_por_n[n] = {'max_obj': 0.0, 'p_val_obj': 1.0, 'p_val_1pct': 1.0}
                continue
                
            max_obj_val = 0.0
            p_val_at_max = 1.0
            p_val_at_1pct = 1.0
            
            for obj in objetivos:
                t_stat, p_val = stats.ttest_1samp(retornos_ventanas, popmean=obj)
                p_unilateral = p_val / 2 if t_stat > 0 else 1.0 - p_val / 2
                
                if abs(obj - 0.01) < 1e-9:
                    p_val_at_1pct = p_unilateral
                    
                if p_unilateral < 0.05:
                    max_obj_val = obj
                    p_val_at_max = p_unilateral
                    
            robustez_por_n[n] = {
                'max_obj': max_obj_val,
                'p_val_obj': p_val_at_max,
                'p_val_1pct': p_val_at_1pct
            }

        # JUEZ SUPREMO v3.0 (Mayor Retorno Anual con P-Value < 0.05 en t-test Unilateral de la Matriz)
        comites_validos = [n for n in tamanos_comite if robustez_por_n[n]['max_obj'] > 0.0]
        if comites_validos:
            # Mayor max_obj; ante empate, menor p-value en ese max_obj; ante empate, menor N (comité más simple)
            best_n = max(comites_validos, key=lambda n: (robustez_por_n[n]['max_obj'], -robustez_por_n[n]['p_val_obj'], -n))
            best_obj = robustez_por_n[best_n]['max_obj']
            best_p_val = robustez_por_n[best_n]['p_val_obj']
        else:
            # Fallback: menor p-value en el objetivo del 1%, y ante empate, menor N
            best_n = min(tamanos_comite, key=lambda n: (robustez_por_n[n]['p_val_1pct'], n))
            best_obj = 0.0
            best_p_val = robustez_por_n[best_n]['p_val_1pct']
            
        best_metrics = metrics_by_n[best_n]

        # Imprimir Benchmark SPY y Ganador
        ret_spy, mdd_spy, _, _, _, _, _, ret_mensual_spy = calcular_metricas_oos(curva_spy, curva_spy)
        ret_anual_spy_val = (max(np.prod(1 + np.array(curva_spy)), 1e-9) ** (252.0 / max(len(curva_spy), 1))) - 1.0
        print("-" * 80)
        print(f"Benchmark SPY   | {ret_spy*100:9.2f}% | {ret_anual_spy_val*100:10.2f}% | {ret_mensual_spy*100:9.2f}% | {mdd_spy*100:7.2f}% | {'--':>9}")
        print("-" * 80)
        print(f"⭐ GANADOR POR ROBUSTEZ: Comité de {best_n} modelos | Retorno Seguro: {best_obj*100:.2f}% | P-Value Matriz: {best_p_val:.4f}")

        # Imprimir la Matriz
        print("\n" + "="*80)
        print("📊 MATRIZ DE ROBUSTEZ INSTITUCIONAL (P-Values via t-test Unilateral)")
        print("   H0: La media de retornos OOS es INFERIOR o IGUAL al objetivo (menor es mejor / rechazo de H0)")
        print("="*80)
        
        header = "Objetivo | " + " | ".join([f"N={n:<3}" for n in tamanos_comite if ret_anual_por_ventana_por_n[n]])
        print(header)
        print("-" * len(header))
        
        for obj in objetivos:
            fila = f"  {obj*100:4.0f}%  | "
            vals = []
            for n in tamanos_comite:
                retornos_ventanas = ret_anual_por_ventana_por_n[n]
                n_v = len(retornos_ventanas)
                if not retornos_ventanas or n_v < 2: 
                    vals.append(" --  ")
                    continue
                
                # t-test unilateral
                t_stat, p_val = stats.ttest_1samp(retornos_ventanas, popmean=obj)
                p_unilateral = p_val / 2 if t_stat > 0 else 1.0 - p_val / 2
                
                marker = f"*" if p_unilateral < 0.05 else " "
                
                if n_v < 15:
                    vals.append(f"{marker}{p_unilateral:4.3f}~")
                else:
                    vals.append(f"{marker}{p_unilateral:4.3f}")
            print(fila + " | ".join(vals))
        print("="*80)
        
        if len(tamanos_comite) > 0 and tamanos_comite[0] in ret_anual_por_ventana_por_n:
            n_ventanas_usadas = len(ret_anual_por_ventana_por_n[tamanos_comite[0]])
            print(f"   (N ventanas para t-test: {n_ventanas_usadas}  |  ~ = potencia baja, <15 ventanas)")
        print("\n")



    finally:
        sys.stdout = original_stdout

    # Guardado de Ejecución en carpeta temporal con timestamp
    guardar_ejecucion = not config.get('_no_guardar_ejecucion', False)
    if not guardar_ejecucion:
        print("[*] Modo de prueba activo. Omitiendo guardado de ejecución en disco y registro en CSV.")

    if guardar_ejecucion:
        timestamp_folder = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        ejecucion_dir = os.path.join("ejecuciones", timestamp_folder)
        os.makedirs(ejecucion_dir, exist_ok=True)
        graficas_dir = os.path.join(ejecucion_dir, "graficas")
        os.makedirs(graficas_dir, exist_ok=True)
        
        with open(os.path.join(ejecucion_dir, "resultados.txt"), "w", encoding="utf-8") as f:
            f.write(results_out.getvalue())
            
        # Generar y guardar el gráfico de la matriz de p-values
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import seaborn as sns
            import pandas as pd
            
            p_matrix_data = []
            for obj in objetivos:
                row_data = {}
                for n in tamanos_comite:
                    retornos_ventanas = ret_anual_por_ventana_por_n[n]
                    n_v = len(retornos_ventanas)
                    if not retornos_ventanas or n_v < 2:
                        row_data[f"N={n}"] = np.nan
                    else:
                        t_stat, p_val = stats.ttest_1samp(retornos_ventanas, popmean=obj)
                        p_unilateral = p_val / 2 if t_stat > 0 else 1.0 - p_val / 2
                        row_data[f"N={n}"] = p_unilateral
                p_matrix_data.append(row_data)
            
            df_pvals = pd.DataFrame(p_matrix_data, index=[f"{int(obj*100)}%" for obj in objetivos])
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(df_pvals, annot=True, fmt=".4f", cmap='RdYlGn_r', vmin=0.0, vmax=0.20, cbar_kws={'label': 'p-value'})
            plt.title("Matriz de Robustez Institucional (P-Values)")
            plt.ylabel("Objetivo de Beneficio Anual")
            plt.xlabel("Número de Modelos en el Comité")
            plt.tight_layout()
            
            dest_img = os.path.join(graficas_dir, "matriz_robustez_pvalues.png")
            plt.savefig(dest_img, dpi=150)
            plt.close()
            print(f"[✔] Gráfico de robustez guardado en: {dest_img}")
        except Exception as e:
            print(f"[!] Error al generar gráfico de robustez: {e}")
            
        # Normalizar configuración para guardar exactamente lo usado sin sufijos de fase
        # Normalizar configuración para guardar exactamente lo usado sin sufijos de fase
        config_limpia = get_fase_config("FASE_3")
                
        # Eliminar cualquier parámetro con sufijo de fase redundante si lo hubiera
        for key in list(config_limpia.keys()):
            if any(key.endswith(suf) for suf in ['_f1', '_f2', '_f3']):
                config_limpia.pop(key, None)

        with open(os.path.join(ejecucion_dir, "config_ejecucion.json"), "w", encoding="utf-8") as f:
            json.dump(config_limpia, f, indent=2, ensure_ascii=False)
        
        print(f"[✔] Ejecución histórica guardada en {ejecucion_dir}")
        CONFIG['_last_ejecucion_dir'] = ejecucion_dir

    # ── GUARDADO EN CSV MAESTRO DE EJECUCIONES ──
    if guardar_ejecucion:
        try:
            import csv
            ruta_csv_maestro = os.path.join("ejecuciones", "registro_ejecuciones.csv")
            existe_csv = os.path.exists(ruta_csv_maestro)
            
            headers_nuevos = ['carpeta', 'Ret_Anual', 'n_mejores', 'P_Value', 'experimento', 'descripcion']
            
            # Migración automática si las cabeceras del CSV anterior difieren
            if existe_csv:
                try:
                    df_old = pd.read_csv(ruta_csv_maestro)
                    if list(df_old.columns) != headers_nuevos:
                        print("[*] Formato de CSV anterior detectado. Migrando registro_ejecuciones.csv...")
                        df_new = pd.DataFrame(columns=headers_nuevos)
                        df_new['carpeta'] = df_old.get('carpeta', [])
                        df_new['Ret_Anual'] = df_old.get('Ret_Anual', df_old.get('Ret_Acum', '0.00%'))
                        df_new['n_mejores'] = df_old.get('n_mejores', df_old.get('n_mejores', 1))
                        df_new['P_Value'] = df_old.get('P_Value', 1.0)
                        df_new['experimento'] = df_old.get('experimento', 'None')
                        df_new['descripcion'] = df_old.get('descripcion', 'None')
                        df_new.to_csv(ruta_csv_maestro, index=False)
                except Exception as e_migra:
                    print(f"[!] No se pudo migrar el CSV viejo: {e_migra}. Se creará uno nuevo.")
                    try:
                        os.remove(ruta_csv_maestro)
                        existe_csv = False
                    except:
                        pass
            
            fila_csv = {
                'carpeta': timestamp_folder,
                'Ret_Anual': f"{best_obj*100:.2f}%",
                'n_mejores': best_n,
                'P_Value': round(best_p_val, 4),
                'experimento': config.get('_nombre', 'None'),
                'descripcion': config.get('_descripcion', 'None')
            }
            
            with open(ruta_csv_maestro, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=headers_nuevos)
                if not existe_csv:
                    writer.writeheader()
                writer.writerow(fila_csv)
                
            # ── AUTO-ORDENADO POR P-VALUE (ASCENDENTE) ──
            try:
                df_maestro = pd.read_csv(ruta_csv_maestro)
                
                # Auxiliares numéricos para realizar ordenación exacta
                df_maestro['Ret_Anual_num'] = pd.to_numeric(df_maestro['Ret_Anual'].str.rstrip('%'), errors='coerce')
                df_maestro['P_Value_num']   = pd.to_numeric(df_maestro['P_Value'], errors='coerce')
                df_maestro['n_mejores_num'] = pd.to_numeric(df_maestro['n_mejores'], errors='coerce')
                
                # Ordenar: Rentabilidad (desc), P-Value (asc), n_mejores (asc)
                df_maestro = df_maestro.sort_values(
                    by=['Ret_Anual_num', 'P_Value_num', 'n_mejores_num'], 
                    ascending=[False, True, True]
                )
                
                df_maestro = df_maestro.drop(columns=['Ret_Anual_num', 'P_Value_num', 'n_mejores_num'])
                df_maestro.to_csv(ruta_csv_maestro, index=False)
                print(f"[✔] Registro CSV ordenado por Rentabilidad Anual (desc), P-Value (asc) y n_mejores (asc).")
            except Exception as e_sort:
                print(f"[!] No se pudo ordenar el CSV: {e_sort}")
    
        except Exception as e_csv:
            print(f"[!] Error al registrar en CSV maestro: {e_csv}")

    # ── GENERACIÓN DE GRÁFICO (OPCIÓN A) ──
    if guardar_ejecucion:
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            
            pesos_arr = np.concatenate(historia_pesos[best_n], axis=0)
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [1.5, 1]}, sharex=True)
            
            fechas = pd.to_datetime(fechas_oos)
            eq_spy = np.cumprod(1 + np.array(curva_spy)) * 100
            eq_ia = np.cumprod(1 + np.array(curvas_ia[best_n])) * 100
            
            # Subplot 1: Curva de Capital
            ax1.plot(fechas, eq_spy, label='SPY (Benchmark)', color='#555555', linewidth=1.5)
            ax1.plot(fechas, eq_ia, label=f'IA (Comité N={best_n})', color='#2ca02c', linewidth=2)
            ax1.fill_between(fechas, eq_spy, eq_ia, where=(eq_ia > eq_spy), interpolate=True, color='#2ca02c', alpha=0.2)
            ax1.fill_between(fechas, eq_spy, eq_ia, where=(eq_ia <= eq_spy), interpolate=True, color='#d62728', alpha=0.2)
            
            ax1.set_title(f"Simulación Walk-Forward — Retorno Absoluto | Comité N={best_n}", fontsize=14, fontweight='bold')
            ax1.set_ylabel("Crecimiento de Capital (%)", fontsize=12)
            ax1.legend(loc='upper left')
            ax1.grid(alpha=0.3)
            
            # Subplot 2: Asignación de Activos (Area Plot)
            activos = config['tickers'] + [config['ticker_cash']]
            # Manejo de colores dinámico para múltiples activos
            cmap = plt.cm.tab20 if len(activos) <= 21 else plt.cm.nipy_spectral
            colors = cmap(np.linspace(0, 1, len(activos)-1)).tolist() + ['#dddddd']
            ax2.stackplot(fechas, pesos_arr.T, labels=activos, colors=colors)
            ax2.set_ylabel("Asignación de Activos (%)", fontsize=12)
            
            # Ajuste dinámico de columnas de la leyenda según cantidad de activos
            n_cols = max(1, (len(activos) + 7) // 8)
            ax2.legend(loc='upper left', bbox_to_anchor=(1.01, 1), ncol=n_cols, fontsize=9, title="Activos")
            
            ax2.set_ylim(0, 1)
            ax2.grid(alpha=0.3)
            
            ax2.xaxis.set_major_locator(mdates.YearLocator())
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
            
            plt.tight_layout()
            plot_path = os.path.join(graficas_dir, "analisis_asignacion_oos.png")
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"[✔] Gráfico visual guardado en {plot_path}")
        except ImportError:
            print("[!] Matplotlib no está instalado. Omitiendo la generación del gráfico.")
        except Exception as e:
            print(f"[!] Error al generar el gráfico de asignación: {e}")

    # ── GRÁFICO DE EVOLUCIÓN POR GENERACIÓN ──
    if guardar_ejecucion and config.get('track_gen_evolution', False) and evolucion_gens_raw:
        try:
            import matplotlib.pyplot as plt

            # Alinear todas las ventanas al número mínimo de generaciones registradas
            min_gens = min(len(v) for v in evolucion_gens_raw)
            gens_idx    = [e['gen']           for e in evolucion_gens_raw[0][:min_gens]]
            ret_matrix  = np.array([[e['ret_anual_OOS'] for e in v[:min_gens]] for v in evolucion_gens_raw], dtype=float)
            fit_matrix  = np.array([[e['fit_IS']        for e in v[:min_gens]] for v in evolucion_gens_raw], dtype=float)

            # Ignorar NaN por ventanas con fallos
            ret_mean = np.nanmean(ret_matrix, axis=0)
            ret_std  = np.nanstd(ret_matrix,  axis=0)
            fit_mean = np.nanmean(fit_matrix, axis=0)

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                           gridspec_kw={'height_ratios': [2, 1]})

            # — Retorno anual OOS —
            ax1.plot(gens_idx, ret_mean * 100, color='#2ca02c', linewidth=2.5, label='Ret. Anual OOS (media)')
            ax1.fill_between(gens_idx,
                             (ret_mean - ret_std) * 100,
                             (ret_mean + ret_std) * 100,
                             color='#2ca02c', alpha=0.18, label='±1 std')
            ax1.axhline(0, color='#888888', linewidth=0.8, linestyle='--')
            # Línea SPY anualizada como referencia
            if curva_spy:
                spy_anual = float(max(np.prod(1 + np.array(curva_spy)), 1e-9) ** (252.0 / max(len(curva_spy), 1)) - 1.0)
                ax1.axhline(spy_anual * 100, color='#d62728', linewidth=1.2, linestyle=':', label=f'SPY anual ({spy_anual*100:.1f}%)')
            ax1.set_ylabel('Retorno Anual OOS (%)', fontsize=11)
            ax1.set_title('Evolución de Rentabilidad OOS por Generación\n'
                          f'({len(evolucion_gens_raw)} ventanas walk-forward)', fontsize=13, fontweight='bold')
            ax1.legend(fontsize=9)
            ax1.grid(alpha=0.3)

            # — Fitness IS —
            ax2.plot(gens_idx, fit_mean, color='#1f77b4', linewidth=2, label='Fitness IS (media)')
            ax2.set_ylabel('Fitness IS', fontsize=11)
            ax2.set_xlabel('Generación', fontsize=11)
            ax2.legend(fontsize=9)
            ax2.grid(alpha=0.3)

            plt.tight_layout()
            gen_plot_path = os.path.join(graficas_dir, "evolucion_generaciones.png")
            plt.savefig(gen_plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"[✔] Gráfico de evolución por generación guardado en {gen_plot_path}")

            # Guardar también los datos numéricos en JSON
            gen_data_path = os.path.join(ejecucion_dir, "evolucion_generaciones.json")
            with open(gen_data_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'gens': gens_idx,
                    'ret_anual_OOS_mean': [round(float(x), 5) for x in ret_mean],
                    'ret_anual_OOS_std':  [round(float(x), 5) for x in ret_std],
                    'fit_IS_mean':        [round(float(x), 5) for x in fit_mean],
                    'n_ventanas':         len(evolucion_gens_raw),
                }, f, indent=2)
            print(f"[✔] Datos numéricos de evolución guardados en {gen_data_path}")
            # Guardar también los pesos/ADNs reales de todas las generaciones
            adns_dict = {}
            for res_v in resultados_raw:
                w_idx = res_v['idx_v']
                historia = res_v.get('historia_generaciones_raw', [])
                for g_idx, fit_IS, adn_np in historia:
                    adns_dict[f"w{w_idx:02d}_g{g_idx:03d}"] = adn_np
            
            if adns_dict:
                adns_path = os.path.join(ejecucion_dir, "modelos_generaciones.npz")
                np.savez_compressed(adns_path, **adns_dict)
                print(f"[✔] Pesos de modelos de todas las generaciones guardados en {adns_path}")
        except ImportError:
            print("[!] Matplotlib no disponible. Omitiendo gráfico de evolución.")
        except Exception as e_gen:
            print(f"[!] Error al generar gráfico de evolución: {e_gen}")

    # ── ANÁLISIS DE DECAIMIENTO TEMPORAL DE LA CAPACIDAD PREDICTIVA (STEP SIZE DECAY) ──
    try:
        print("\n" + "="*80)
        print("📊 ANÁLISIS DE DECAIMIENTO TEMPORAL (¿El Step Size es muy grande?)")
        print("="*80)
        
        rets_por_paso_ventana = []
        for res_v in resultados_raw:
            if best_n in res_v['rets_por_n']:
                rets_por_paso_ventana.append(res_v['rets_por_n'][best_n])
                
        if rets_por_paso_ventana:
            min_len = min(len(r) for r in rets_por_paso_ventana)
            rets_matrix = np.array([r[:min_len] for r in rets_por_paso_ventana])  # (N_ventanas, min_len)
            
            media_diaria = np.mean(rets_matrix, axis=0)
            ret_acum_medio = np.cumprod(1 + media_diaria) - 1
            retorno_diario_bps = media_diaria * 10000
            
            dias_eje = np.arange(1, min_len + 1)
            slope, intercept, r_value, p_value_trend, std_err = stats.linregress(dias_eje, retorno_diario_bps)
            
            print(f"[*] Análisis de estabilidad sobre {len(rets_por_paso_ventana)} ventanas y {min_len} días de OOS:")
            print(f"    - Retorno Diario Medio inicial (Día 1): {retorno_diario_bps[0]:.1f} bps | Final (Día {min_len}): {retorno_diario_bps[-1]:.1f} bps")
            print(f"    - Tendencia (pendiente lineal de Retorno Diario): {slope:.4f} bps por día (p-value: {p_value_trend:.4f})")
            
            decay_significativo = p_value_trend < 0.05 and slope < 0
            if decay_significativo:
                print("    [!] ADVERTENCIA: Se detecta un decaimiento estadísticamente significativo en la capacidad predictiva.")
                print(f"        El step_size de {config['step_size']} podría ser demasiado grande. Considera reducirlo.")
            else:
                print("    [✔] Capacidad predictiva estable. No se detecta decaimiento temporal significativo.")
                print(f"        El step_size de {config['step_size']} es adecuado para este régimen.")
                
            if guardar_ejecucion:
                decay_data_path = os.path.join(ejecucion_dir, "analisis_decaimiento_paso.json")
                with open(decay_data_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        'step_size_analizado': config['step_size'],
                        'min_len': min_len,
                        'n_ventanas': len(rets_por_paso_ventana),
                        'comite_n': best_n,
                        'retorno_medio_diario': [round(float(x), 6) for x in media_diaria],
                        'retorno_diario_bps': [round(float(x), 4) for x in retorno_diario_bps],
                        'retorno_acumulado_medio': [round(float(x), 6) for x in ret_acum_medio],
                        'tendencia_slope': round(float(slope), 6),
                        'tendencia_p_value': round(float(p_value_trend), 6),
                        'step_size_muy_grande': bool(decay_significativo)
                    }, f, indent=2)
                print(f"[✔] Datos de decaimiento guardados en {decay_data_path}")
                
                try:
                    import matplotlib.pyplot as plt
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
                    
                    ax1.plot(dias_eje, ret_acum_medio * 100, color='#1f77b4', linewidth=2.5, label='Retorno Acumulado Medio OOS')
                    ax1.axhline(0, color='grey', linestyle='--', linewidth=0.8)
                    ax1.set_ylabel('Retorno Acumulado (%)', fontsize=11)
                    ax1.set_title(f'Decaimiento de la Capacidad Predictiva sobre Ventana OOS (N={best_n})', fontsize=13, fontweight='bold')
                    ax1.grid(alpha=0.3)
                    ax1.legend(loc='upper left')
                    
                    ax2.bar(dias_eje, retorno_diario_bps, color='#aec7e8', alpha=0.6, label='Retorno Diario Medio (bps)')
                    ax2.plot(dias_eje, slope * dias_eje + intercept, color='#d62728', linewidth=2, linestyle='--',
                             label=f'Tendencia (p={p_value_trend:.3f}, slope={slope:.4f} bps/día)')
                    ax2.axhline(0, color='grey', linestyle='-', linewidth=0.8)
                    ax2.set_ylabel('Retorno Diario Medio (bps)', fontsize=11)
                    ax2.set_xlabel('Días Transcurridos desde Entrenamiento (Paso)', fontsize=11)
                    ax2.grid(alpha=0.3)
                    ax2.legend(loc='upper left')
                    
                    plt.tight_layout()
                    decay_plot_path = os.path.join(graficas_dir, "analisis_decaimiento_paso.png")
                    plt.savefig(decay_plot_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    print(f"[✔] Gráfico de decaimiento guardado en {decay_plot_path}")
                except ImportError:
                    print("[!] Matplotlib no disponible. Omitiendo gráfico de decaimiento.")
                except Exception as e_plot:
                    print(f"[!] Error al generar gráfico de decaimiento: {e_plot}")
    except Exception as e_decay:
        print(f"[!] Error en el análisis de decaimiento temporal: {e_decay}")

    # El código de guardado original sigue a partir de aquí...
    config['n_mejores'] = best_n
    guardar_pipeline_state(config, "fase_3")
    
    equity_curves = { "spy": curva_spy }
    for n, m in results_summary:
        equity_curves[f"ia_N{n}"] = curvas_ia[n]
    
    os.makedirs("modelos", exist_ok=True)
    with open("modelos/equity_curves.json", "w") as f:
        json.dump(equity_curves, f)
    print("[✔] Curvas de capital guardadas en modelos/equity_curves.json")

    # Guardar diagnóstico completo
    diagnostico_run = {
        'run_timestamp': run_timestamp,
        'config_snapshot': {
            k: v for k, v in config.items()
            if not isinstance(v, (np.ndarray, list)) or k in ['tickers', 'ocultas']
        },
        'arquitectura': config.get('ocultas', [config.get('neuronas_capa', '?')]),
        'n_ventanas': len(ventanas_args),
        'resultados_finales': {
            str(n): {
                'ret_total': round(float(np.prod(1 + np.array(curvas_ia[n])) - 1), 4),
                'prob_calmar_gt1_mean': round(float(np.mean(prob_alpha_total[n])), 4),
                'calmar_wins': alpha_wins_top[n],
            }
            for n in tamanos_comite
        },
        'spy_ret_total': round(float(np.prod(1 + np.array(curva_spy)) - 1), 4),
        'ventanas': diagnostico_ventanas,
    }
    
    diag_path = os.path.join("modelos", "diagnostico_walkforward.json")
    with open(diag_path, "w") as f:
        json.dump(diagnostico_run, f, indent=2, ensure_ascii=False, default=str)
    print(f"[✔] Diagnóstico detallado guardado en {diag_path}")

if __name__ == "__main__":
    import multiprocessing
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    run_fase_3()

