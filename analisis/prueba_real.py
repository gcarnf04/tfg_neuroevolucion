import torch
import gc
import numpy as np
import os
import time
import json
import pandas as pd
from copy import deepcopy
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aux.datos import GestorDatos
from config import CONFIG, get_fase_config
from motor_ga import entrenar_ventana
from modelo.modelo_comite import ModeloComite
from modelo.inferencia import ensamblar_y_predecir, simular_trading_vectorizado

# Importar funciones compartidas de la Fase 3
from analisis.fase_3_walkforward import (
    _precomputar_logits_pool_mps,
    calcular_metricas_oos,
    calcular_stats_montecarlo
)

# =====================================================================
# 🏭 FASE 4: VALIDACIÓN WALK-FORWARD CIEGA (Datos oos_general)
# =====================================================================
# Este script realiza una validación ciega (walk-forward) utilizando
# parámetros obtenidos en ejecuciones previas de las Fases 1 a 3.
# Se evalúa estrictamente en el período 'oos_general' (los últimos N
# días especificados en config.json bajo EJECUCION).
#
# Regla de Oro 'oos_general':
#   Los días de 'oos_general' se recortan y ocultan automáticamente
#   para las fases 1, 2 y 3, impidiendo que los modelos los vean.
#   Este script (Fase 4) los recupera para realizar una prueba ciega
#   de rendimiento en datos completamente inéditos.
#
# Este script NO debe ejecutarse a través de 'master_runner.py',
# sino de forma puramente manual cuando se desee validar robustez.
#
# Parámetros:
#   - CARPETA_EJECUCION_PASADA: Nombre/ruta de una carpeta histórica
#     (ej. "2026-05-17_21-35-56") para cargar su 'config_ejecucion.json'.
#   - Si es None: Se carga el 'config.json' de la raíz del proyecto.
# =====================================================================
CARPETA_EJECUCION_PASADA = "2026-05-23_14-50-26"

_DIR_SCRIPT = os.path.dirname(os.path.abspath(__file__))
_DIR_LABORATORIO = os.path.dirname(_DIR_SCRIPT) if 'analisis' in _DIR_SCRIPT else _DIR_SCRIPT

def run_fase_4():
    print("\n" + "="*80)
    print(" 🏭 FASE 4: VALIDACIÓN WALK-FORWARD CIEGA (Ensambles de Comités) 🏭")
    print("="*80)
    
    # Carga de la configuración correspondiente
    if CARPETA_EJECUCION_PASADA is not None:
        ruta_posible_1 = os.path.join(_DIR_LABORATORIO, "ejecuciones", CARPETA_EJECUCION_PASADA, "config_ejecucion.json")
        ruta_posible_2 = os.path.join(CARPETA_EJECUCION_PASADA, "config_ejecucion.json")
        ruta_posible_3 = CARPETA_EJECUCION_PASADA if CARPETA_EJECUCION_PASADA.endswith(".json") else os.path.join(CARPETA_EJECUCION_PASADA, "config_ejecucion.json")
        
        ruta_final = None
        for r in [ruta_posible_1, ruta_posible_2, ruta_posible_3]:
            if os.path.exists(r):
                ruta_final = r
                break
                
        if ruta_final is None:
            raise FileNotFoundError(f"No se pudo encontrar 'config_ejecucion.json' en la carpeta especificada: {CARPETA_EJECUCION_PASADA}")
            
        with open(ruta_final, "r", encoding="utf-8") as f:
            config = json.load(f)
        print(f"[*] Configuración cargada con éxito desde histórico: {ruta_final}")
    else:
        config = deepcopy(CONFIG)
        _f4_cfg = get_fase_config("FASE_4")
        for k, v in _f4_cfg.items():
            config[k] = v
        print("[*] Configuración por defecto de FASE_4 cargada desde 'config.json' (raíz del proyecto).")
    
    # Parámetros dinámicos de Meta-Comité
    k_comite_meta = config.get("k_comite_meta", 5)
    
    # 1. Validar la existencia del parámetro oos_general
    oos_general = config.get('oos_general', 0)
    if oos_general <= 0:
        print("[⚠️] Error: 'oos_general' no está configurado o es menor/igual a 0 en config.json.")
        print("[*] Configura 'oos_general' en la sección EJECUCION con un valor > 0 (ej. 252).")
        return
        
    print(f"[*] Período reservado 'oos_general' detectado: {oos_general} días de trading.")
    
    # 2. Descargar y procesar los datos COMPLETOS (bypass del recorte general)
    fecha_fin_wf = config.get('fecha_fin')
    if not fecha_fin_wf:
        fecha_fin_wf = time.strftime("%Y-%m-%d")
        config['fecha_fin'] = fecha_fin_wf
    gestor = GestorDatos(config['tickers'], config['ticker_cash'], config['ticker_macro'], config['fecha_inicio'], fecha_fin_wf)
    
    print("[*] Cargando serie temporal completa (bypass del recorte general)...")
    X, R, idx_c, idx_m, idx_l = gestor.obtener_datos_listos(sin_recorte_general=True)
    fechas_index = gestor.features.index
    
    N = len(X)
    idx_inicio_wf = N - oos_general
    
    fecha_inicio_oos = fechas_index[idx_inicio_wf].strftime("%Y-%m-%d")
    fecha_fin_oos = fechas_index[-1].strftime("%Y-%m-%d")
    
    print(f"[✔] Dataset cargado: {N} días de trading en total.")
    print(f"[*] Período de Validación Walk-Forward Ciega: {fecha_inicio_oos} a {fecha_fin_oos} ({oos_general} días)")
    
    # 3. Configuración de ventanas walk-forward en la zona ciega
    step_size = config.get('step_size', 63)
    train_size = config.get('train_size', 756)
    
    ventanas_args = []
    idx_v = 0
    idx_primer_train = idx_inicio_wf - train_size
    
    for i in range(idx_primer_train, N - train_size - step_size + 1, step_size):
        ventanas_args.append((idx_v, i))
        idx_v += 1
        
    if len(ventanas_args) == 0:
        print(f"[⚠️] Advertencia: 'oos_general' ({oos_general}) es menor que 'step_size' ({step_size}).")
        print("[*] Forzando una ventana única de validación ciega.")
        ventanas_args = [(0, idx_inicio_wf - train_size)]
        
    print(f"[*] Diseñando simulación: {len(ventanas_args)} ventanas Walk-Forward ciegas detectadas.")
    
    # Obtener el comité óptimo ya encontrado en la configuración guardada
    n_comite_opt = config.get('n_mejores', config.get('n_comite', 1))
    
    # 4. Inicializar estructuras para las corridas individuales y el Meta-Comité
    print(f"[*] Entrenando exactamente K = {k_comite_meta} comités independientes de N = {n_comite_opt} modelos por ventana...")
    
    curvas_ia_runs = {run_idx: [] for run_idx in range(1, k_comite_meta + 1)}
    curva_ia_meta = []
    curva_spy = []
    
    # Configuración de ejecución
    config_w_run = config.copy()
    config_w_run['silent'] = True
    config_w_run['n_mejores'] = n_comite_opt
    config_w_run['n_comite'] = n_comite_opt
    config_w_run['n_fundadores'] = max(config_w_run.get('n_fundadores', 1), n_comite_opt)
    
    # Overrides de Fase 3 si se usa config.json vivo
    if CARPETA_EJECUCION_PASADA is None:
        _f3_cfg = get_fase_config("FASE_3")
        for key in ['poblacion', 'generaciones', 'paciencia', 'timeout_minado',
                    'n_fundadores', 'poblacion_mining', 'generaciones_mining', 'cpus']:
            key_f3 = f'{key}_f3'
            if key_f3 in _f3_cfg:
                config_w_run[key] = _f3_cfg[key_f3]
            
    # Instanciar el comite comúnmente
    inf_model = ModeloComite(
        idx_c, idx_m, idx_l,
        len(config['tickers']) + 1,
        ocultas=config.get('ocultas', [56])
    )
    
    # Bucle principal de Walk-Forward por ventanas
    for idx_v, i_tr_l in tqdm(ventanas_args, desc="WF Ciego (Meta-Comité)"):
        f_oos_ini = i_tr_l + train_size
        f_oos_fin = min(f_oos_ini + step_size, N)
        X_oos = X[f_oos_ini:f_oos_fin]
        R_oos = R[f_oos_ini:f_oos_fin]
        
        spy_rets_oos = R_oos[1:, 0]
        curva_spy.extend(spy_rets_oos.tolist())
        
        logits_ventana_runs = []
        
        for run_idx in range(1, k_comite_meta + 1):
            run_seed = 42 + run_idx
            torch.manual_seed(run_seed)
            np.random.seed(run_seed)
            
            # Entrenamiento del comité para esta corrida
            res = entrenar_ventana((idx_v, i_tr_l, X, R, idx_c, idx_m, idx_l, config_w_run))
            
            # Normalización OOS
            X_oos_norm = (X_oos - res['stats_norm'][0]) / res['stats_norm'][1]
            
            # Precomputar Logits
            logits_pool = _precomputar_logits_pool_mps(
                res['pesos_ensemble'], X_oos_norm, config_w_run, idx_c, idx_m, idx_l
            )
            logits_comite = logits_pool[:n_comite_opt]
            logits_ventana_runs.append(logits_comite)
            
            # Simular OOS para esta corrida individual
            p_individual, _ = ensamblar_y_predecir(
                None, None, config, inf_model,
                all_opinions_raw=logits_comite
            )
            rets_individual = simular_trading_vectorizado(p_individual, R_oos, config)
            curvas_ia_runs[run_idx].extend(rets_individual.tolist())
            
            # Limpieza inmediata de memoria
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            gc.collect()
            
        # Concatenar logits precomputados de todas las corridas en esta ventana
        logits_meta_ventana = np.concatenate(logits_ventana_runs, axis=0) # shape (k_comite_meta * n_comite_opt, T, A)
        
        # Simular OOS del Meta-Comité
        p_meta, _ = ensamblar_y_predecir(
            None, None, config, inf_model,
            all_opinions_raw=logits_meta_ventana
        )
        rets_meta = simular_trading_vectorizado(p_meta, R_oos, config)
        curva_ia_meta.extend(rets_meta.tolist())
        
    # 5. Calcular métricas detalladas
    metrics_runs = {
        'ret_acum': [], 'ret_anual': [], 'ret_mens': [], 'mdd': [],
        'sharpe': [], 'sortino': [], 'calmar': [], 'p_val': []
    }
    
    for run_idx in range(1, k_comite_meta + 1):
        ret_acu, mdd, _, sharpe, sortino, calmar, _, ret_mensual = calcular_metricas_oos(curvas_ia_runs[run_idx], curva_spy)
        ret_anual_ia, _, p_val = calcular_stats_montecarlo(curvas_ia_runs[run_idx], curva_spy, iteraciones=300)
        
        metrics_runs['ret_acum'].append(ret_acu)
        metrics_runs['ret_anual'].append(ret_anual_ia)
        metrics_runs['ret_mens'].append(ret_mensual)
        metrics_runs['mdd'].append(mdd)
        metrics_runs['sharpe'].append(sharpe)
        metrics_runs['sortino'].append(sortino)
        metrics_runs['calmar'].append(calmar)
        metrics_runs['p_val'].append(p_val)
        
    ret_acu_meta, mdd_meta, _, sharpe_meta, sortino_meta, calmar_meta, _, ret_mensual_meta = calcular_metricas_oos(curva_ia_meta, curva_spy)
    ret_anual_meta, _, p_val_meta = calcular_stats_montecarlo(curva_ia_meta, curva_spy, iteraciones=300)
    
    ret_spy, mdd_spy, _, sharpe_spy, sortino_spy, calmar_spy, _, ret_mensual_spy = calcular_metricas_oos(curva_spy, curva_spy)
    ret_anual_spy_val = (max(np.prod(1 + np.array(curva_spy)), 1e-9) ** (252.0 / max(len(curva_spy), 1))) - 1.0
    
    # 6. Presentar Resultados Comparativos
    print("\n" + "🏆"*5 + " RESULTADOS DE VALIDACIÓN WALK-FORWARD CIEGA (Ensambles de Comités) " + "🏆"*5)
    print(f"\n📊 Detalle para Ensambles de Comités ({k_comite_meta} Corridas | Comité de {n_comite_opt} modelos):")
    print(f"{'Entidad / Corrida':<25} | {'Ret. Acum':<10} | {'Ret. Anual':<11} | {'Ret. Mens':<10} | {'Max DD':<8} | {'Sharpe':<7} | {'Sortino':<8} | {'Calmar':<7} | {'P(Calmar)':<9}")
    print("-" * 122)
    
    for r_idx in range(k_comite_meta):
        print(f"Run {r_idx+1:02d}                      | {metrics_runs['ret_acum'][r_idx]*100:9.2f}% | {metrics_runs['ret_anual'][r_idx]*100:10.2f}% | {metrics_runs['ret_mens'][r_idx]*100:9.2f}% | {metrics_runs['mdd'][r_idx]*100:7.2f}% | {metrics_runs['sharpe'][r_idx]:7.2f} | {metrics_runs['sortino'][r_idx]:8.2f} | {metrics_runs['calmar'][r_idx]:7.2f} | {metrics_runs['p_val'][r_idx]:9.4f}")
        
    print("-" * 122)
    
    stats_summary = {}
    for key in ['ret_acum', 'ret_anual', 'ret_mens', 'mdd', 'sharpe', 'sortino', 'calmar', 'p_val']:
        vals = np.array(metrics_runs[key])
        stats_summary[key] = np.mean(vals)
        
    m = stats_summary
    print(f"MEDIA INDIVIDUAL          | {m['ret_acum']*100:9.2f}% | {m['ret_anual']*100:10.2f}% | {m['ret_mens']*100:9.2f}% | {m['mdd']*100:7.2f}% | {m['sharpe']:7.2f} | {m['sortino']:8.2f} | {m['calmar']:7.2f} | {m['p_val']:9.4f}")
    print("-" * 122)
    
    print(f"⭐ META-COMITÉ ⭐         | {ret_acu_meta*100:9.2f}% | {ret_anual_meta*100:10.2f}% | {ret_mensual_meta*100:9.2f}% | {mdd_meta*100:7.2f}% | {sharpe_meta:7.2f} | {sortino_meta:8.2f} | {calmar_meta:7.2f} | {p_val_meta:9.4f}")
    print("-" * 122)
    
    print(f"Benchmark SPY             | {ret_spy*100:9.2f}% | {ret_anual_spy_val*100:10.2f}% | {ret_mensual_spy*100:9.2f}% | {mdd_spy*100:7.2f}% | {sharpe_spy:7.2f} | {sortino_spy:8.2f} | {calmar_spy:7.2f} | {'--':>9}")
    print("=" * 122 + "\n")
    
    # Juez Supremo
    print(f"⭐ JUEZ SUPREMO DE ROBUSTEZ: Ensambles de Comités vs Modelos Individuales")
    print(f"   Meta-Comité -> Rentabilidad Anual: {ret_anual_meta*100:.2f}% | Sharpe: {sharpe_meta:.2f} | Calmar: {calmar_meta:.2f} | P-value: {p_val_meta:.4f}")
    print(f"   Media Indiv -> Rentabilidad Anual: {m['ret_anual']*100:.2f}% | Sharpe: {m['sharpe']:.2f} | Calmar: {m['calmar']:.2f}")
    print("="*122 + "\n")

if __name__ == "__main__":
    f_out = None
    original_stdout = sys.stdout
    ruta_log = None
    
    if CARPETA_EJECUCION_PASADA is not None:
        ruta_posible_1 = os.path.join(_DIR_LABORATORIO, "ejecuciones", CARPETA_EJECUCION_PASADA, "config_ejecucion.json")
        ruta_posible_2 = os.path.join(CARPETA_EJECUCION_PASADA, "config_ejecucion.json")
        ruta_posible_3 = CARPETA_EJECUCION_PASADA if CARPETA_EJECUCION_PASADA.endswith(".json") else os.path.join(CARPETA_EJECUCION_PASADA, "config_ejecucion.json")
        
        ruta_final = None
        for r in [ruta_posible_1, ruta_posible_2, ruta_posible_3]:
            if os.path.exists(r):
                ruta_final = r
                break
                
        if ruta_final is not None:
            carpeta_ejecucion = os.path.dirname(ruta_final)
            ruta_log = os.path.join(carpeta_ejecucion, "resultado_prueba_real.txt")
            f_out = open(ruta_log, "w", encoding="utf-8")
            
            class Tee(object):
                def __init__(self, file1, file2):
                    self.file1 = file1
                    self.file2 = file2
                def write(self, data):
                    self.file1.write(data)
                    self.file2.write(data)
                def flush(self):
                    self.file1.flush()
                    self.file2.flush()
                    
            sys.stdout = Tee(original_stdout, f_out)
            
    try:
        run_fase_4()
    finally:
        if f_out is not None:
            sys.stdout = original_stdout
            f_out.close()
            print(f"[✔] Resultados de prueba_real guardados en: {ruta_log}")
