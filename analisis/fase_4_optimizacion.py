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
from config import CONFIG, get_fase_config, save_optimized_params
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
# 🏭 FASE 4 OPTIMIZACIÓN: BÚSQUEDA DEL TAMAÑO ÓPTIMO DE META-COMITÉ
# =====================================================================
# Este script optimiza el número de comités independientes a ensamblar
# en el Meta-Comité para encontrar el punto óptimo que minimice el p-value
# y maximice el rendimiento anual y el control del Drawdown en oos_general.
# =====================================================================
CARPETA_EJECUCION_PASADA = None  # Si es None, busca y carga automáticamente la última ejecución en 'ejecuciones/'

_DIR_SCRIPT = os.path.dirname(os.path.abspath(__file__))
_DIR_LABORATORIO = os.path.dirname(_DIR_SCRIPT) if 'analisis' in _DIR_SCRIPT else _DIR_SCRIPT

def run_fase_4_optimizacion():
    print("\n" + "="*80)
    print(" 🏭 FASE 4 OPTIMIZACIÓN: TAMAÑO ÓPTIMO DE ENSAMBLE DE COMITÉS 🏭")
    print("="*80)
    
    # Carga de la configuración
    carpeta_resolvida = CARPETA_EJECUCION_PASADA
    if carpeta_resolvida is None:
        import glob
        pattern = os.path.join(_DIR_LABORATORIO, "ejecuciones", "20*")
        dirs = sorted(glob.glob(pattern))
        dirs = [d for d in dirs if os.path.isdir(d)]
        
        real_dirs = []
        for d in dirs:
            cfg_path = os.path.join(d, "config_ejecucion.json")
            if os.path.exists(cfg_path):
                try:
                    with open(cfg_path, "r", encoding="utf-8") as f:
                        cfg_temp = json.load(f)
                    if cfg_temp.get("oos_general", 0) > 0:
                        real_dirs.append(d)
                except:
                    pass
                    
        if real_dirs:
            carpeta_resolvida = os.path.basename(real_dirs[-1])
            print(f"[*] Detectada última ejecución real automáticamente: {carpeta_resolvida}")
        else:
            print("[⚠️] Advertencia: No se encontraron carpetas de ejecuciones pasadas reales con oos_general > 0.")
            
    ruta_final = None
    if carpeta_resolvida is not None:
        ruta_posible_1 = os.path.join(_DIR_LABORATORIO, "ejecuciones", carpeta_resolvida, "config_ejecucion.json")
        ruta_posible_2 = os.path.join(carpeta_resolvida, "config_ejecucion.json")
        ruta_posible_3 = carpeta_resolvida if carpeta_resolvida.endswith(".json") else os.path.join(carpeta_resolvida, "config_ejecucion.json")
        
        for r in [ruta_posible_1, ruta_posible_2, ruta_posible_3]:
            if os.path.exists(r):
                ruta_final = r
                break
                
        if ruta_final is None:
            raise FileNotFoundError(f"No se pudo encontrar 'config_ejecucion.json' en la carpeta: {carpeta_resolvida}")
            
        with open(ruta_final, "r", encoding="utf-8") as f:
            config = json.load(f)
        print(f"[*] Configuración cargada desde histórico: {ruta_final}")
    else:
        config = deepcopy(CONFIG)
        _f4_cfg = get_fase_config("FASE_4")
        for k, v in _f4_cfg.items():
            config[k] = v
        print("[*] Configuración por defecto de FASE_4 cargada desde 'config.json'.")
    
    # Parámetros dinámicos cargados de config.json
    K_MAX_RUNS = config.get("K_MAX_RUNS", 10)
    p_val_max = config.get("p_val_max", 0.10)
    
    # 1. Validar oos_general
    oos_general = config.get('oos_general', 0)
    if oos_general <= 0:
        print("[⚠️] Error: 'oos_general' no está configurado en config.json.")
        return
        
    print(f"[*] Período reservado 'oos_general': {oos_general} días de trading.")
    print(f"[*] Rango de búsqueda para ensamble K: 1 a {K_MAX_RUNS} (Corte P-Value: {p_val_max})")
    
    # 2. Descargar y procesar datos (CON recorte general activo para blindaje total)
    fecha_fin_wf = config.get('fecha_fin')
    if not fecha_fin_wf:
        fecha_fin_wf = time.strftime("%Y-%m-%d")
        config['fecha_fin'] = fecha_fin_wf
    gestor = GestorDatos(config['tickers'], config['ticker_cash'], config['ticker_macro'], config['fecha_inicio'], fecha_fin_wf)
    
    # sin_recorte_general=False garantiza que oos_general queda estrictamente oculto y recortado
    X, R, idx_c, idx_m, idx_l = gestor.obtener_datos_listos(sin_recorte_general=False)
    fechas_index = gestor.features.index
    
    N = len(X)
    fecha_inicio_is = fechas_index[0].strftime("%Y-%m-%d")
    fecha_fin_is = fechas_index[-1].strftime("%Y-%m-%d")
    
    print(f"[✔] Dataset cargado: {N} días.")
    print(f"[*] Período de Optimización In-Sample (IS): {fecha_inicio_is} a {fecha_fin_is} (oos_general blindado)")
    
    # 3. Configuración de ventanas walk-forward en el espacio IS recortado
    step_size = config.get('step_size', 63)
    train_size = config.get('train_size', 756)
    
    ventanas_args = []
    idx_v = 0
    
    # Barremos hasta las últimas 4 ventanas del dataset recortado de entrenamiento para robustez
    idx_ultimo_train = N - train_size - step_size
    for i in range(idx_ultimo_train, idx_ultimo_train - step_size * 4 - 1, -step_size):
        if i >= 0:
            ventanas_args.insert(0, (idx_v, i))
            idx_v += 1
            
    if len(ventanas_args) == 0:
        ventanas_args = [(0, max(0, N - train_size))]
        
    print(f"[*] Diseñando simulación: {len(ventanas_args)} ventanas In-Sample de optimización detectadas.")
    
    n_comite_opt = config.get('n_mejores', config.get('n_comite', 1))
    
    # 4. Inicializar curvas para cada tamaño de ensamble K (de 1 a K_MAX_RUNS)
    curvas_meta_k = {k: [] for k in range(1, K_MAX_RUNS + 1)}
    ret_anual_por_ventana_por_k = {k: [] for k in range(1, K_MAX_RUNS + 1)}
    curva_spy = []
    
    config_w_run = config.copy()
    config_w_run['silent'] = True
    config_w_run['n_mejores'] = n_comite_opt
    config_w_run['n_comite'] = n_comite_opt
    config_w_run['n_fundadores'] = max(config_w_run.get('n_fundadores', 1), n_comite_opt)
    
    if CARPETA_EJECUCION_PASADA is None:
        _f3_cfg = get_fase_config("FASE_3")
        for key in ['poblacion', 'generaciones', 'paciencia', 'timeout_minado',
                    'n_fundadores', 'poblacion_mining', 'generaciones_mining', 'cpus']:
            key_f3 = f'{key}_f3'
            if key_f3 in _f3_cfg:
                config_w_run[key] = _f3_cfg[key_f3]
            
    inf_model = ModeloComite(
        idx_c, idx_m, idx_l,
        len(config['tickers']) + 1,
        ocultas=config.get('ocultas', [56])
    )
    
    # Bucle por ventanas (Evaluación In-Sample)
    for idx_v, i_tr_l in tqdm(ventanas_args, desc="WF Ciego (Optimización IS de K)"):
        # Extraer datos In-Sample de entrenamiento para esta ventana
        X_is = X[i_tr_l : i_tr_l + train_size]
        R_is = R[i_tr_l : i_tr_l + train_size]
        
        spy_rets_is = R_is[1:, 0]
        curva_spy.extend(spy_rets_is.tolist())
        
        logits_ventana_runs = []
        
        # Entrenar K_MAX_RUNS comités independientes
        for run_idx in range(1, K_MAX_RUNS + 1):
            run_seed = 42 + run_idx
            torch.manual_seed(run_seed)
            np.random.seed(run_seed)
            
            res = entrenar_ventana((idx_v, i_tr_l, X, R, idx_c, idx_m, idx_l, config_w_run))
            
            # Normalización IS
            X_is_norm = (X_is - res['stats_norm'][0]) / res['stats_norm'][1]
            
            # Precomputar Logits sobre datos IS
            logits_pool = _precomputar_logits_pool_mps(
                res['pesos_ensemble'], X_is_norm, config_w_run, idx_c, idx_m, idx_l
            )
            logits_comite = logits_pool[:n_comite_opt]
            logits_ventana_runs.append(logits_comite)
            
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            gc.collect()
            
        # Simular trading IS para cada tamaño de Meta-Comité k de 1 a K_MAX_RUNS
        for k in range(1, K_MAX_RUNS + 1):
            logits_meta_k = np.concatenate(logits_ventana_runs[:k], axis=0)
            p_meta_k, _ = ensamblar_y_predecir(
                None, None, config, inf_model,
                all_opinions_raw=logits_meta_k
            )
            # Simulación de trading estrictamente In-Sample
            rets_meta_k = simular_trading_vectorizado(p_meta_k, R_is, config)
            curvas_meta_k[k].extend(rets_meta_k.tolist())
            
            # Guardar retornos anualizados independientes por ventana
            if len(rets_meta_k) >= 5:
                ret_anual = (max(np.prod(1 + rets_meta_k), 1e-9) ** (252.0 / len(rets_meta_k))) - 1.0
                ret_anual_por_ventana_por_k[k].append(ret_anual)
            
    # 5. Calcular métricas para cada tamaño de ensamble e identificar la robustez
    results_opt = []
    objetivos = np.arange(0.01, 0.16, 0.01)  # 1% al 15%
    robustez_por_k = {}
    
    for k in range(1, K_MAX_RUNS + 1):
        ret_acu, mdd, _, _, _, _, _, ret_mensual = calcular_metricas_oos(curvas_meta_k[k], curva_spy)
        ret_anual_ia, _, p_val = calcular_stats_montecarlo(curvas_meta_k[k], curva_spy, iteraciones=300)
        
        # Calcular robustez via t-test Unilateral de un grupo
        retornos_ventanas = ret_anual_por_ventana_por_k[k]
        n_v = len(retornos_ventanas)
        
        if not retornos_ventanas or n_v < 2:
            robustez_por_k[k] = {'max_obj': 0.0, 'p_val_obj': 1.0, 'p_val_1pct': 1.0}
        else:
            max_obj_val = 0.0
            p_val_at_max = 1.0
            p_val_at_1pct = 1.0
            
            for obj in objetivos:
                t_stat, p_val_tt = stats.ttest_1samp(retornos_ventanas, popmean=obj)
                p_unilateral = p_val_tt / 2 if t_stat > 0 else 1.0 - p_val_tt / 2
                
                if abs(obj - 0.01) < 1e-9:
                    p_val_at_1pct = p_unilateral
                    
                if p_unilateral < 0.05:
                    max_obj_val = obj
                    p_val_at_max = p_unilateral
                    
            robustez_por_k[k] = {
                'max_obj': max_obj_val,
                'p_val_obj': p_val_at_max,
                'p_val_1pct': p_val_at_1pct
            }
            
        results_opt.append({
            'k': k,
            'ret_acum': ret_acu,
            'ret_anual': ret_anual_ia,
            'ret_mens': ret_mensual,
            'mdd': mdd,
            'p_val': p_val
        })
        
    # Benchmark SPY
    ret_spy, mdd_spy, _, _, _, _, _, ret_mensual_spy = calcular_metricas_oos(curva_spy, curva_spy)
    ret_anual_spy_val = (max(np.prod(1 + np.array(curva_spy)), 1e-9) ** (252.0 / max(len(curva_spy), 1))) - 1.0
    
    # JUEZ SUPREMO DE ROBUSTEZ (Mayor Retorno Seguro con P-Value < 0.05)
    comites_validos = [k for k in range(1, K_MAX_RUNS + 1) if robustez_por_k[k]['max_obj'] > 0.0]
    if comites_validos:
        best_k = max(comites_validos, key=lambda k: (robustez_por_k[k]['max_obj'], -robustez_por_k[k]['p_val_obj'], -k))
        best_obj = robustez_por_k[best_k]['max_obj']
        best_p_val = robustez_por_k[best_k]['p_val_obj']
    else:
        best_k = min(range(1, K_MAX_RUNS + 1), key=lambda k: (robustez_por_k[k]['p_val_1pct'], k))
        best_obj = 0.0
        best_p_val = robustez_por_k[best_k]['p_val_1pct']
        
    best_r = next(r for r in results_opt if r['k'] == best_k)

    # 6. Construir y Mostrar Tabla de Optimización en el formato solicitado
    output_lines = []
    output_lines.append("📊📊📊📊📊 TABLA DE OPTIMIZACIÓN: TAMAÑO DEL META-COMITÉ (K) 📊📊📊📊📊\n")
    output_lines.append(f"{'Tamaño K':<10} | {'Ret. Acum':<10} | {'Ret. Anual':<11} | {'Ret. Mens':<10} | {'Max DD':<8} | {'P-Value':<9}")
    output_lines.append("-" * 70)
    
    for r in results_opt:
        marcador = " ⭐" if r['k'] == best_r['k'] else ""
        output_lines.append(f"K = {r['k']:02d}{marcador:<5} | {r['ret_acum']*100:9.2f}% | {r['ret_anual']*100:10.2f}% | {r['ret_mens']*100:9.2f}% | {r['mdd']*100:7.2f}% | {r['p_val']:9.4f}")
        
    output_lines.append("-" * 70)
    output_lines.append(f"SPY Bchmark| {ret_spy*100:9.2f}% | {ret_anual_spy_val*100:10.2f}% | {ret_mensual_spy*100:9.2f}% | {mdd_spy*100:7.2f}% | {'--':>9}")
    output_lines.append("=" * 70 + "\n")
    
    # MATRIZ DE ROBUSTEZ DE FASE 4 (P-Values via t-test Unilateral)
    output_lines.append("📊 MATRIZ DE ROBUSTEZ DE FASE 4 (P-Values via t-test Unilateral) 📊")
    output_lines.append("   H0: La media de retornos anualizados IS es INFERIOR o IGUAL al objetivo (menor es mejor)")
    output_lines.append("=" * 140)
    
    columnas_obj = " | ".join([f"{obj*100:4.1f}%" for obj in objetivos])
    header_matriz = f"{'Tamaño K':<8} | {columnas_obj}"
    output_lines.append(header_matriz)
    output_lines.append("-" * len(header_matriz))
    
    for k in range(1, K_MAX_RUNS + 1):
        retornos_ventanas = ret_anual_por_ventana_por_k[k]
        n_v = len(retornos_ventanas)
        
        fila_vals = []
        for obj in objetivos:
            if not retornos_ventanas or n_v < 2:
                fila_vals.append(" --  ")
                continue
                
            t_stat, p_val_tt = stats.ttest_1samp(retornos_ventanas, popmean=obj)
            p_unilateral = p_val_tt / 2 if t_stat > 0 else 1.0 - p_val_tt / 2
            
            marker = "*" if p_unilateral < 0.05 else " "
            fila_vals.append(f"{marker}{p_unilateral:5.3f}")
            
        valores_fila = " | ".join(fila_vals)
        marcador_k = "⭐" if k == best_k else " "
        output_lines.append(f"K = {k:02d} {marcador_k} | {valores_fila}")
        
    output_lines.append("=" * len(header_matriz) + "\n")
    
    output_lines.append("⭐ RECOMENDACIÓN DE ENSAMBLE ÓPTIMO ⭐")
    output_lines.append(f"   El tamaño de ensamble recomendado es K = {best_r['k']} comités independientes.")
    output_lines.append(f"   Métricas Esperadas: Ret. Anual: {best_r['ret_anual']*100:.2f}% | Max DD: {best_r['mdd']*100:.2f}% | P-value: {best_r['p_val']:.4f}")
    output_lines.append("="*70)
    
    texto_completo = "\n".join(output_lines)
    print("\n" + texto_completo + "\n")
    
    # Guardar en archivo resultado_fase_4.txt dentro de la carpeta de ejecución
    if ruta_final:
        carpeta_destino = os.path.dirname(ruta_final)
        graficas_dir = os.path.join(carpeta_destino, "graficas")
        os.makedirs(graficas_dir, exist_ok=True)
        
        ruta_txt = os.path.join(carpeta_destino, "resultado_fase_4.txt")
        try:
            with open(ruta_txt, "w", encoding="utf-8") as f:
                f.write(texto_completo)
            print(f"[✔] Resultados de Fase 4 guardados en: {ruta_txt}")
        except Exception as e:
            print(f"[⚠️] Error guardando 'resultado_fase_4.txt': {e}")
            
        # Generar y guardar el gráfico de la matriz de p-values para el Meta-Comité
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import seaborn as sns
            import pandas as pd
            
            p_matrix_data_f4 = []
            for obj in objetivos:
                row_data = {}
                for k in range(1, K_MAX_RUNS + 1):
                    retornos_ventanas = ret_anual_por_ventana_por_k[k]
                    n_v = len(retornos_ventanas)
                    if not retornos_ventanas or n_v < 2:
                        row_data[f"K={k:02d}"] = np.nan
                    else:
                        t_stat, p_val_tt = stats.ttest_1samp(retornos_ventanas, popmean=obj)
                        p_unilateral = p_val_tt / 2 if t_stat > 0 else 1.0 - p_val_tt / 2
                        row_data[f"K={k:02d}"] = p_unilateral
                p_matrix_data_f4.append(row_data)
                
            df_pvals_f4 = pd.DataFrame(p_matrix_data_f4, index=[f"{int(obj*100)}%" for obj in objetivos])
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(df_pvals_f4, annot=True, fmt=".4f", cmap='RdYlGn_r', vmin=0.0, vmax=0.20, cbar_kws={'label': 'p-value'})
            plt.title("Matriz de Robustez de Fase 4 (Meta-Comité P-Values)")
            plt.ylabel("Objetivo de Beneficio Anual")
            plt.xlabel("Tamaño del Meta-Comité (K)")
            plt.tight_layout()
            
            dest_img_f4 = os.path.join(graficas_dir, "matriz_meta_comite_pvalues.png")
            plt.savefig(dest_img_f4, dpi=150)
            plt.close()
            print(f"[✔] Gráfico de robustez de Fase 4 guardado en: {dest_img_f4}")
        except Exception as e:
            print(f"[!] Error al generar gráfico de robustez en Fase 4: {e}")
            
        # Guardar el nuevo parámetro en config_ejecucion.json de la ejecución
        try:
            with open(ruta_final, "r", encoding="utf-8") as f:
                cfg_exec = json.load(f)
            
            cfg_exec["k_comite_meta"] = best_r['k']
            if "FASE_4" not in cfg_exec:
                cfg_exec["FASE_4"] = {}
            cfg_exec["FASE_4"]["k_comite_meta"] = best_r['k']
            
            with open(ruta_final, "w", encoding="utf-8") as f:
                json.dump(cfg_exec, f, indent=2, ensure_ascii=False)
            print(f"[✔] Parámetro 'k_comite_meta' persistido en la ejecución: {ruta_final}")
        except Exception as e:
            print(f"[⚠️] Error persistiendo en config_ejecucion.json: {e}")
            
    # Persistir el parámetro optimizado k_comite_meta en config.json
    print(f"[*] Persistiendo 'k_comite_meta' = {best_r['k']} en config.json...")
    save_optimized_params("FASE_4", {"k_comite_meta": best_r['k']})
    print("[✔] Configuración de Fase 4 persistida globalmente.")

if __name__ == "__main__":
    run_fase_4_optimizacion()
