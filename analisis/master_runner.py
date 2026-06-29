"""
╔══════════════════════════════════════════════════════════════════╗
║         🧪 ALPHA HUNTER — MASTER RUNNER v1.0                    ║
║         Ejecuta el pipeline completo (F1 → F2 → F3)             ║
║         o un experimento multi-hipótesis desatendido.            ║
╚══════════════════════════════════════════════════════════════════╝

USO BÁSICO (pipeline completo con la config actual):
    python analisis/master_runner.py

USO CON HIPÓTESIS (para dejar corriendo de fondo):
    Editar la sección HIPÓTESIS más abajo y ejecutar:
    python analisis/master_runner.py --hipotesis

FLAGS:
    --solo-f2       Solo ejecuta Fase 2 + Fase 3 (salta Fase 1)
    --solo-f3       Solo ejecuta Fase 3
    --hipotesis     Activa el modo multi-experimento
"""

import sys
import os
import json
import gc
import time
import csv
import argparse
import traceback
from copy import deepcopy
from datetime import datetime
import io

# --- Añadir raíz del laboratorio al path ---
_DIR_SCRIPT = os.path.dirname(os.path.abspath(__file__))
_DIR_LAB    = os.path.dirname(_DIR_SCRIPT)
sys.path.insert(0, _DIR_LAB)

import torch
import config as cfg_module  # Importamos el módulo entero para acceder al builder
from config import CONFIG, MAPA_SLIPPAGE

# ═══════════════════════════════════════════════════════════
class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            if isinstance(obj, str): f.write(obj)
            else: f.write(str(obj))
            f.flush()
    def flush(self):
        for f in self.files: f.flush()

# ═══════════════════════════════════════════════════════════
# 📋 HIPÓTESIS A PROBAR (Editar aquí para modo multi-experimento)
# ═══════════════════════════════════════════════════════════
HIPOTESIS = [
]
# ═══════════════════════════════════════════════════════════


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def _banner(texto: str, ancho: int = 70):
    print("\n" + "═" * ancho)
    print(f"  {texto}")
    print("═" * ancho)


def _limpiar_memoria():
    """Libera caché de MPS/CUDA y fuerza recolección de basura."""
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def _resetear_caches_fase1():
    """Invalida el dataset cacheado de Fase 1 para que se regenere."""
    import analisis.fase_1_metrica as f1
    f1.DATASET_FASE_1 = None


def _resetear_datos_globales():
    """Invalida los datos globales de cada fase para forzar recarga."""
    import analisis.fase_1_metrica as f1
    import analisis.fase_2_hiperparametros as f2
    f1.X_global = f1.R_global = f1.DATASET_FASE_1 = None
    f1.idx_c_global = f1.idx_m_global = f1.idx_l_global = None
    f2.X_global = f2.R_global = None
    f2.idx_c_global = f2.idx_m_global = f2.idx_l_global = None
    
    # Limpieza de Hardware
    _limpiar_memoria()


def _aplicar_hipotesis_a_config(hipotesis: dict):
    """
    Solución Nuclear de Memoria: Genera una config nueva y actualiza
    in-place la referencia global compartida por todos los módulos.
    """
    # 1. Recargar configuración base LIMPIA desde el archivo
    nueva_config = cfg_module._build_config()

    # 2. Inyectar historial de optimizaciones pasadas (Soberanía Nivel 2)
    fase1_params = {k: v for k, v in cfg_module.get_fase_config("FASE_1").items() if not k.startswith("_")}
    fase2_params = {k: v for k, v in cfg_module.get_fase_config("FASE_2").items() if not k.startswith("_")}
    fase3_params = {k: v for k, v in cfg_module.get_fase_config("FASE_3").items() if not k.startswith("_")}
    nueva_config.update(fase1_params)
    nueva_config.update(fase2_params)
    nueva_config.update(fase3_params)

    # 3. Preparar y aplicar overrides del Experimento/Hipótesis (Soberanía Nivel 3 - GANADOR)
    overrides = {k: v for k, v in hipotesis.items() if not k.startswith("_")}
    
    if "_nombre" in hipotesis:
        overrides["_nombre"] = hipotesis["_nombre"]
    if "_descripcion" in hipotesis:
        overrides["_descripcion"] = hipotesis["_descripcion"]

    # Garantizar integridad del benchmark (SPY el primero)
    if "tickers" in overrides:
        nuevos_tickers = list(overrides["tickers"])
        if "SPY" in nuevos_tickers:
            nuevos_tickers.remove("SPY")
        nuevos_tickers.insert(0, "SPY")
        overrides["tickers"] = nuevos_tickers
    
    # Aplicar Hipótesis al diccionario temporal
    nueva_config.update(overrides)
    
    # 4. Sincronizar comisiones según los tickers FINALES de la hipótesis
    tickers_actuales = nueva_config.get('tickers', [])
    nueva_config['comisiones'] = [MAPA_SLIPPAGE.get(t, MAPA_SLIPPAGE["DEFAULT"]) for t in tickers_actuales] + [0.0]
    
    # Evitar envenenar el config.json físico durante experimentos
    nueva_config['_no_persistir'] = True

    # 5. SINCRONIZACIÓN NUCLEAR EN MEMORIA (IN-PLACE)
    # Vaciamos el diccionario original de CONFIG y le inyectamos los nuevos datos.
    # Así TODOS los módulos que lo hayan importado verán el cambio sin importar la caché de Python.
    global CONFIG
    CONFIG.clear()
    CONFIG.update(nueva_config)

    print(f"[*] Jerarquía aplicada: Base -> Fases -> Hipótesis. Soberanía: {hipotesis.get('_nombre', 'Manual')}")



# ─────────────────────────────────────────────────────────────
# ETAPAS DEL PIPELINE
# ─────────────────────────────────────────────────────────────

def ejecutar_fase_1() -> bool:
    _banner("🔬 FASE 1 — Meta-optimización de la Métrica de Fitness")
    try:
        from analisis.fase_1_metrica import run_optimization
        _resetear_caches_fase1()
        run_optimization()
        _limpiar_memoria()
        return True
    except KeyboardInterrupt:
        print("\n[!] Fase 1 interrumpida por el usuario.")
        return False
    except Exception:
        print(f"\n[ERROR] Fase 1 ha fallado:")
        traceback.print_exc()
        return False


def ejecutar_fase_2() -> bool:
    _banner("⚙️  FASE 2 — Optimización Bayesiana (Arquitectura + Riesgo)")
    try:
        from analisis.fase_2_hiperparametros import run_fase_2
        run_fase_2()
        _limpiar_memoria()
        return True
    except KeyboardInterrupt:
        print("\n[!] Fase 2 interrumpida por el usuario.")
        return False
    except Exception:
        print(f"\n[ERROR] Fase 2 ha fallado:")
        traceback.print_exc()
        return False


def ejecutar_fase_3() -> bool:
    _banner("🏆 FASE 3 — Walk-Forward Académico (Examen Final)")
    try:
        from analisis.fase_3_walkforward import run_fase_3
        run_fase_3()
        _limpiar_memoria()
        return True
    except KeyboardInterrupt:
        print("\n[!] Fase 3 interrumpida por el usuario.")
        return False
    except Exception:
        print(f"\n[ERROR] Fase 3 ha fallado:")
        traceback.print_exc()
        return False


def ejecutar_fase_4() -> bool:
    _banner("🏭 FASE 4 — Optimización de Robustez y Meta-Comité")
    try:
        from analisis.fase_4_optimizacion import run_fase_4_optimizacion
        run_fase_4_optimizacion()
        _limpiar_memoria()
        return True
    except KeyboardInterrupt:
        print("\n[!] Fase 4 interrumpida por el usuario.")
        return False
    except Exception:
        print(f"\n[ERROR] Fase 4 ha fallado:")
        traceback.print_exc()
        return False


# ─────────────────────────────────────────────────────────────
# MODOS DE EJECUCIÓN
# ─────────────────────────────────────────────────────────────

def pipeline_completo(saltar_f1: bool = False, saltar_f2: bool = False):
    """Pipeline estándar: F1 → F2 → F3 → F4 con la config actual."""
    t_start = time.time()
    log_buffer = io.StringIO()
    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, log_buffer)
    
    try:
        _banner("🚀 ALPHA HUNTER — PIPELINE COMPLETO", ancho=70)
        print(f"  Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Modo: {'F1→F2→F3→F4' if not (saltar_f1 or saltar_f2) else ('F2→F3→F4' if not saltar_f2 else 'Solo F3→F4')}")

        if not saltar_f1 and not saltar_f2:
            ok = ejecutar_fase_1()
            if not ok:
                print("[!] Pipeline abortado en Fase 1.")
                return

        if not saltar_f2:
            ok = ejecutar_fase_2()
            if not ok:
                print("[!] Pipeline abortado en Fase 2.")
                return

        ok = ejecutar_fase_3()
        if not ok:
            print("[!] Pipeline terminado con errores en Fase 3.")
            return

        ok = ejecutar_fase_4()
        if not ok:
            print("[!] Pipeline terminado con errores en Fase 4.")
            return

        elapsed = time.time() - t_start
        _banner(f"✅ PIPELINE COMPLETADO en {elapsed/60:.1f} minutos")
        
        # Guardar log completo si tenemos la ruta
        path = CONFIG.get('_last_ejecucion_dir')
        if path:
            with open(os.path.join(path, "resultados.txt"), "w", encoding="utf-8") as f:
                f.write(log_buffer.getvalue())
    finally:
        sys.stdout = original_stdout


def pipeline_multi_hipotesis():
    """
    Modo laboratorio: prueba múltiples hipótesis en secuencia y
    genera un CSV comparativo con los resultados de cada una.
    """
    _banner("🧪 MODO MULTI-HIPÓTESIS — LABORATORIO DE ESTRATEGIAS", ancho=70)
    print(f"  Total de hipótesis a probar: {len(HIPOTESIS)}")
    print(f"  Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    for i, hip in enumerate(HIPOTESIS):
        import io
        log_buffer = io.StringIO()
        original_stdout = sys.stdout
        sys.stdout = Tee(original_stdout, log_buffer)
        CONFIG['_last_ejecucion_dir'] = None
        
        nombre      = hip.get("_nombre", f"hipotesis_{i+1}")
        descripcion = hip.get("_descripcion", "")
        t_hip_start = time.time()

        try:
            _banner(f"[{i+1}/{len(HIPOTESIS)}] Hipótesis: {nombre}", ancho=70)
            print(f"  Descripción: {descripcion}")
            
            # OBTENER OVERRIDES PUROS para imprimir
            overrides = {k: v for k, v in hip.items() if not k.startswith("_")}
            print(f"  Overrides:   {overrides if overrides else '(ninguno — config base)'}")

            # 1. APLICAR SOBERANÍA Y LIMPIAR CACHÉS GLOBALES (Fundamental para divergencia)
            _aplicar_hipotesis_a_config(hip)
            _resetear_datos_globales()
            _resetear_caches_fase1()

            # --- AUDITORÍA VISUAL DE PARÁMETROS CRÍTICOS ---
            print("\n" + "📊" * 5 + f" AUDITORÍA: {nombre} " + "📊" * 5)
            print(f"   > Tickers   : {CONFIG.get('tickers', 'N/A')}")
            print(f"   > w_ret     : {CONFIG.get('w_ret', 'N/A')} | w_mdd: {CONFIG.get('w_mdd', 'N/A')} | w_cobardia: {CONFIG.get('w_cobardia', 'N/A')}")
            print(f"   > Rebalanceo: {CONFIG.get('umbral_rebalanceo', 'N/A')} | Kelly: {CONFIG.get('kelly_fraction', 'N/A')}")
            print("=" * 60 + "\n")
            # -----------------------------------------------

            # 2. EJECUTAR FASES SEGÚN EL FLAG _solo_fase DE LA HIPÓTESIS
            solo_fase = hip.get("_solo_fase", 0) # 0=todo, 2=F2+F3, 3=F3

            ok_f1 = True
            if solo_fase < 2:
                ok_f1 = ejecutar_fase_1()
                if not ok_f1:
                    print(f"[!] Hipótesis '{nombre}' abortada en Fase 1. Continuando con la siguiente...")
                    continue

            ok_f2 = True
            if solo_fase < 3:
                ok_f2 = ejecutar_fase_2()
                if not ok_f2:
                    print(f"[!] Hipótesis '{nombre}' abortada en Fase 2. Continuando con la siguiente...")
                    continue

            ok_f3 = ejecutar_fase_3()
            if ok_f3:
                ok_f4 = ejecutar_fase_4()
                if ok_f4:
                    elapsed = time.time() - t_hip_start
                    print(f"[✔] Hipótesis '{nombre}' completada con éxito en {elapsed/60:.1f} min.")
                else:
                    print(f"[!] Hipótesis '{nombre}' terminó con errores en Fase 4.")
            else:
                print(f"[!] Hipótesis '{nombre}' terminó con errores en Fase 3.")

            # Guardar log completo en resultados.txt
            path = CONFIG.get('_last_ejecucion_dir')
            if path:
                with open(os.path.join(path, "resultados.txt"), "w", encoding="utf-8") as f:
                    f.write(log_buffer.getvalue())

        except KeyboardInterrupt:
            sys.stdout = original_stdout
            print(f"\n[!] Experimento interrumpido por el usuario en hipótesis '{nombre}'.")
            break
        except Exception:
            sys.stdout = original_stdout
            print(f"\n[ERROR] Hipótesis '{nombre}' ha fallado con una excepción inesperada:")
            import traceback
            traceback.print_exc()
            continue
        finally:
            sys.stdout = original_stdout
            _limpiar_memoria()

    _banner("🏁 EXPERIMENTO COMPLETADO")
    print(f"  Resultados guardados en: ejecuciones/registro_ejecuciones.csv")


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Alpha Hunter — Master Runner v1.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--hipotesis",
        action="store_true",
        help="Activa el modo multi-experimento (itera sobre HIPOTESIS definidas en el script).",
    )
    parser.add_argument(
        "--solo-f2",
        action="store_true",
        dest="solo_f2",
        help="Salta la Fase 1 y ejecuta solo F2 + F3.",
    )
    parser.add_argument(
        "--solo-f3",
        action="store_true",
        dest="solo_f3",
        help="Ejecuta únicamente la Fase 3 (Walk-Forward).",
    )

    args = parser.parse_args()

    if args.hipotesis:
        pipeline_multi_hipotesis()

    elif args.solo_f3:
        pipeline_completo(saltar_f1=True, saltar_f2=True)
    elif args.solo_f2:
        pipeline_completo(saltar_f1=True, saltar_f2=False)
    else:
        pipeline_completo()