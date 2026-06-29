import os
import json
import csv
from datetime import datetime
import numpy as np
import hashlib


# ─── Utilidad de serialización ────────────────────────────────────────────────
def _serializable(v):
    """Convierte tipos numpy/torch a tipos nativos de Python."""
    if isinstance(v, (np.int64, np.int32)):
        return int(v)
    if isinstance(v, (np.float64, np.float32)):
        return float(v)
    return v


def _clean_config(config: dict) -> dict:
    """Devuelve una copia del dict limpia de tipos no serializables."""
    return {k: _serializable(v) for k, v in config.items()
            if isinstance(v, (list, dict, str, int, float, bool, type(None)))
            or isinstance(v, (np.int64, np.int32, np.float64, np.float32))}


# ─── Mapa: campos de config que pertenecen a cada fase ────────────────────────
_FASE_FIELDS = {
    "fase_1": {"w_ret", "w_mdd", "w_sharpe_dif", "w_sortino", "w_linealidad", "w_l2", "w_turnover", "w_cobardia", "w_dominancia", "w_decorrel"},
    "fase_2": {"generaciones", "umbral_rebalanceo", "tasa_mutacion",
               "fuerza_mutacion", "ratio_inmigrantes", "poblacion", "ocultas",
               "train_size", "step_size", "kelly_fraction", "umbral_miedo_macro",
               "feature_dropout_rate"},  # ← Opción E: optimizado en Fase 2
    "fase_3": {"n_mejores"},
}


# ─── Funciones de guardado originales (sin cambios de interfaz) ───────────────

def crear_directorio_modelo(base_path="modelos"):
    """Crea una carpeta única para el modelo basada en la fecha y hora."""
    os.makedirs(base_path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ruta_modelo = os.path.join(base_path, f"modelo_{timestamp}")
    os.makedirs(ruta_modelo)
    return ruta_modelo, timestamp


def guardar_datos_modelo(ruta, pesos, config):
    """Guarda los pesos del ensamble y la configuración en la carpeta del modelo."""
    np.save(os.path.join(ruta, "mejor_ensemble.npy"), pesos)
    with open(os.path.join(ruta, "config.json"), "w") as f:
        json.dump(_clean_config(config), f, indent=4)


def actualizar_registro_csv(base_path, timestamp, metricas):
    """Actualiza o crea el archivo CSV con el registro de todos los modelos."""
    archivo_csv = os.path.join(base_path, "registro_modelos.csv")
    fila = {"id_modelo": f"modelo_{timestamp}",
            "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            **metricas}
    file_exists = os.path.isfile(archivo_csv)
    with open(archivo_csv, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fila.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(fila)
    print(f"[OK] Registro actualizado en: {archivo_csv}")


def registrar_modelo_completo(pesos, config, metricas, base_path="modelos", csv_path=None):
    """Orquestador para guardar todo el proceso de un modelo."""
    if csv_path is None:
        csv_path = base_path
    ruta, ts = crear_directorio_modelo(base_path)
    guardar_datos_modelo(ruta, pesos, config)
    actualizar_registro_csv(csv_path, ts, metricas)
    print(f"[OK] Modelo guardado en: {ruta}")
    return ruta


# ─── Guardado del estado del pipeline ────────────────────────────────────────

def guardar_pipeline_state(config: dict, fase: str) -> None:
    """
    Persiste el estado del pipeline directamente en config.json.
    FIX BUG 1: Si estamos en modo experimento (no_persistir), abortamos la escritura física.
    """
    if config.get('_no_persistir', False):
        print(f"[*] Modo Experimento: Omitiendo persistencia física de {fase} en config.json")
        return
        
    fase_key = fase.lower()

    # ── 1. Extraer sólo los campos que produce esta fase ──────────────
    owned = _FASE_FIELDS.get(fase_key, set())
    params_fase = {k: _serializable(v)
                   for k, v in config.items()
                   if k in owned and v is not None}

    # ── 2. Delegar en config.py → escribe en FASE_N y en PARAMETROS_OPTIMIZABLES ─────
    if params_fase:
        try:
            from config import save_optimized_params
            save_optimized_params(fase, params_fase)
            print(f"[*] Parámetros de fitness a guardar: "
              f"w_ret={config.get('w_ret', 0.0):.3f}, "
              f"w_mdd={config.get('w_mdd', 0.0):.3f}, "
              f"w_sharpe_dif={config.get('w_sharpe_dif', 0.0):.3f}, "
              f"w_sortino={config.get('w_sortino', 0.0):.3f}, "
              f"w_linealidad={config.get('w_linealidad', 0.0):.3f}")
        except Exception as e:
            print(f"[!] No se pudo actualizar config.json: {e}")
