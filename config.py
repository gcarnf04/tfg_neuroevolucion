"""
config.py — Adaptador de config.json para retrocompatibilidad.

Fuente de verdad: config.json › sección "PARAMETROS_OPTIMIZABLES".
Los scripts hacen `from config import CONFIG` sin cambios.

Flujo de datos:
  PARAMETROS_ESTATICOS (valores fijos)
    ↓ (inicializa)
  PARAMETROS_OPTIMIZABLES  ← sección dinámica en tiempo de ejecución
    ↑ (sobreescrita a medida que avanzan las fases)
  FASE_1 / FASE_2 / FASE_3  ← historial inmutable de resultados
"""
import os
import json

_DIR = os.path.dirname(os.path.abspath(__file__))
_CONFIG_JSON_PATH = os.path.join(_DIR, "config.json")


# UNIVERSO CAZADOR DE ALPHA V2 (Sin redundancias, rotación sectorial perfecta)
TICKERS_ACTIVOS = ['SPY', 'QQQ', 'GLD', 'EEM', 'XLE', 'IWM']

# ─── Mapa de comisiones por activo (fuente de verdad dinámica) ──────────────
MAPA_SLIPPAGE = {
    # Índices Mayores (Alta liquidez: 5bps)
    "SPY": 0.0005, "QQQ": 0.0005, "DIA": 0.0005, "VTI": 0.0005,
    
    # Sectores SPDR (Liquidez media: 10bps)
    "XLK": 0.0005, "XLF": 0.0010, "XLE": 0.0010, "XLV": 0.0010,
    "XLP": 0.0010, "XLY": 0.0010, "XLI": 0.0010, "XLB": 0.0010,
    "XLU": 0.0010, "SMH": 0.0010, "IWM": 0.0010,
    
    # Commodities / Alternativos (15bps)
    "GLD": 0.0015, "SLV": 0.0015, "USO": 0.0015, "UNG": 0.0015,
    
    # Bonos / Tesoros (Dinámico)
    "TLT": 0.0008, "IEF": 0.0008, "SHY": 0.0006,
    
    # Cash / Equivalentes (0-5bps)
    "SHV": 0.0000, "BIL": 0.0000,
    
    "DEFAULT": 0.0015
}


def _load_raw() -> dict:
    with open(_CONFIG_JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_config() -> dict:
    raw = _load_raw()
    cfg: dict = {}

    # 1. Cargar secciones semánticas (Nueva Arquitectura v2.0)
    # Estas secciones contienen los valores operativos (no rangos de búsqueda)
    secciones_operativas = ["UNIVERSO", "GA", "MINADO", "EJECUCION", "OPTIMIZABLES", "PARAMETROS_ESTATICOS"]
    for seccion in secciones_operativas:
        cfg.update(raw.get(seccion, {}))

    # cpus: null en JSON → os.cpu_count() - 1
    if cfg.get("cpus") is None:
        cfg["cpus"] = os.cpu_count() - 1

    # 2. SOBREESCRIBIR con resultados de fases previas (Cascada de Inteligencia)
    # Solo sobreescribimos las variables OPTIMIZADAS de las fases para evitar colisiones operativas
    _OPT_KEYS = {
        "w_ret", "w_mdd", "w_cobardia", "w_dominancia", "w_decorrel", "w_sharpe_dif", 
        "w_sortino", "w_oportunidad", "w_l2", "w_turnover", "w_linealidad",
        "ocultas", "train_size", "step_size", "umbral_rebalanceo", "conviccion_minima", 
        "n_comite", "kelly_fraction", "feature_dropout_rate", "k_comite_meta"
    }

    fases_ordenadas = ["FASE_1", "FASE_2", "FASE_3", "FASE_4"]
    for fase_name in fases_ordenadas:
        fase_data = raw.get(fase_name, {})
        for k, v in fase_data.items():
            if k in _OPT_KEYS:
                cfg[k] = v

    # 3. ALIASES DE RETROCOMPATIBILIDAD (Saneamiento)
    _ALIASES = {
        'batch_size':              'eval_chunk_size',
        'modelos_objetivo':        'n_fundadores',
        'n_mejores':               'n_comite',
        'kelly_max_weight':        'kelly_fraction',
    }
    for viejo, nuevo in _ALIASES.items():
        if nuevo in cfg:
            cfg[viejo] = cfg[nuevo]

    # comisiones: derivar del mapa global si no existen
    if not cfg.get("comisiones"):
        cfg["comisiones"] = (
            [MAPA_SLIPPAGE.get(t, MAPA_SLIPPAGE["DEFAULT"]) for t in cfg.get("tickers", [])] + [0.0]
        )

    return cfg


# ─── API pública ─────────────────────────────────────────────────────────────

CONFIG: dict = _build_config()


def reload_config() -> dict:
    """Recarga config.json en caliente y actualiza CONFIG en memoria."""
    global CONFIG
    CONFIG = _build_config()
    return CONFIG

def get_fase_config(fase_name: str) -> dict:
    """
    Devuelve los parámetros de una fase combinándolos con los valores globales por defecto
    definidos en las secciones operativas principales si no están especificados en la fase.
    """
    raw = _load_raw()
    fase_key = fase_name.upper()
    fase_data = raw.get(fase_key, {})

    # Combinación recursiva con fallback al config global
    cfg_fase = _build_config()
    for k, v in fase_data.items():
        if not k.startswith("_"):
            cfg_fase[k] = v
    return cfg_fase


def save_optimized_params(fase: str, params: dict) -> None:
    """
    Escribe los parámetros optimizados de una fase en config.json:
      1. En la sección FASE_N  → historial inmutable.
      2. En la sección PARAMETROS_OPTIMIZABLES → parámetros operativos en uso.

    fase: 'FASE_1' | 'FASE_2' | 'FASE_3'
    params: dict con las claves tal cual deben quedar en la configuración
            (no las claves *_opt, sino los nombres reales de config).
    """
    from datetime import datetime

    raw = _load_raw()

    # ── Redondeo dinámico de parámetros ──────────────────────────────
    rounded_params = {}
    for k, v in params.items():
        if isinstance(v, float):
            if k.startswith("w_"):
                rounded_params[k] = round(v, 2)
            else:
                rounded_params[k] = round(v, 2)
        else:
            rounded_params[k] = v

    # ── Historial: guardar en FASE_N con claves de la fase ───────────
    _OPT_SUFFIX_MAP = {
        "FASE_1": {"w_ret", "w_mdd", "w_cobardia", "w_dominancia", "w_decorrel", "w_sharpe_dif", "w_sortino", "w_oportunidad", "w_l2", "w_turnover", "w_linealidad"},
        "FASE_2": {"ocultas", "train_size", "step_size", "umbral_rebalanceo", "conviccion_minima", "n_comite", "kelly_fraction", "feature_dropout_rate"},
        "FASE_3": {"n_comite"},
        "FASE_4": {"k_comite_meta"},
    }
    fase_key = fase.upper()
    section = raw.setdefault(fase_key, {})
    owned_keys = _OPT_SUFFIX_MAP.get(fase_key, set())
    
    for k, v in rounded_params.items():
        if k in owned_keys or k == "ocultas":
            section[k] = v
            
    section["_ultima_actualizacion"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── OPTIMIZABLES: sobreescribir los parámetros que ha producido esta fase ──
    optimizables = raw.setdefault("OPTIMIZABLES", {})
    for k, v in rounded_params.items():
        if v is not None and (k in owned_keys or k == "ocultas"):
            optimizables[k] = v

    with open(_CONFIG_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(raw, f, indent=2, ensure_ascii=False)

    reload_config()
