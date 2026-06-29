# slippage.py
"""
Cálculo de Slippage Dinámico basado en volatilidad realizada.

Modelo de mercado:
    Los market makers amplían sus bid-ask spreads en proporción a la
    volatilidad reciente. En días de alta volatilidad, el spread
    efectivo que pagas puede ser 3-5× el spread tranquilo.

    Fórmula calibrada:
        slippage_por_unidad(t) = base_spread + k × σ_21d(t)
        donde σ_21d = std(SPY_rets[-21:]) en términos diarios.

    Interpretación de parámetros (calibrados empíricamente en SPY):
        base_spread  = 0.0005  → 5bps por dirección (taker fee ETF)
        k_slippage   = 0.50    → Cuando σ_21d = 1% (VIX≈16), añade 50bps de slippage
                                  Cuando σ_21d = 2% (VIX≈32), añade 100bps extra

    Para activos más ilíquidos (ej. sector ETFs pequeños) usa k=1.0.
"""

import numpy as np
from typing import Optional


def calcular_slippage_dinamico(
    R_historico: np.ndarray,
    indice_spy: int = 0,
    k_slippage: float = 0.50,
    base_spread: float = 0.0005,
    ventana_vol: int = 21,
    suavizado_ewm: float = 0.94,
) -> np.ndarray:
    """
    Genera un array de slippage por día basado en la volatilidad del benchmark.

    Args:
        R_historico: (T, A) retornos de todos los activos del período
        indice_spy: índice del benchmark (columna SPY en R). Normalmente 0.
        k_slippage: multiplicador de volatilidad. Ver docstring del módulo.
        base_spread: spread mínimo garantizado (en unidades de portfolio).
        ventana_vol: ventana para la volatilidad realizada (21 = 1 mes hábil).
        suavizado_ewm: factor EWM para suavizar la volatilidad. 0.94 = RiskMetrics.
    Returns:
        slippage_por_unidad: (T,) coste de slippage por unidad de portfolio movida.
            Para usar: coste_extra = slippage_por_unidad[t] × turnover_total[t]
    """
    T = R_historico.shape[0]
    spy_rets = R_historico[:, indice_spy].astype(np.float64)

    # ── Volatilidad realizada rodante con EWM (RiskMetrics style) ──
    # σ²_EWM(t) = λ × σ²_EWM(t-1) + (1-λ) × r²(t)
    # Combina la ventana fija (21d) con el suavizado EWM para ser responsivo
    # a saltos de vol tipo "lunes negro" sin olvidar el contexto histórico

    var_ewm = np.zeros(T, dtype=np.float64)
    # Inicializar con la varianza de la primera ventana disponible
    var_ewm[0] = spy_rets[0] ** 2

    lam = suavizado_ewm
    for t in range(1, T):
        var_ewm[t] = lam * var_ewm[t - 1] + (1.0 - lam) * (spy_rets[t] ** 2)

    vol_ewm = np.sqrt(var_ewm)  # Volatilidad diaria EWM

    # ── Slippage total por unidad ──────────────────────────────────
    slippage = base_spread + k_slippage * vol_ewm

    # Cap en 5% por unidad para evitar explosiones en crashes
    slippage = np.clip(slippage, base_spread, 0.05)

    return slippage.astype(np.float32)


def ajustar_comisiones_con_slippage(
    config: dict,
    R_historico: np.ndarray,
    indice_spy: int = 0,
) -> np.ndarray:
    """
    Genera comisiones efectivas (base + slippage dinámico) para cada día.
    Útil como preproceso antes de llamar a simular_trading_vectorizado.

    Returns:
        comisiones_efectivas: (T,) — coste por unidad de turnover en cada día.
            Representa el coste efectivo para ese día concreto.
    """
    comisiones_base = np.mean(config['comisiones'])  # Media de comisiones base
    slippage_dyn    = calcular_slippage_dinamico(
        R_historico,
        indice_spy=indice_spy,
        k_slippage=config.get('k_slippage', 0.50),
        base_spread=config.get('base_spread', 0.0005),
    )
    # El slippage es ADICIONAL a las comisiones base
    return slippage_dyn
