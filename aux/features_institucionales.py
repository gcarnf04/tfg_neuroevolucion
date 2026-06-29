# features_institucionales.py
"""
Refactor Alpha v3.2: Escalado Robusto y Re-indexación Temporal.
- Mahalanobis (Riesgo Macro) -> idx_m
- Information Ratios (Momentum) -> idx_c
- Eigenportfolios (Estructura) -> idx_l
"""

import numpy as np
import pandas as pd
from typing import List, Tuple


# ═══════════════════════════════════════════════════════════════════
# UTILS: ESCALADO ROBUSTO SIN LOOK-AHEAD
# ═══════════════════════════════════════════════════════════════════
def _robust_scale_no_lookahead(data: np.ndarray, idx_fin_train: int = None) -> np.ndarray:
    """
    Normaliza cada columna usando RobustScaler (mediana y p10-p90).
    Si idx_fin_train es None, usa los primeros 2/3 de los datos como fallback.
    """
    T, F = data.shape
    idx_split = idx_fin_train if idx_fin_train is not None else int(T * 2 // 3)
    data_scaled = data.copy()

    for i in range(F):
        subset = data[:idx_split, i]
        # Filtrar ceros (warm-up) para el cálculo de estadísticas
        subset_filt = subset[np.abs(subset) > 1e-9]
        
        if len(subset_filt) < 10:
            median = 0.0
            scale = 1.0
        else:
            median = np.median(subset_filt)
            p10 = np.percentile(subset_filt, 10)
            p90 = np.percentile(subset_filt, 90)
            scale = (p90 - p10) / 2.0  # Dividimos por 2 para que la escala sea más "amigable"
            if scale < 1e-6: scale = 1.0
            
        data_scaled[:, i] = (data[:, i] - median) / scale
        
    return np.clip(data_scaled, -5.0, 5.0).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════
# FEATURE 1: MAHALANOBIS DISTANCE RODANTE
# ═══════════════════════════════════════════════════════════════════
def _mahalanobis_rolling(
    R: np.ndarray,
    ventana: int = 63,
    regularizacion: float = 1e-4,
) -> np.ndarray:
    """
    Distancia de Mahalanobis rodante para detectar anomalías multivariantes.
    MD(t) = sqrt( (r_t - μ)ᵀ Σ⁻¹ (r_t - μ) )
    """
    T, A = R.shape
    maha = np.zeros((T, 1), dtype=np.float32)

    for t in range(ventana, T):
        window = R[t - ventana:t, :]                        # (ventana, A)
        mu     = window.mean(axis=0)                         # (A,)
        cov    = np.cov(window.T)                            # (A, A)
        cov    += regularizacion * np.eye(A)                 # Regularización Tikhonov

        try:
            cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            cov_inv = np.eye(A)

        diff = R[t, :] - mu                                  # (A,)
        md_sq = float(diff @ cov_inv @ diff)
        maha[t, 0] = np.sqrt(max(md_sq, 0.0))

    return maha  # (T, 1)


# ═══════════════════════════════════════════════════════════════════
# FEATURE 2: INFORMATION RATIO RODANTE POR ACTIVO
# ═══════════════════════════════════════════════════════════════════
def _rolling_information_ratio(
    R: np.ndarray,
    ventana_corta: int = 21,
    ventana_larga: int = 63,
) -> np.ndarray:
    """
    ALPHA v4.9: Eliminado el IR del benchmark (columna 0) y del Cash (última columna).
    """
    T, A = R.shape
    # Devolvemos A - 2 columnas (omitimos benchmark y cash)
    ir_features = np.zeros((T, A - 2), dtype=np.float32)

    for ventana in [ventana_corta, ventana_larga]:
        for t in range(ventana, T):
            bench = R[t - ventana:t, 0]          # Benchmark = SPY
            ret_bench_anual = (np.prod(1.0 + bench) ** (252.0 / ventana)) - 1.0

            # Omitimos benchmark (0) y cash (A-1)
            for i in range(1, A - 1):
                activo = R[t - ventana:t, i]
                ret_activo_anual = (np.prod(1.0 + activo) ** (252.0 / ventana)) - 1.0
                tracking_error = np.std(activo - bench) * np.sqrt(252.0) + 1e-6
                ir = (ret_activo_anual - ret_bench_anual) / tracking_error
                ir_features[t, i - 1] += ir

    ir_features /= 2.0  # Promedio de ventanas
    return ir_features  # (T, A - 2)


# ═══════════════════════════════════════════════════════════════════
# FEATURE 3: EIGENPORTFOLIO EXPOSURE (PCA RODANTE)
# ═══════════════════════════════════════════════════════════════════
def _eigenportfolio_rolling(
    R: np.ndarray,
    ventana: int = 63,
    n_componentes: int = 3,
) -> np.ndarray:
    """
    Proyección de los retornos sobre los N eigenvectores principales (PCA).
    """
    T, A = R.shape
    n_comp = min(n_componentes, A)
    eigen_expo = np.zeros((T, n_comp), dtype=np.float32)

    for t in range(ventana, T):
        window = R[t - ventana:t, :]
        mu       = window.mean(axis=0)
        centered = window - mu 
        cov      = centered.T @ centered / (ventana - 1)
        cov     += 1e-6 * np.eye(A)

        try:
            eigenvals, eigenvecs = np.linalg.eigh(cov)
            idx_sort   = np.argsort(eigenvals)[::-1]
            eigenvecs  = eigenvecs[:, idx_sort]
            top_vecs   = eigenvecs[:, :n_comp]
            
            r_today_centered = R[t, :] - mu
            eigen_expo[t, :] = r_today_centered @ top_vecs
        except np.linalg.LinAlgError:
            eigen_expo[t, :] = 0.0

    return eigen_expo  # (T, n_comp)


# ═══════════════════════════════════════════════════════════════════
# FUNCIÓN DE VALIDACIÓN DE ÍNDICES
# ═══════════════════════════════════════════════════════════════════
def validate_indices(X: np.ndarray, idx_c: List[int], idx_m: List[int], idx_l: List[int]):
    """
    Verifica que no haya solapamiento entre índices y que cubran todo el espectro de X.
    """
    T, F = X.shape
    total_idx = len(idx_c) + len(idx_m) + len(idx_l)
    
    if total_idx != F:
        raise ValueError(f"CRÍTICO: El número total de índices ({total_idx}) no coincide con el número de columnas de X ({F}).")
    
    # Comprobar solapamiento
    set_c, set_m, set_l = set(idx_c), set(idx_m), set(idx_l)
    if not set_c.isdisjoint(set_m) or not set_c.isdisjoint(set_l) or not set_m.isdisjoint(set_l):
        raise ValueError("CRÍTICO: Solapamiento detectado entre idx_c, idx_m e idx_l.")
        
    # Comprobar rangos
    combined = sorted(list(set_c | set_m | set_l))
    if combined[0] != 0 or combined[-1] != F - 1:
        raise ValueError(f"CRÍTICO: Los índices no cubren el rango [0, {F-1}]. Rango actual: [{combined[0]}, {combined[-1]}]")
    
    print("[✔] Validación de índices exitosa.")


# ═══════════════════════════════════════════════════════════════════
# FUNCIÓN PRINCIPAL DE INTEGRACIÓN
# ═══════════════════════════════════════════════════════════════════
def agregar_features_institucionales(
    X: np.ndarray,
    R: np.ndarray,
    idx_c: List[int],
    idx_m: List[int],
    idx_l: List[int],
    n_pca: int = 3,
    idx_fin_train: int = None,
    aplicar_escalado_global: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[int], List[int], List[int]]:
    T, F = X.shape
    n_activos = R.shape[1]

    # 1. Cálculos base
    print("[*] Calculando Mahalanobis Distance rodante...")
    maha = _mahalanobis_rolling(R)
    print("[*] Calculando Information Ratio rodante...")
    ir = _rolling_information_ratio(R)
    print(f"[*] Calculando Eigenportfolio Exposure ({n_pca} comp)...")
    eigen = _eigenportfolio_rolling(R, n_componentes=n_pca)

    # 2. ESCALADO ROBUSTO
    if aplicar_escalado_global:
        maha_scaled  = _robust_scale_no_lookahead(maha,  idx_fin_train=idx_fin_train)
        ir_scaled    = _robust_scale_no_lookahead(ir,    idx_fin_train=idx_fin_train)
        eigen_scaled = _robust_scale_no_lookahead(eigen, idx_fin_train=idx_fin_train)
    else:
        maha_scaled  = maha.astype(np.float32)
        ir_scaled    = ir.astype(np.float32)
        eigen_scaled = eigen.astype(np.float32)

    # --- FIX LEAKAGE: SHIFT(1) A FEATURES INSTITUCIONALES ---
    maha_scaled  = pd.DataFrame(maha_scaled).shift(1).fillna(0).values.astype(np.float32)
    ir_scaled    = pd.DataFrame(ir_scaled).shift(1).fillna(0).values.astype(np.float32)
    eigen_scaled = pd.DataFrame(eigen_scaled).shift(1).fillna(0).values.astype(np.float32)

    # 3. INTEGRACIÓN Y RE-INDEXACIÓN
    X_aug = np.hstack([X, maha_scaled, ir_scaled, eigen_scaled]).astype(np.float32)
    
    curr_f = F
    # Mahalanobis (1) -> idx_m
    idx_m_new = list(idx_m) + [curr_f]
    curr_f += 1
    
    # Information Ratios (n_activos - 2) -> idx_c (Omitimos benchmark y cash)
    idx_c_new = list(idx_c) + list(range(curr_f, curr_f + n_activos - 2))
    curr_f += n_activos - 2
    
    # Eigenportfolios (n_pca) -> idx_l
    idx_l_new = list(idx_l) + list(range(curr_f, curr_f + n_pca))
    
    # 4. VALIDACIÓN
    validate_indices(X_aug, idx_c_new, idx_m_new, idx_l_new)

    print(f"[✔] Alpha v5.0: {X_aug.shape[1]-F} features institucionales añadidas (Mahalanobis + IR + Eigenportfolio).")
    return X_aug, R, idx_c_new, idx_m_new, idx_l_new
