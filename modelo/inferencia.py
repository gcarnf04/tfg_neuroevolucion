import numpy as np
from typing import Optional, Tuple, List

# ═══════════════════════════════════════════════════════════════════
# FUNCIÓN CENTRAL: ensamblar_y_predecir — v3.1 (Democracia Aritmética)
# ═══════════════════════════════════════════════════════════════════
def ensamblar_y_predecir(
    adns: List[np.ndarray],
    X_oos_norm: np.ndarray,
    config: dict,
    inf_model,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pipeline de inferencia con Democracia de Pesos (v6.16):
        1. Recoger pesos ya normalizados de cada experto (o calcularlos)
        2. Realizar la media aritmética de las carteras
    """
    all_opinions_raw = kwargs.get('all_opinions_raw')
    
    if all_opinions_raw is None:
        # Si no nos pasan los pesos precalculados, los calculamos uno a uno
        # Nota: Aquí inf_model.predecir ya debería devolver pesos ReLU-Normalizados 
        # si el motor_ga subyacente ha sido actualizado.
        props = []
        dropout_rate = config.get('feature_dropout_rate', 0.25)
        in_dim = X_oos_norm.shape[1]
        proy_base = np.arange(1, in_dim + 1) * 123.456
        
        for adn in adns:
            semilla = np.sum(adn[:5])
            mask = (np.abs(np.sin(semilla * proy_base)) > dropout_rate).astype(np.float32)
            X_masked = X_oos_norm * mask
            
            inf_model.set_pesos_aplanados(adn)
            raw_weights = inf_model.predecir(X_masked) # Ya normalizados por ReLU+Linear en el motor
            props.append(raw_weights)
        all_opinions_raw = np.array(props, dtype=np.float64)
    
    # ALPHA v6.16: Promedio Aritmético de Carteras
    p_fin = np.mean(all_opinions_raw, axis=0)    # (T, A)
    
    # Re-normalización de seguridad: garantiza que la cartera final sume exactamente 100%
    suma_total = np.sum(p_fin, axis=1, keepdims=True)
    p_fin = np.where(suma_total > 1e-9, p_fin / (suma_total + 1e-10), p_fin)

    return p_fin.astype(np.float32), np.zeros((p_fin.shape[0], 1))

# ═══════════════════════════════════════════════════════════════════
# simular_trading_vectorizado — v3.0
# ═══════════════════════════════════════════════════════════════════
def simular_trading_vectorizado(p_final, R_oos_np, config):
    """
    Simula drift + rebalanceo con Slippage Dinámico (Market Impact Model).
    """
    T = len(R_oos_np)
    A = p_final.shape[1]
    umbral = config.get('umbral_rebalanceo', 0.10)
    comisiones_np = np.array(config['comisiones'])
    impacto_cuadratico = config.get('k_slippage', 0.5) 
    
    p_real = np.zeros_like(p_final[:-1])
    p_real[0] = p_final[0]
    costes = np.zeros(T - 1)
    
    p_actual = p_final[0].copy()
    
    # Coste inicial (Capa 1: entrada al mercado)
    desv_ini = np.abs(p_actual)
    penalizacion_ini = 1.0 + (desv_ini * impacto_cuadratico)
    costes[0] = np.sum(desv_ini * comisiones_np * penalizacion_ini)
    
    for t in range(1, T - 1):
        p_drift = p_actual * np.maximum(1.0 + R_oos_np[t], 0.0001)
        p_drift /= max(np.sum(p_drift), 1e-9)
        
        cambio_necesario = np.abs(p_final[t] - p_drift)
        
        if np.max(cambio_necesario) > umbral:
            penalizacion_tamano = 1.0 + (cambio_necesario * impacto_cuadratico)
            coste_trade = np.sum(cambio_necesario * comisiones_np * penalizacion_tamano)
            
            p_real[t] = p_final[t]
            costes[t] = coste_trade
            p_actual = p_final[t].copy()
        else:
            p_real[t] = p_drift
            p_actual = p_drift.copy()
            
    return np.sum(p_real * R_oos_np[1:], axis=1) - costes

def simular_trading_batch(p_batch, R_oos_np, config):
    """
    Versión batched con Slippage Dinámico.
    """
    B, T, A = p_batch.shape
    umbral = config.get('umbral_rebalanceo', 0.10)
    comisiones_np = np.array(config['comisiones'], dtype=np.float64)
    impacto_cuadratico = config.get('k_slippage', 0.5)

    if T < 3:
        return np.zeros((B, max(T - 1, 0)), dtype=np.float32)

    p_real  = np.zeros((B, T - 1, A), dtype=np.float64)
    costes  = np.zeros((B, T - 1),    dtype=np.float64)

    p_actual = p_batch[:, 0, :].copy().astype(np.float64)
    p_real[:, 0, :] = p_actual

    desv_inicial = np.abs(p_actual)
    penalizacion_ini = 1.0 + (desv_inicial * impacto_cuadratico)
    costes[:, 0] = np.sum(desv_inicial * comisiones_np * penalizacion_ini, axis=1)

    for t in range(1, T - 1):
        factor   = np.maximum(1.0 + R_oos_np[t].astype(np.float64), 1e-4)
        p_drift_raw = p_actual * factor[np.newaxis, :]
        suma_drift  = np.sum(p_drift_raw, axis=1, keepdims=True)
        suma_drift  = np.where(suma_drift < 1e-9, 1.0, suma_drift)
        p_drift = p_drift_raw / suma_drift

        target    = p_batch[:, t, :].astype(np.float64)
        cambio_necesario = np.abs(target - p_drift)
        desv_max   = np.max(cambio_necesario, axis=1)
        mask_reb   = desv_max > umbral

        penalizacion_tamano = 1.0 + (cambio_necesario * impacto_cuadratico)
        coste_trade = np.sum(cambio_necesario * comisiones_np * penalizacion_tamano, axis=1)
        costes[:, t] = np.where(mask_reb, coste_trade, 0.0)

        p_actual = np.where(mask_reb[:, np.newaxis], target, p_drift)
        p_real[:, t, :] = p_actual

    R_next = R_oos_np[1:].astype(np.float64)
    gross = np.einsum('bta,ta->bt', p_real, R_next)
    return (gross - costes).astype(np.float32)
