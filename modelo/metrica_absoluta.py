import torch
import numpy as np

def calcular_fitness_torch(rets_netos, rets_spy,
                           w_ret=0.60, w_mdd=0.40,
                           w_cobardia=0.0, w_dominancia=0.0,
                           w_decorrel=0.0, w_sharpe_dif=0.0, w_sortino=0.0,
                           w_oportunidad=0.0,
                           w_l2=0.05, w_turnover=0.5,
                           target_return_anual=0.12,
                           w_linealidad=0.0,
                           l2_penalty=0.0, turnover_medio=0.0,
                           benchmark_info=None):
    """
    Metrica Absoluta v6.15 — Optimización de Retorno, Riesgo, Fricciones y Linealidad Target.
    """
    n_periodos = rets_netos.shape[1]
    device_t = rets_netos.device

    # 1. Retorno Anualizado Absoluto con Utilidad Saturada (v6.8)
    ret_acu = torch.prod(1.0 + rets_netos, dim=1)
    ret_anual = (torch.clamp(ret_acu, min=1e-6) ** (252.0 / n_periodos)) - 1.0
    
    # Objetivo: Alcanzar el Target (ej. 12%) con el mínimo riesgo posible.
    target = target_return_anual
    # Utilidad: crece lineal hasta el target, luego el exceso solo suma un 30% de su valor
    exceso = torch.clamp(ret_anual - target, min=0.0)
    utilidad_ret = torch.clamp(ret_anual, max=target) + (exceso * 0.3)
    
    score_ret = utilidad_ret * w_ret * 5.0

    # 2. Penalización de MDD (lineal — más interpretable y estable)
    cap = torch.cumprod(1.0 + rets_netos, dim=1)
    max_cap, _ = torch.cummax(cap, dim=1)
    mdd = torch.max(torch.abs((max_cap - cap) / (max_cap + 1e-9)), dim=1)[0]
    castigo_mdd = mdd * w_mdd * 5.0

    # 3. Calidad de ejecución: Sharpe & Sortino
    volatilidad = torch.std(rets_netos, dim=1) * torch.sqrt(torch.tensor(252.0, device=device_t))
    ret_medio_anual = torch.mean(rets_netos, dim=1) * 252.0
    sharpe = ret_medio_anual / (volatilidad + 1e-9)
    score_sharpe = sharpe * w_sharpe_dif * 0.5

    rets_neg = torch.clamp(rets_netos, max=0.0)
    vol_down = torch.std(rets_neg, dim=1) * torch.sqrt(torch.tensor(252.0, device=device_t))
    sortino = ret_medio_anual / (vol_down + 1e-9)
    score_sortino = sortino * w_sortino * 0.5

    # 4. Linealidad contra Target (v6.15)
    score_linealidad = 0.0
    if w_linealidad > 0:
        # R² contra la línea objetivo: qué tan bien sigue la curva de capital la línea del 12% anual
        # Pendiente objetivo diaria (logarítmica para coherencia con log_equity)
        m_target = torch.log(torch.tensor(1.0 + target_return_anual, device=device_t)) / 252.0
        
        t_idx = torch.arange(n_periodos, device=device_t, dtype=torch.float32)
        log_equity = torch.log(cap + 1e-9)
        
        # Línea ideal que empieza en 0 (log(1)) y sube con pendiente m_target
        y_target = t_idx * m_target
        
        # SS_res: desviación de la curva real respecto a la línea ideal
        ss_res_target = torch.sum((log_equity - y_target)**2, dim=1)
        
        # SS_tot: varianza total de log_equity respecto a su propia media (R² estándar)
        log_equity_mean = log_equity.mean(dim=1, keepdim=True)
        ss_tot_target = torch.sum((log_equity - log_equity_mean)**2, dim=1) + 1e-9 
        
        # Métrica de fidelidad al target (0 a 1)
        r2 = torch.clamp(1.0 - (ss_res_target / ss_tot_target), min=0.0, max=1.0)
        score_linealidad = r2 * w_linealidad * 2.0

    # 5. Fricciones y Penalizaciones
    fitness = score_ret - castigo_mdd + score_sharpe + score_sortino + score_linealidad
    fitness = fitness - (w_l2 * l2_penalty) - (w_turnover * turnover_medio)

    return fitness
