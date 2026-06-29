import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Añadir el path base para importar los módulos del proyecto
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aux.datos import GestorDatos
from config import CONFIG

def setup_style():
    """Configura el estilo de los gráficos para que se vean premium."""
    sns.set_theme(style="darkgrid")
    plt.rcParams['figure.facecolor'] = '#121212'
    plt.rcParams['axes.facecolor'] = '#1e1e1e'
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['axes.labelcolor'] = '#e0e0e0'
    plt.rcParams['xtick.color'] = '#b0b0b0'
    plt.rcParams['ytick.color'] = '#b0b0b0'
    plt.rcParams['text.color'] = '#ffffff'
    plt.rcParams['grid.color'] = '#2a2a2a'
    plt.rcParams['font.family'] = 'sans-serif'

def main():
    print("\n" + "="*70)
    print(" 📊 EXPLORATORY DATA ANALYSIS (EDA) - ALPHA HUNTER 📊")
    print("="*70)
    
    setup_style()
    
    # 1. Carga de Datos
    print("[*] Descargando y procesando datos...")
    gestor = GestorDatos(
        CONFIG['tickers'], 
        CONFIG['ticker_cash'], 
        CONFIG['ticker_macro'], 
        CONFIG['fecha_inicio'], 
        datetime.now().strftime("%Y-%m-%d")
    )
    
    # Obtenemos los datos listos (X: features, R: retornos)
    # Usamos escalar_global=False para ver los datos en su escala original si es posible,
    # aunque GestorDatos ya devuelve features calculadas.
    X, R, i_c, i_m, i_l = gestor.obtener_datos_listos(escalar_global=True)
    
    # Recuperar nombres de columnas de features
    # Nota: GestorDatos.features tiene los nombres antes de agregar los institucionales
    feat_names = list(gestor.features.columns)
    
    # Nombres de las features institucionales (según aux/features_institucionales.py)
    n_activos = R.shape[1]
    # ALPHA v4.9: Omitimos benchmark y cash del Information Ratio
    tickers_con_ir = CONFIG['tickers'][1:] 
    inst_names = ['Maha_Dist'] + [f'IR_{t}' for t in tickers_con_ir] + [f'Eigen_{i}' for i in range(3)]
    all_feat_names = feat_names + inst_names
    
    df_features = pd.DataFrame(X, columns=all_feat_names)
    df_returns = pd.DataFrame(R, columns=CONFIG['tickers'] + [CONFIG['ticker_cash']])
    
    print(f"[✔] Datos cargados: {X.shape[0]} observaciones, {X.shape[1]} features.")
    
    plot_dir = "eda/plots"
    os.makedirs(plot_dir, exist_ok=True)
    
    # 2. Matriz de Correlación
    print("[*] Generando Matriz de Correlación...")
    plt.figure(figsize=(20, 16))
    # Correlación de features con el retorno del SPY (primer activo)
    df_full = pd.concat([df_features, df_returns.iloc[:, 0]], axis=1)
    df_full.rename(columns={df_returns.columns[0]: 'TARGET_RET'}, inplace=True)
    
    corr = df_full.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    sns.heatmap(corr, mask=mask, cmap='vlag', center=0, 
                linewidths=.5, cbar_kws={"shrink": .5}, annot=False)
    plt.title("Matriz de Correlación de Features y Retorno Benchmark", fontsize=18, pad=20)
    plt.savefig(f"{plot_dir}/01_correlation_matrix.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Correlación Específica con el Target
    print("[*] Generando Correlación con el Target...")
    plt.figure(figsize=(10, 15))
    target_corr = corr['TARGET_RET'].sort_values(ascending=False).drop('TARGET_RET')
    colors = ['#2ca02c' if x > 0 else '#d62728' for x in target_corr]
    target_corr.plot(kind='barh', color=colors)
    plt.title("Correlación de cada Feature con el Retorno Próximo Día", fontsize=15)
    plt.xlabel("Coeficiente de Correlación")
    plt.axvline(x=0, color='white', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/02_target_correlation.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 4. Distribución de Features Clave
    print("[*] Generando Distribuciones de Features Clave...")
    # Seleccionamos unas cuantas variadas
    keys = ['VIX_Level', 'Maha_Dist', 'TNX_Level', f'{CONFIG["tickers"][0]}_Ret_D', f'{CONFIG["tickers"][0]}_Vol_21']
    keys = [k for k in keys if k in all_feat_names]
    
    fig, axes = plt.subplots(len(keys), 1, figsize=(12, 4*len(keys)))
    for i, k in enumerate(keys):
        sns.histplot(df_features[k], kde=True, ax=axes[i], color='#3498db', bins=50)
        axes[i].set_title(f"Distribución de {k}", fontsize=14)
        axes[i].set_xlabel("")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/03_feature_distributions.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 5. Evolución Temporal de Señales Macro/Sentimiento
    print("[*] Generando Series Temporales...")
    plt.figure(figsize=(15, 8))
    for k in ['VIX_Level', 'Maha_Dist', 'TNX_Level']:
        if k in all_feat_names:
            # Normalizar para visualización comparativa
            vals = df_features[k]
            norm_vals = (vals - vals.mean()) / (vals.std() + 1e-9)
            plt.plot(norm_vals.iloc[-500:], label=k, alpha=0.8)
    
    plt.title("Evolución Temporal de Señales Macro (Normalizadas) - Últimos 500 días", fontsize=16)
    plt.legend()
    plt.savefig(f"{plot_dir}/04_macro_signals_time_series.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 6. PCA (Análisis de Componentes Principales)
    print("[*] Generando Proyección PCA...")
    try:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        plt.figure(figsize=(10, 8))
        # Color según el retorno del SPY (Cuantiles)
        rets = df_returns.iloc[:, 0].values
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=rets, cmap='RdYlGn', alpha=0.6, s=10)
        plt.colorbar(scatter, label='Retorno Benchmark (%)')
        plt.title(f"Proyección PCA 2D del Espacio de Features\nVarianza Explicada: {pca.explained_variance_ratio_.sum()*100:.2f}%", fontsize=14)
        plt.xlabel("Componente Principal 1")
        plt.ylabel("Componente Principal 2")
        plt.savefig(f"{plot_dir}/05_pca_projection.png", dpi=150, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"[!] Error en PCA: {e}")

    print("\n" + "="*70)
    print(f" [✔] EDA Completado con éxito.")
    print(f" [✔] Visualizaciones guardadas en: {plot_dir}/")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
