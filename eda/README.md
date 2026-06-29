# 📊 Alpha Hunter - Exploratory Data Analysis (EDA)

Este directorio contiene herramientas para entender los datos que alimentan a nuestros modelos de IA. La comprensión de las "features" es crítica para depurar por qué un modelo toma ciertas decisiones de inversión.

## 📈 Visualizaciones Generadas

### 1. `01_correlation_matrix.png`
**Qué es:** Una matriz de calor (Heatmap) que muestra la correlación de Pearson entre todas las variables de entrada y el retorno objetivo.
- **Utilidad:** Identifica colinealidad (features que dicen lo mismo) y qué variables tienen una relación lineal fuerte con los movimientos del mercado.
- **Interpretación:** Colores rojos intensos indican correlación positiva fuerte; azules intensos, negativa.

### 2. `02_target_correlation.png`
**Qué es:** Un ranking de las variables más correlacionadas con el retorno del benchmark (SPY).
- **Utilidad:** Revela qué "características" tienen más peso estadístico predictivo *a priori*. 
- **Nota:** En finanzas, estas correlaciones suelen ser bajas (< 0.1), pero incluso valores pequeños son señales valiosas para el modelo.

### 3. `03_feature_distributions.png`
**Qué es:** Histogramas y estimaciones de densidad (KDE) de variables clave como el VIX, la Distancia de Mahalanobis y el Nivel de Tipos (TNX).
- **Utilidad:** Permite detectar sesgos, "fat tails" (colas pesadas) o si la normalización robusta está funcionando correctamente.

### 4. `04_macro_signals_time_series.png`
**Qué es:** Evolución temporal de las señales macro y de sentimiento.
- **Utilidad:** Ayuda a visualizar regímenes de mercado. ¿Están las señales en máximos históricos? ¿Cómo se comportaron durante crisis pasadas?

### 5. `05_pca_projection.png`
**Qué es:** Una reducción de dimensionalidad (PCA) que proyecta las 50+ variables en solo 2 dimensiones.
- **Utilidad:** Permite ver si los datos forman "clusters" naturales (regímenes de mercado) y si los retornos positivos/negativos están separados en el espacio de características.
- **Interpretación:** Si ves grupos de puntos verdes (buenos retornos) separados de los rojos, el modelo tiene un trabajo fácil. Si están mezclados, la relación es altamente no lineal.

## 🚀 Cómo ejecutar
Simplemente corre desde la raíz:
```bash
python eda/main.py
```
Los gráficos se actualizarán automáticamente en la carpeta `eda/plots/`.
