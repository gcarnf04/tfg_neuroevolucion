# Guía de Hiperparámetros de Alpha Hunter

Este documento detalla exhaustivamente todos los parámetros que componen el archivo `config.json` del pipeline de Alpha Hunter. El archivo está estructurado en dos bloques principales: `PARAMETROS_ESTATICOS` (variables estructurales que dictan la mecánica base de la simulación) y `PARAMETROS_OPTIMIZABLES` (variables que mutan mediante la optimización de Optuna/Algoritmos Genéticos a lo largo de las distintas Fases).

---

## 1. PARÁMETROS ESTÁTICOS (`PARAMETROS_ESTATICOS`)

### 1.1 Universo de Activos
* **`tickers`**: `[list]`
  Lista de símbolos o ETFs en los que la red neuronal puede invertir. El algoritmo ajustará los pesos de estos activos dinámicamente según sus predicciones.
* **`ticker_cash`**: `string`
  El ETF que se usa como valor refugio libre de riesgo (ej. `SHV` para bonos del tesoro a corto plazo) cuando el algoritmo detecta un alto riesgo de caída en la bolsa.
* **`ticker_macro`**: `string`
  El indicador macroeconómico que actúa como termómetro de estrés externo (ej. `^TNX` para el rendimiento de los bonos a 10 años).

### 1.2 Límites Temporales y Fechas
* **`fecha_inicio`**: `string`
  Fecha de inicio histórico desde la que se descargan los datos para la validación y el entrenamiento.
* **`fecha_fin_entrenamiento`**: `string` (o `null`)
  Fecha de cierre del dataset. Si se deja en `null`, se toma la fecha actual del día de ejecución por defecto.

### 1.3 Fricción y Costes de Trading
* **`k_slippage`**: `float`
  Multiplicador de *slippage cuadrático*. Modela el "Market Impact": la dificultad o el encarecimiento de cambiar grandes volúmenes de cartera al mismo tiempo. Penaliza los cambios bruscos masivos en los pesos.
* **`base_spread`**: `float`
  Spread u horquilla base que siempre se pierde al realizar un rebalanceo por la diferencia de precios Bid/Ask del bróker.

### 1.4 Aceleración de Hardware
* **`cpus`**: `int` (o `null`)
  Límite de hilos a usar para paralelización. Si se establece en `null`, usa automáticamente el total de núcleos del procesador menos 1.

### 1.5 Arquitectura de Entrenamiento Base y Control de Tiempo (GA Core)
* **`batch_size`**: `int`
  Cantidad de genomas procesados a la vez en las matrices del tensor. Ayuda a aprovechar mejor la memoria gráfica/VRAM.
* **`ciclos_decay`**: `int`
  Número de iteraciones del learning rate decay o "enfriamiento genético" si aplica (normalmente afecta a la mutación en algoritmos continuos).
* **`decay_rate`**: `float`
  Velocidad a la que decaen las tasas de aprendizaje o mutación en simulaciones largas.
* **`paciencia`**: `int`
  Número de generaciones seguidas sin mejora en el algoritmo genético para declarar un "Early Stopping" y detener el entrenamiento prematuramente, ahorrando horas de simulación innecesaria.
* **`min_delta`**: `float`
  El aumento de fitness mínimo que se considera una mejora real en el contador de `paciencia`.

### 1.6 Exploración del Motor Genético (Mining)
* **`n_warm_seed`**: `int`
  Número de "Semillas Fundadoras" o ADNs preentrenados que se necesitan encontrar antes de permitir que la evolución genética general comience.
* **`generaciones_mining`**: `int`
  Tolerancia (en número de generaciones de búsqueda) antes de aplicar el *Decay Adaptativo* y rebajar el nivel de exigencia cuando el minado no encuentra candidatos viables.
* **`poblacion_mining`**: `int`
  Tamaño de la micropoblación exploratoria usada para generar rápidamente mutaciones de los candidatos durante la búsqueda K-Fold.
* **`k_folds`**: `int`
  Número de "partes" en las que se divide la ventana de entrenamiento temporal. Cada semilla generada debe pasar de forma independiente las métricas en cada trozo para no ser rechazada.
* **`cv_umbral`**: `float`
  Umbral estricto para el K-Fold (Cross-Validation Threshold). Es el *fitness* mínimo innegociable que debe superar la red en CADA fold aislado para que se permita su vida.
* **`timeout_minado`**: `int`
  Segundos máximos permitidos para encontrar semillas en el minado inicial antes de lanzar una excepción (o antes de rebajar drásticamente el umbral si está configurado en fallback).

### 1.7 Segmentación de Walk-Forward
* **`train_size`**: `int`
  Días del bloque rodante de entrenamiento base In-Sample (IS). Calibrado óptimamente en `315` días hábiles de mercado.
* **`step_size`**: `int`
  Días de avance (paso) que da la ventana temporal en cada ciclo de la simulación OOS. Calibrado óptimamente en `64` días de mercado.
* **`reservado_oos`**: `int`
  Días de seguridad que se dejan ciegos por la derecha en el Walk Forward para probar simulaciones Out Of Sample a nivel global.
* **`reservado_validacion_f1`**: `int`
  Tamaño especial de días que se aísla de Fase 1 para realizar la validación de correlación cruzada de hiperparámetros de fitness.

### 1.8 Modos y Entornos
* **`modelo_produccion`**: `bool`
  Si está en `true`, Alpha Hunter asume que esto es ejecución Realtime y no reservará ningún OOS. Entrenará con absolutamente toda la historia disponible y escupirá los pesos para operar mañana.

---

## 2. PARÁMETROS OPTIMIZABLES (`PARAMETROS_OPTIMIZABLES`)

Estos valores pueden venir dados, pero serán sobreescritos y optimizados en las distintas fases del pipeline (por ejemplo Fase 1 ajustará las `w_` y la Fase 2 ajustará los regímenes).

### 2.1 Pesos de la Función de Fitness Híbrida (Fase 1)
Determinan qué es lo que "premia" la IA a la hora de evolucionar. **La suma total de los pesos se normaliza automáticamente.** Optuna busca la combinación óptima de estos pesos que maximiza la correlación Spearman IS↔OOS.
* **`w_ret`**: `float`
  Importancia de la rentabilidad y retornos acumulados netos del modelo.
* **`w_mdd`**: `float`
  Importancia de evitar o mitigar el "Maximum Drawdown" (caída desde pico a valle).
* **`w_sharpe_dif`**: `float`
  Premia la diferencia de Sharpe Ratio del modelo frente al benchmark (S&P 500).
* **`w_sortino`**: `float`
  Premia la rentabilidad ajustada al riesgo del downside (desviación estándar negativa).
* **`w_linealidad`**: `float`
  Premia la regularidad y linealidad en el crecimiento de la curva de valor liquidativo.
* **`w_l2`**: `float`
  Coeficiente de penalización por complejidad L2 de la red neuronal para prevenir el sobreajuste.
* **`w_turnover`**: `float`
  Peso de penalización por rotación de cartera (turnover) excesiva, controlando los costes de transacción.

### 2.2 Hiperparámetros de Trading (Fase 2)
* **`umbral_rebalanceo`**: `float`
  El umbral estático de tolerancia (Turnover Threshold). El bot no ejecutará órdenes de rebalanceo a no ser que la discrepancia de pesos exceda este porcentaje (calibrado en `0.09` o `9\%`), protegiendo la cartera de comisiones y bid-ask spreads redundantes.
* **`kelly_fraction`**: `float`
  La fracción sobre el Criterio de Kelly (Apalancamiento estadístico). Dicta cómo de agresiva debe ser la escala de las posiciones ganadoras (1.0 = Full Kelly, 0.5 = Half Kelly).

### 2.3 Regímenes de Supervivencia y Riesgo (Fase 2)
* **`umbral_miedo_macro`**: `float`
  El nivel crítico de ruido en los activos externos y bonos en los que la red detecta recesión o pánico e infiere que debe abortar posiciones y rotar al activo de Cash protector.

### 2.4 Control Genético (Fase 2)
* **`generaciones`**: `int`
  Número de épocas evolutivas del motor genético principal durante el ciclo de entrenamiento regular.
* **`poblacion`**: `int`
  Cantidad de genomas concurrentes simulando carteras cada generación.
* **`tasa_mutacion`**: `float`
  Probabilidad de que un gen sufra una alteración brusca durante el cruce generacional.
* **`fuerza_mutacion`**: `float`
  La varianza (magnitud) que puede llegar a tener un gen mutado comparado con su estado original.
* **`ratio_inmigrantes`**: `float`
  Porcentaje de la población que se inyecta con ADN totalmente fresco y aleatorio cada generación, impidiendo que el ecosistema entero se estanque por incesto.

### 2.5 Arquitectura (Fase 2)
* **`ocultas`**: `[list]`
  Lista de números enteros. Representa la arquitectura profunda neuronal en capas. Por ejemplo, `[48, 24, 10]` significa 3 capas ocultas, reduciendo su densidad paulatinamente.

### 2.6 Comité / Ensemble (Fase 3)
* **`n_mejores`**: `int`
  El tamaño de la asamblea final. Número de las redes neuronales más fuertes y probadas que compondrán el comité de votación final en OOS. Las votaciones mitigan el riesgo de sobreajuste por "suerte".

---

## 3. PARÁMETROS ESTÁTICOS — INFERENCIA Y RIESGO

Estos parámetros controlan el comportamiento del modelo en inferencia OOS. Son fijos por defecto pero pueden sobreescribirse por experimento vía `master_runner.py`.

* **`conviccion_minima`**: `float` *(Nuevo — Alpha v5.3, default: 0.0 = desactivado)*
  Umbral de **entropía normalizada** de la predicción del modelo. Cuando la predicción es muy difusa (el modelo no tiene preferencia clara entre activos), la cartera se mezcla progresivamente con una distribución refugio de 100% SPY. A mayor valor (ej. `0.70`), más agresivo es el filtro y más rápido se rota a SPY ante incertidumbre. Útil para evitar pérdidas en mercados alcistas donde el modelo no tiene ventaja comprobada.

* **`usar_riesgo_complejo`**: `bool` *(default: false)*
  Activa el pipeline completo de gestión de riesgo OOS: Kelly Criterion y Miedo Híbrido (idiosincrásico + macro + régimen). Con `N=1` modelo en el comité estos mecanismos no añaden valor (no hay desacuerdo entre modelos), pero con comités de `N>=3` pueden mejorar la robustez de las posiciones.

* **`step_size`**: `int` *(configurable por experimento)*
  Días de avance de la ventana OOS. El valor base es `126` (semestral), pero puede reducirse a `63` (trimestral) por hipótesis para duplicar el número de observaciones OOS y mejorar la potencia estadística del P-Value (efecto ~√2).
---

## 4. SECCIONES DE PERSISTENCIA (Fase 1, Fase 2, Fase 3)
Estas secciones (`FASE_1`, `FASE_2` y `FASE_3`) las gestiona automáticamente el pipeline cuando guardan un estado usando la función `guardar_pipeline_state`. Tienen un fin estrictamente documental para que las distintas fases compartan los progresos o recuperen el punto si se corta la ejecución. **`FASE_1` persiste los 7 pesos de fitness** (`w_ret`, `w_mdd`, `w_cobardia`, `w_dominancia`, `w_decorrel`, `w_sharpe_dif`, `w_oportunidad`). No se recomienda editarlas manualmente a no ser que se busque un reinicio forzado.
