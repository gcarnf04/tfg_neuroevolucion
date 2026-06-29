import yfinance as yf
import pandas as pd
import numpy as np
import warnings
import logging
import hashlib
import pickle
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Silencio absoluto de los deprecations internos de Pandas 4 y YFinance
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.filterwarnings("ignore")
logging.getLogger('yfinance').setLevel(logging.CRITICAL)

class GestorDatos:
    def __init__(self, tickers, ticker_cash, ticker_macro, fecha_inicio, fecha_fin):
        from datetime import datetime, timedelta
        self.tickers = tickers
        self.ticker_cash = ticker_cash
        self.ticker_macro = ticker_macro
        
        # Reservar automáticamente un año antes para el calentamiento de features (warm-up)
        self.fecha_inicio_solicitada = fecha_inicio
        dt = datetime.strptime(fecha_inicio, "%Y-%m-%d")
        dt_descarga = dt - timedelta(days=365)
        self.fecha_inicio = dt_descarga.strftime("%Y-%m-%d")
        
        self.fecha_fin = fecha_fin
        self.datos_precio = None
        self.features = None 
        self.retornos = None 
        
        # Índices dinámicos de entrada para la Red Neuronal
        self.idx_c = []  # Corto Plazo
        self.idx_m = []  # Medio Plazo
        self.idx_l = []  # Largo Plazo

    def _descargar_ticker(self, ticker):
        """Descarga un único ticker con reintentos."""
        for intento in range(3):
            try:
                # Usar Ticker().history() omitiendo "end" esquiva fechas futuras crasheadas
                t_obj = yf.Ticker(ticker)
                df = t_obj.history(start=self.fecha_inicio)
                
                if df.empty or 'Close' not in df.columns:
                    # Fallback robusto para índices frágiles como ^VIX
                    df = yf.download(ticker, start=self.fecha_inicio, progress=False)
                    if isinstance(df.columns, pd.MultiIndex):
                        df = df.xs('Close', axis=1, level=0)
                
                if not df.empty and 'Close' in df.columns:
                    df.index = df.index.tz_localize(None)
                    return ticker, df['Close'].astype(float).squeeze()
            except Exception:
                pass
            time.sleep(1.5)
        print(f"[!] CRÍTICO: Fallo descargando {ticker}")
        return ticker, None

    def descargar_datos(self):
        # 1. Comprobar caché en disco
        cache_key = hashlib.md5(
            f"{sorted(self.tickers)}{self.ticker_cash}{self.ticker_macro}{self.fecha_inicio}{self.fecha_fin}".encode()
        ).hexdigest()[:8]
        cache_path = f".cache/datos_{cache_key}.pkl"
        
        if os.path.exists(cache_path):
            print(f"[*] Cargando datos desde caché ({cache_path})...")
            try:
                with open(cache_path, 'rb') as f:
                    self.datos_precio = pickle.load(f)
                return
            except Exception:
                print("[!] Error leyendo caché, descargando de nuevo...")
        
        # 2. Descarga paralela
        print(f"Descargando datos de {self.tickers} y el índice de miedo VIX (Paralelo)...")
        all_tickers = self.tickers + [self.ticker_cash, self.ticker_macro, '^VIX']
        datos = {}
        
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = {executor.submit(self._descargar_ticker, t): t for t in all_tickers}
            for future in as_completed(futures):
                ticker, serie = future.result()
                if serie is not None:
                    datos[ticker] = serie
        
        if not datos:
            raise RuntimeError("CRÍTICO: No se pudo descargar ningún dato de mercado.")
            
        self.datos_precio = pd.DataFrame(datos).ffill().dropna()
        
        # 3. Guardar en caché
        os.makedirs(".cache", exist_ok=True)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(self.datos_precio, f)
            print(f"[*] Datos guardados en caché ({cache_path})")
        except Exception as e:
            print(f"[!] No se pudo guardar caché: {e}")

    def calcular_features(self):
        print("Calculando indicadores técnicos y sentimiento de mercado (Alpha v2.0)...")
        lista_features = []
        n_col = 0 # Contador dinámico de columnas
        
        # Pre-calcular el retorno del benchmark (normalmente SPY, primer ticker) para las correlaciones
        ret_spy = self.datos_precio[self.tickers[0]].pct_change(1)
        
        # 1. Features para los N activos principales
        for ticker in self.tickers:
            precio = self.datos_precio[ticker]
            df_t = pd.DataFrame(index=precio.index)
            
            # --- CORTO PLAZO (idx_c) ---
            ret_1d = precio.pct_change(1)
            df_t[f'{ticker}_Ret_D'] = ret_1d
            # Efficiency Ratio (Kaufman)
            change = (precio - precio.shift(10)).abs()
            volatility = ret_1d.abs().rolling(10).sum()
            df_t[f'{ticker}_ER_10'] = change / (volatility + 1e-9)
            
            self.idx_c.extend([n_col, n_col+1])
            n_col += 2
            
            # --- MEDIO PLAZO (idx_m) ---
            sma_50 = precio.rolling(window=50).mean()
            vol_21 = ret_1d.rolling(21).std()
            
            df_t[f'{ticker}_Vol_21'] = vol_21
            self.idx_m.append(n_col)
            n_col += 1
            
            if ticker != "QQQ": # Anti-Multicolinealidad
                df_t[f'{ticker}_Dist_50'] = (precio / sma_50) - 1
                self.idx_m.append(n_col)
                n_col += 1
                
            if ticker != self.tickers[0]: # Anti-Variable Fantasma
                df_t[f'{ticker}_Corr_SPY_21'] = ret_1d.rolling(21).corr(ret_spy)
                self.idx_m.append(n_col)
                n_col += 1
            
            # --- LARGO PLAZO (idx_l) ---
            ret_21 = precio.pct_change(21)
            df_t[f'{ticker}_Sharpe_21'] = ret_21 / (vol_21 * np.sqrt(252) + 0.02)
            max_252 = precio.rolling(window=252).max()
            df_t[f'{ticker}_DD_252'] = (precio / max_252) - 1.0 
            
            self.idx_l.extend([n_col, n_col+1])
            n_col += 2
            
            if ticker != "QQQ": # Anti-Multicolinealidad
                df_t[f'{ticker}_Mom_3M'] = precio.pct_change(63)
                sma_200 = precio.rolling(200).mean()
                df_t[f'{ticker}_MA_Cross'] = (sma_50 / (sma_200 + 1e-9)) - 1
                self.idx_l.extend([n_col, n_col+1])
                n_col += 2
            
            lista_features.append(df_t)

        # 2. Fuerza Relativa Dinámica (Todos contra el primer activo "Benchmark", ej. SPY)
        df_rel = pd.DataFrame(index=self.datos_precio.index)
        if len(self.tickers) > 1:
            benchmark = self.tickers[0]
            for i in range(1, len(self.tickers)):
                t = self.tickers[i]
                # Ratio de precios absoluto (Sin momentum del ratio para evitar ruido de 2º orden)
                df_rel[f'{t}_vs_{benchmark}'] = self.datos_precio[t] / self.datos_precio[benchmark]
                self.idx_l.append(n_col) # Estructural de Largo Plazo
                n_col += 1
            lista_features.append(df_rel)

        # 3. Features del VIX (Sentimiento de Mercado)
        vix = self.datos_precio['^VIX']
        df_vix = pd.DataFrame(index=vix.index)
        df_vix['VIX_Level'] = vix / 100 
        df_vix['VIX_Trend'] = vix.pct_change(5)
        # Volatilidad de la Volatilidad (VVIX proxy)
        df_vix['VIX_Vol'] = df_vix['VIX_Level'].pct_change(1).rolling(21).std()
        
        # NUEVA FEATURE: MACD del VIX (Aceleración del Miedo)
        vix_ema_12 = df_vix['VIX_Level'].ewm(span=12, adjust=False).mean()
        vix_ema_26 = df_vix['VIX_Level'].ewm(span=26, adjust=False).mean()
        df_vix['VIX_MACD'] = vix_ema_12 - vix_ema_26
        
        self.idx_m.extend([n_col, n_col+1, n_col+2, n_col+3])
        n_col += 4
        lista_features.append(df_vix)

        # 4. Features Macro (^TNX)
        tnx = self.datos_precio[self.ticker_macro]
        df_tnx = pd.DataFrame(index=tnx.index)
        df_tnx['TNX_Level'] = tnx / 100.0
        df_tnx['TNX_Dist_50'] = (tnx / tnx.rolling(window=50).mean()) - 1
        self.idx_l.extend([n_col, n_col+1])
        n_col += 2
        lista_features.append(df_tnx)

        # ═════════════════════════════════════════════════════════
        # ALPHA v5.0: FEATURES DE ESTRUCTURA Y RÉGIMEN DE MERCADO
        # ═════════════════════════════════════════════════════════
        df_regime = pd.DataFrame(index=self.datos_precio.index)

        # --- FAMILIA B: ESTRUCTURA DE CORRELACIONES ---
        # Cross-Correlation Media (ventana 21d) entre todos los activos del universo
        # Predictor directo de si el alpha-picking es posible (corr alta = no diversificación)
        ret_matrix = self.datos_precio[self.tickers].pct_change(1)
        
        n_tickers = len(self.tickers)
        if n_tickers > 1:
            ret_matrix_clean = ret_matrix.dropna(how='any', axis=0)  # solo filas completas para la corr
            n_assets = ret_matrix.shape[1]
            pairs = [(i, j) for i in range(n_assets) for j in range(i+1, n_assets)]
            pair_corrs = []
            for i, j in pairs:
                c = ret_matrix_clean.iloc[:, i].rolling(21).corr(ret_matrix_clean.iloc[:, j])
                pair_corrs.append(c)
            rolling_corr_mean = pd.concat(pair_corrs, axis=1).mean(axis=1)
            # Reindex to original index
            rolling_corr_mean = rolling_corr_mean.reindex(ret_matrix.index).fillna(0.0)
        else:
            rolling_corr_mean = pd.Series(0.0, index=ret_matrix.index)
            
        df_regime['Cross_Corr_21'] = rolling_corr_mean
        self.idx_m.append(n_col); n_col += 1

        # Dispersión transversal (std de retornos diarios entre activos)
        df_regime['Dispersion_XS'] = ret_matrix.std(axis=1)
        self.idx_m.append(n_col); n_col += 1

        # --- FAMILIA A: RÉGIMEN MACRO ---
        # SPY distancia a MA200 (complementa la de MA50 que ya existe)
        spy_precio = self.datos_precio[self.tickers[0]]
        sma_200_spy = spy_precio.rolling(200).mean()
        df_regime['SPY_Dist_200'] = (spy_precio / (sma_200_spy + 1e-9)) - 1
        self.idx_m.append(n_col); n_col += 1

        # RSI del SPY (14 períodos) — sobrecompra/sobreventa del benchmark
        delta = spy_precio.diff(1)
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / (loss + 1e-9)
        df_regime['SPY_RSI_14'] = (100 - (100 / (1 + rs))) / 100  # Normalizado [0,1]
        self.idx_m.append(n_col); n_col += 1

        # Ratio Vol Corta / Vol Larga del SPY (cambio de régimen en tiempo real)
        ret_spy_raw = spy_precio.pct_change(1)
        vol_5  = ret_spy_raw.rolling(5).std()
        vol_63 = ret_spy_raw.rolling(63).std()
        df_regime['SPY_Vol_Ratio'] = vol_5 / (vol_63 + 1e-9)
        self.idx_m.append(n_col); n_col += 1

        # Pendiente de la curva (TNX trend 21d vs nivel): proxy de ciclo económico
        tnx_series = self.datos_precio[self.ticker_macro]
        df_regime['TNX_Slope_21'] = tnx_series.diff(21) / 100.0  # En puntos porcentuales
        self.idx_l.append(n_col); n_col += 1

        # --- FAMILIA D: MOMENTUM RELATIVO CROSS-SECTIONAL ---
        # Rango percentil del retorno 21d de cada activo dentro del universo (0=peor, 1=mejor)
        ret_21d_matrix = self.datos_precio[self.tickers].pct_change(21)
        for ticker_rank in self.tickers:
            ranks = ret_21d_matrix.rank(axis=1, pct=True)[ticker_rank]
            df_regime[f'{ticker_rank}_MomRank_21'] = ranks
            self.idx_c.append(n_col); n_col += 1  # Corto plazo: ranking actualizable rápido

        lista_features.append(df_regime)

        # Unimos todo en una gran matriz
        self.features = pd.concat(lista_features, axis=1)
        
        # Retornos de los 4 activos principales + Asset 5 (Cash ETF)
        # La forma de los retornos será (Dias, 5)
        lista_retornos = self.tickers + [self.ticker_cash]
        self.retornos = self.datos_precio[lista_retornos].pct_change(1)

        # --- PROTECCIÓN CRÍTICA: SHIFT TEMPORAL ---
        # Ahora SÍ desplazamos TODAS las columnas (incluyendo las relativas y momentum)
        self.features = self.features.shift(1)

        self.features.dropna(inplace=True)
        # Recortar para que comience exactamente en la fecha de inicio solicitada por el usuario
        self.features = self.features.loc[self.fecha_inicio_solicitada:]
        self.retornos = self.retornos.loc[self.features.index]

    def obtener_datos_listos(self, idx_fin_train=None, escalar_global=True, sin_recorte_general=False):
        if self.features is None:
            self.descargar_datos()
            self.calcular_features()
            self._features_completas = self.features.copy()
            self._retornos_completas = self.retornos.copy()
            
        from config import CONFIG
        oos_general = CONFIG.get('oos_general', 0)
        
        if not sin_recorte_general and oos_general > 0:
            self.features = self._features_completas.iloc[:-oos_general]
            self.retornos = self._retornos_completas.iloc[:-oos_general]
        else:
            self.features = self._features_completas.copy()
            self.retornos = self._retornos_completas.copy()
        
        # --- ALPHA v3.0: FEATURES INSTITUCIONALES (BUG-4 FIX) ---
        from .features_institucionales import agregar_features_institucionales
        X, R, i_c, i_m, i_l = agregar_features_institucionales(
            self.features.values, self.retornos.values, self.idx_c, self.idx_m, self.idx_l,
            idx_fin_train=idx_fin_train,
            aplicar_escalado_global=escalar_global
        )
        return X, R, i_c, i_m, i_l