"""
DataTransformer - Veri dönüşümleri

Bu modül scaling, transformations ve differencing işlemlerini gerçekleştirir.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from scipy.stats import boxcox, yeojohnson
import warnings

warnings.filterwarnings("ignore")


class DataTransformer:
    """
    Bitcoin fiyat verilerini dönüştürür.

    Attributes:
        data (pd.DataFrame): Dönüştürülecek veri
        transformation_results (Dict): Dönüşüm sonuçları
        logger (logging.Logger): Logging objesi
    """

    def __init__(self, log_level: str = "INFO"):
        """
        DataTransformer'ı başlatır.

        Args:
            log_level (str): Logging seviyesi
        """
        self.data: Optional[pd.DataFrame] = None
        self.transformation_results: Dict[str, Any] = {}
        self.logger = self._setup_logger(log_level)

    def _setup_logger(self, log_level: str) -> logging.Logger:
        """Logger'ı kurar."""
        logger = logging.getLogger(f"{__name__}.DataTransformer")
        logger.setLevel(getattr(logging, log_level.upper()))

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def set_data(self, data: pd.DataFrame) -> None:
        """
        Dönüştürülecek veriyi ayarlar.

        Args:
            data (pd.DataFrame): Dönüştürülecek veri
        """
        self.data = data.copy()
        self.logger.info(f"Dönüşüm için veri ayarlandı: {self.data.shape}")

    def scale_data(
        self, columns: Optional[List[str]] = None, method: str = "standard"
    ) -> pd.DataFrame:
        """
        Veriyi ölçeklendirir.

        Args:
            columns (List[str], optional): Ölçeklendirilecek sütunlar
            method (str): Ölçeklendirme yöntemi ('standard', 'minmax', 'robust')

        Returns:
            pd.DataFrame: Ölçeklendirilmiş veri
        """
        try:
            if columns is None:
                columns = ["open", "high", "low", "close", "volume"]

            # Null safety kontrolü
            if self.data is None:
                return {"error": "Data is None"}

            self.logger.info(f"Veri ölçeklendirme başlatılıyor (method: {method})")

            # Veriyi hazırla
            data_subset = self.data[columns].copy()

            # Ölçeklendirici seç
            if method == "standard":
                scaler = StandardScaler()
            elif method == "minmax":
                scaler = MinMaxScaler()
            elif method == "robust":
                scaler = RobustScaler()
            else:
                raise ValueError(f"Desteklenmeyen ölçeklendirme yöntemi: {method}")

            # Ölçeklendirme yap
            scaled_data = scaler.fit_transform(data_subset)

            # Sonuçları DataFrame'e çevir
            scaled_df = pd.DataFrame(
                scaled_data,
                index=data_subset.index,
                columns=[f"{col}_scaled" for col in columns],
            )

            # Orijinal veri ile birleştir
            result_data = self.data.copy()
            for i, col in enumerate(columns):
                result_data[f"{col}_scaled"] = scaled_df[f"{col}_scaled"]

            # Sonuçları kaydet
            self.transformation_results[f"scaling_{method}"] = {
                "method": method,
                "columns": columns,
                "scaler_params": {
                    "mean": scaler.mean_.tolist() if hasattr(scaler, "mean_") else None,
                    "scale": (
                        scaler.scale_.tolist() if hasattr(scaler, "scale_") else None
                    ),
                    "min": (
                        scaler.data_min_.tolist()
                        if hasattr(scaler, "data_min_")
                        else None
                    ),
                    "max": (
                        scaler.data_max_.tolist()
                        if hasattr(scaler, "data_max_")
                        else None
                    ),
                    "center": (
                        scaler.center_.tolist() if hasattr(scaler, "center_") else None
                    ),
                    "scale_": (
                        scaler.scale_.tolist() if hasattr(scaler, "scale_") else None
                    ),
                },
            }

            self.logger.info(f"Veri ölçeklendirme tamamlandı: {method}")
            return result_data

        except Exception as e:
            self.logger.error(f"Veri ölçeklendirme hatası: {str(e)}")
            raise

    def transform_data(
        self, columns: Optional[List[str]] = None, method: str = "log"
    ) -> pd.DataFrame:
        """
        Veriyi dönüştürür.

        Args:
            columns (List[str], optional): Dönüştürülecek sütunlar
            method (str): Dönüşüm yöntemi ('log', 'boxcox', 'yeojohnson')

        Returns:
            pd.DataFrame: Dönüştürülmüş veri
        """
        try:
            if columns is None:
                columns = ["open", "high", "low", "close", "volume"]

            # Null safety kontrolü
            if self.data is None:
                return {"error": "Data is None"}

            self.logger.info(f"Veri dönüşümü başlatılıyor (method: {method})")

            result_data = self.data.copy()
            transformation_params = {}

            for col in columns:
                if col in self.data.columns:
                    # Veriyi hazırla
                    data_clean = self.data[col].dropna()

                    if len(data_clean) == 0:
                        self.logger.warning(f"{col} sütunu için veri bulunamadı")
                        continue

                    # Pozitif değer kontrolü
                    if method in ["log", "boxcox"] and (data_clean <= 0).any():
                        self.logger.warning(
                            f"{col} sütunu negatif/zero değerler içeriyor, yeojohnson kullanılacak"
                        )
                        method = "yeojohnson"

                    # Dönüşüm yap
                    if method == "log":
                        transformed_data = np.log(data_clean)
                        transformation_params[col] = {"method": "log", "lambda": None}

                    elif method == "boxcox":
                        # Box-Cox dönüşümü
                        try:
                            transformed_data, lambda_val = boxcox(data_clean)
                            transformation_params[col] = {
                                "method": "boxcox",
                                "lambda": lambda_val,
                            }
                        except Exception as e:
                            self.logger.warning(
                                f"Box-Cox dönüşümü hatası ({col}): {str(e)}, log dönüşümü kullanılacak"
                            )
                            transformed_data = np.log(data_clean)
                            transformation_params[col] = {
                                "method": "log_fallback",
                                "lambda": None,
                            }

                    elif method == "yeojohnson":
                        # Yeo-Johnson dönüşümü
                        try:
                            transformed_data, lambda_val = yeojohnson(data_clean)
                            transformation_params[col] = {
                                "method": "yeojohnson",
                                "lambda": lambda_val,
                            }
                        except Exception as e:
                            self.logger.warning(
                                f"Yeo-Johnson dönüşümü hatası ({col}): {str(e)}, log dönüşümü kullanılacak"
                            )
                            transformed_data = np.log(np.abs(data_clean) + 1)
                            transformation_params[col] = {
                                "method": "log_fallback",
                                "lambda": None,
                            }

                    else:
                        raise ValueError(f"Desteklenmeyen dönüşüm yöntemi: {method}")

                    # Sonuçları kaydet
                    result_data[f"{col}_transformed"] = transformed_data

                    # Eksik değerleri doldur
                    result_data[f"{col}_transformed"] = result_data[
                        f"{col}_transformed"
                    ].fillna(transformed_data.mean())

            # Sonuçları kaydet
            self.transformation_results[f"transformation_{method}"] = {
                "method": method,
                "columns": columns,
                "transformation_params": transformation_params,
            }

            self.logger.info(f"Veri dönüşümü tamamlandı: {method}")
            return result_data

        except Exception as e:
            self.logger.error(f"Veri dönüşümü hatası: {str(e)}")
            raise

    def difference_data(
        self, columns: Optional[List[str]] = None, order: int = 1
    ) -> pd.DataFrame:
        """
        Veriyi farklar.

        Args:
            columns (List[str], optional): Farklanacak sütunlar
            order (int): Farklama derecesi

        Returns:
            pd.DataFrame: Farklanmış veri
        """
        try:
            if columns is None:
                columns = ["open", "high", "low", "close", "volume"]

            # Null safety kontrolü
            if self.data is None:
                return {"error": "Data is None"}

            self.logger.info(f"Veri farklama başlatılıyor (order: {order})")

            result_data = self.data.copy()
            differencing_params = {}

            for col in columns:
                if col in self.data.columns:
                    # Veriyi hazırla
                    data_clean = self.data[col].dropna()

                    if len(data_clean) < order + 1:
                        self.logger.warning(f"{col} sütunu için yeterli veri yok")
                        continue

                    # Farklama yap
                    if order == 1:
                        differenced_data = data_clean.diff()
                    elif order == 2:
                        differenced_data = data_clean.diff().diff()
                    else:
                        # Yüksek dereceli farklama
                        differenced_data = data_clean
                        for i in range(order):
                            differenced_data = differenced_data.diff()

                    # Sonuçları kaydet
                    result_data[f"{col}_diff_{order}"] = differenced_data

                    # Farklama parametrelerini kaydet
                    differencing_params[col] = {
                        "order": order,
                        "original_mean": data_clean.mean(),
                        "original_std": data_clean.std(),
                        "differenced_mean": differenced_data.mean(),
                        "differenced_std": differenced_data.std(),
                    }

            # Sonuçları kaydet
            self.transformation_results[f"differencing_{order}"] = {
                "order": order,
                "columns": columns,
                "differencing_params": differencing_params,
            }

            self.logger.info(f"Veri farklama tamamlandı: order {order}")
            return result_data

        except Exception as e:
            self.logger.error(f"Veri farklama hatası: {str(e)}")
            raise

    def create_technical_indicators(
        self, columns: Optional[List[str]] = None, windows: List[int] = [5, 10, 20]
    ) -> pd.DataFrame:
        """
        Teknik göstergeler oluşturur.

        Args:
            columns (List[str], optional): Gösterge oluşturulacak sütunlar
            windows (List[int]): Pencere boyutları

        Returns:
            pd.DataFrame: Teknik göstergeler eklenmiş veri
        """
        try:
            if columns is None:
                columns = ["open", "high", "low", "close", "volume"]

            # Null safety kontrolü
            if self.data is None:
                return {"error": "Data is None"}

            self.logger.info(f"Teknik göstergeler oluşturuluyor (windows: {windows})")

            result_data = self.data.copy()
            indicator_params = {}

            for col in columns:
                if col in self.data.columns:
                    col_indicators = {}

                    for window in windows:
                        # Hareketli ortalama
                        ma_col = f"{col}_ma_{window}"
                        result_data[ma_col] = (
                            self.data[col].rolling(window=window).mean()
                        )
                        col_indicators[f"ma_{window}"] = {
                            "window": window,
                            "type": "moving_average",
                        }

                        # Hareketli standart sapma
                        std_col = f"{col}_std_{window}"
                        result_data[std_col] = (
                            self.data[col].rolling(window=window).std()
                        )
                        col_indicators[f"std_{window}"] = {
                            "window": window,
                            "type": "rolling_std",
                        }

                        # Bollinger Bands
                        if col == "close":
                            bb_upper = f"{col}_bb_upper_{window}"
                            bb_lower = f"{col}_bb_lower_{window}"
                            bb_middle = result_data[ma_col]
                            bb_std = result_data[std_col]

                            result_data[bb_upper] = bb_middle + (2 * bb_std)
                            result_data[bb_lower] = bb_middle - (2 * bb_std)

                            col_indicators[f"bb_upper_{window}"] = {
                                "window": window,
                                "type": "bollinger_upper",
                            }
                            col_indicators[f"bb_lower_{window}"] = {
                                "window": window,
                                "type": "bollinger_lower",
                            }

                    # RSI hesaplama (sadece close için)
                    if col == "close":
                        rsi_col = f"{col}_rsi_14"
                        result_data[rsi_col] = self._calculate_rsi(self.data[col], 14)
                        col_indicators["rsi_14"] = {"window": 14, "type": "rsi"}

                    # MACD hesaplama (sadece close için)
                    if col == "close":
                        macd_col = f"{col}_macd"
                        macd_signal_col = f"{col}_macd_signal"
                        macd_hist_col = f"{col}_macd_hist"

                        macd_line, signal_line, histogram = self._calculate_macd(
                            self.data[col]
                        )
                        result_data[macd_col] = macd_line
                        result_data[macd_signal_col] = signal_line
                        result_data[macd_hist_col] = histogram

                        col_indicators["macd"] = {"type": "macd_line"}
                        col_indicators["macd_signal"] = {"type": "macd_signal"}
                        col_indicators["macd_hist"] = {"type": "macd_histogram"}

                    indicator_params[col] = col_indicators

            # Sonuçları kaydet
            self.transformation_results["technical_indicators"] = {
                "columns": columns,
                "windows": windows,
                "indicator_params": indicator_params,
            }

            self.logger.info(f"Teknik göstergeler tamamlandı: {len(windows)} pencere")
            return result_data

        except Exception as e:
            self.logger.error(f"Teknik göstergeler hatası: {str(e)}")
            raise

    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """
        RSI hesaplar.

        Args:
            prices (pd.Series): Fiyat serisi
            window (int): Pencere boyutu

        Returns:
            pd.Series: RSI değerleri
        """
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except Exception as e:
            self.logger.warning(f"RSI hesaplama hatası: {str(e)}")
            return pd.Series(index=prices.index, dtype=float)

    def _calculate_macd(
        self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        MACD hesaplar.

        Args:
            prices (pd.Series): Fiyat serisi
            fast (int): Hızlı EMA periyodu
            slow (int): Yavaş EMA periyodu
            signal (int): Sinyal EMA periyodu

        Returns:
            Tuple[pd.Series, pd.Series, pd.Series]: MACD line, signal line, histogram
        """
        try:
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal).mean()
            histogram = macd_line - signal_line
            return macd_line, signal_line, histogram
        except Exception as e:
            self.logger.warning(f"MACD hesaplama hatası: {str(e)}")
            empty_series = pd.Series(index=prices.index, dtype=float)
            return empty_series, empty_series, empty_series

    def create_lag_features(
        self, columns: Optional[List[str]] = None, lags: List[int] = [1, 2, 3, 5, 10]
    ) -> pd.DataFrame:
        """
        Lag özellikleri oluşturur.

        Args:
            columns (List[str], optional): Lag oluşturulacak sütunlar
            lags (List[int]): Lag değerleri

        Returns:
            pd.DataFrame: Lag özellikleri eklenmiş veri
        """
        try:
            if columns is None:
                columns = ["open", "high", "low", "close", "volume"]

            # Null safety kontrolü
            if self.data is None:
                return {"error": "Data is None"}

            self.logger.info(f"Lag özellikleri oluşturuluyor (lags: {lags})")

            result_data = self.data.copy()
            lag_params = {}

            for col in columns:
                if col in self.data.columns:
                    col_lags = {}

                    for lag in lags:
                        lag_col = f"{col}_lag_{lag}"
                        result_data[lag_col] = self.data[col].shift(lag)
                        col_lags[f"lag_{lag}"] = {"lag": lag, "type": "lag_feature"}

                    lag_params[col] = col_lags

            # Sonuçları kaydet
            self.transformation_results["lag_features"] = {
                "columns": columns,
                "lags": lags,
                "lag_params": lag_params,
            }

            self.logger.info(f"Lag özellikleri tamamlandı: {len(lags)} lag")
            return result_data

        except Exception as e:
            self.logger.error(f"Lag özellikleri hatası: {str(e)}")
            raise

    def run_all_transformations(
        self, config: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Tüm dönüşümleri çalıştırır.

        Args:
            config (Dict[str, Any], optional): Dönüşüm konfigürasyonu

        Returns:
            pd.DataFrame: Tüm dönüşümler uygulanmış veri
        """
        try:
            # Null safety kontrolü
            if self.data is None:
                return {"error": "Data is None"}

            if config is None:
                config = {
                    "scaling": {"method": "robust"},
                    "transformation": {"method": "log"},
                    "differencing": {"order": 1},
                    "technical_indicators": {"windows": [5, 10, 20]},
                    "lag_features": {"lags": [1, 2, 3, 5, 10]},
                }

            self.logger.info("Kapsamlı veri dönüşümü başlatılıyor")

            result_data = self.data.copy()

            # Scaling
            if "scaling" in config:
                result_data = self.scale_data(method=config["scaling"]["method"])

            # Transformation
            if "transformation" in config:
                result_data = self.transform_data(
                    method=config["transformation"]["method"]
                )

            # Differencing
            if "differencing" in config:
                result_data = self.difference_data(
                    order=config["differencing"]["order"]
                )

            # Technical indicators
            if "technical_indicators" in config:
                result_data = self.create_technical_indicators(
                    windows=config["technical_indicators"]["windows"]
                )

            # Lag features
            if "lag_features" in config:
                result_data = self.create_lag_features(
                    lags=config["lag_features"]["lags"]
                )

            # Genel özet
            self.transformation_results["summary"] = {
                "total_transformations": len(self.transformation_results),
                "timestamp": pd.Timestamp.now().isoformat(),
                "config_used": config,
            }

            self.logger.info("Veri dönüşümü tamamlandı")
            return result_data

        except Exception as e:
            self.logger.error(f"Veri dönüşümü hatası: {str(e)}")
            raise

    def get_transformation_report(self) -> Dict[str, Any]:
        """
        Dönüşüm raporunu döndürür.

        Returns:
            Dict[str, Any]: Detaylı dönüşüm raporu
        """
        return {
            "timestamp": pd.Timestamp.now().isoformat(),
            "data_shape": self.data.shape if self.data is not None else None,
            "transformation_results": self.transformation_results,
            "summary": {
                "total_transformations": len(self.transformation_results),
                "new_features_created": (
                    sum(
                        len([col for col in self.data.columns if col.endswith(suffix)])
                        for suffix in [
                            "_scaled",
                            "_transformed",
                            "_diff_1",
                            "_diff_2",
                            "_ma_",
                            "_std_",
                            "_lag_",
                        ]
                    )
                    if self.data is not None
                    else 0
                ),
            },
        }
