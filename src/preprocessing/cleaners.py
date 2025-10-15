"""
DataCleaner - Veri temizleme operasyonları

Bu modül outlier handling, missing value imputation ve smoothing
işlemlerini gerçekleştirir.
"""

import pandas as pd
import numpy as np
import logging
import warnings
from typing import Dict, Any, List, Optional
from sklearn.impute import KNNImputer
from .config import CLEANING_CONFIG

warnings.filterwarnings("ignore")


class DataCleaner:
    """
    Bitcoin fiyat verilerini temizler.

    Attributes:
        data (pd.DataFrame): Temizlenecek veri
        cleaning_results (Dict): Temizleme sonuçları
        logger (logging.Logger): Logging objesi
    """

    def __init__(self, log_level: str = "INFO"):
        """
        DataCleaner'ı başlatır.

        Args:
            log_level (str): Logging seviyesi
        """
        self.data: Optional[pd.DataFrame] = None
        self.cleaning_results: Dict[str, Any] = {}
        self.logger = self._setup_logger(log_level)

    def _setup_logger(self, log_level: str) -> logging.Logger:
        """Logger'ı kurar."""
        logger = logging.getLogger(f"{__name__}.DataCleaner")
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
        Temizlenecek veriyi ayarlar.

        Args:
            data (pd.DataFrame): Temizlenecek veri
        """
        self.data = data.copy()
        self.logger.info(f"Temizleme için veri ayarlandı: {self.data.shape}")

    def handle_outliers(
        self,
        columns: Optional[List[str]] = None,
        method: str = "cap",
        outlier_indices: Optional[Dict] = None,
    ) -> pd.DataFrame:
        """
        Outlier'ları işler.

        Args:
            columns (List[str], optional): İşlenecek sütunlar
            method (str): Outlier handling yöntemi ('remove', 'cap', 'interpolate')
            outlier_indices (Dict, optional): Outlier indeksleri

        Returns:
            pd.DataFrame: Outlier'ları işlenmiş veri
        """
        try:
            if columns is None:
                columns = ["open", "high", "low", "close", "volume"]

            # Null safety kontrolü
            if self.data is None:
                return {"error": "Data is None"}

            self.logger.info(f"Outlier handling başlatılıyor (method: {method})")

            result_data = self.data.copy()
            cleaning_params = {}

            for col in columns:
                if col in self.data.columns:
                    col_params = {"method": method, "outliers_handled": 0}

                    if method == "remove":
                        # Outlier'ları kaldır
                        if outlier_indices and col in outlier_indices:
                            outlier_idx = outlier_indices[col]
                            result_data = result_data.drop(index=outlier_idx)
                            col_params["outliers_handled"] = len(outlier_idx)
                        else:
                            # IQR yöntemiyle outlier tespiti
                            Q1 = result_data[col].quantile(0.25)
                            Q3 = result_data[col].quantile(0.75)
                            IQR = Q3 - Q1
                            lower_bound = Q1 - 1.5 * IQR
                            upper_bound = Q3 + 1.5 * IQR

                            outlier_mask = (result_data[col] < lower_bound) | (
                                result_data[col] > upper_bound
                            )
                            outlier_count = outlier_mask.sum()
                            result_data = result_data[~outlier_mask]
                            col_params["outliers_handled"] = outlier_count

                    elif method == "cap":
                        # Outlier'ları sınırla
                        if outlier_indices and col in outlier_indices:
                            outlier_idx = outlier_indices[col]
                            # Percentile tabanlı sınırlama
                            lower_bound = result_data[col].quantile(0.05)
                            upper_bound = result_data[col].quantile(0.95)

                            result_data.loc[outlier_idx, col] = result_data.loc[
                                outlier_idx, col
                            ].clip(lower=lower_bound, upper=upper_bound)
                            col_params["outliers_handled"] = len(outlier_idx)
                        else:
                            # IQR tabanlı sınırlama
                            Q1 = result_data[col].quantile(0.25)
                            Q3 = result_data[col].quantile(0.75)
                            IQR = Q3 - Q1
                            lower_bound = Q1 - 1.5 * IQR
                            upper_bound = Q3 + 1.5 * IQR

                            result_data[col] = result_data[col].clip(
                                lower=lower_bound, upper=upper_bound
                            )
                            col_params["outliers_handled"] = (
                                (result_data[col] < lower_bound)
                                | (result_data[col] > upper_bound)
                            ).sum()

                    elif method == "interpolate":
                        # Outlier'ları interpolate et
                        if outlier_indices and col in outlier_indices:
                            outlier_idx = outlier_indices[col]
                            # Linear interpolation
                            result_data.loc[outlier_idx, col] = np.nan
                            result_data[col] = result_data[col].interpolate(
                                method="linear"
                            )
                            col_params["outliers_handled"] = len(outlier_idx)
                        else:
                            # IQR tabanlı interpolation
                            Q1 = result_data[col].quantile(0.25)
                            Q3 = result_data[col].quantile(0.75)
                            IQR = Q3 - Q1
                            lower_bound = Q1 - 1.5 * IQR
                            upper_bound = Q3 + 1.5 * IQR

                            outlier_mask = (result_data[col] < lower_bound) | (
                                result_data[col] > upper_bound
                            )
                            outlier_count = outlier_mask.sum()
                            result_data.loc[outlier_mask, col] = np.nan
                            result_data[col] = result_data[col].interpolate(
                                method="linear"
                            )
                            col_params["outliers_handled"] = outlier_count

                    else:
                        raise ValueError(
                            f"Desteklenmeyen outlier handling yöntemi: {method}"
                        )

                    cleaning_params[col] = col_params

            # Sonuçları kaydet
            self.cleaning_results["outlier_handling"] = {
                "method": method,
                "columns": columns,
                "cleaning_params": cleaning_params,
            }

            total_outliers = sum(int(p.get("outliers_handled", 0)) for p in cleaning_params.values())  # type: ignore
            self.logger.info(
                f"Outlier handling tamamlandı: {total_outliers} outlier işlendi"
            )
            return result_data

        except Exception as e:
            self.logger.error(f"Outlier handling hatası: {str(e)}")
            raise

    def impute_missing_values(
        self, columns: Optional[List[str]] = None, method: str = "forward_fill"
    ) -> pd.DataFrame:
        """
        Eksik değerleri doldurur.

        Args:
            columns (List[str], optional): Doldurulacak sütunlar
            method (str): Imputation yöntemi

        Returns:
            pd.DataFrame: Eksik değerleri doldurulmuş veri
        """
        try:
            if columns is None:
                columns = ["open", "high", "low", "close", "volume"]

            # Null safety kontrolü
            if self.data is None:
                return {"error": "Data is None"}

            self.logger.info(
                f"Missing value imputation başlatılıyor (method: {method})"
            )

            result_data = self.data.copy()
            imputation_params = {}

            for col in columns:
                if col in self.data.columns:
                    col_params = {"method": method, "missing_values_filled": 0}

                    # Eksik değer sayısını hesapla
                    missing_count = result_data[col].isnull().sum()
                    col_params["missing_values_filled"] = missing_count

                    if missing_count > 0:
                        if method == "forward_fill":
                            result_data[col] = result_data[col].fillna(method="ffill")

                        elif method == "backward_fill":
                            result_data[col] = result_data[col].fillna(method="bfill")

                        elif method == "interpolate":
                            result_data[col] = result_data[col].interpolate(
                                method="linear"
                            )

                        elif method == "mean":
                            mean_value = result_data[col].mean()
                            result_data[col] = result_data[col].fillna(mean_value)

                        elif method == "median":
                            median_value = result_data[col].median()
                            result_data[col] = result_data[col].fillna(median_value)

                        elif method == "knn":
                            # KNN imputation
                            knn_imputer = KNNImputer(n_neighbors=5)
                            result_data[col] = knn_imputer.fit_transform(
                                result_data[[col]]
                            ).flatten()

                        else:
                            raise ValueError(
                                f"Desteklenmeyen imputation yöntemi: {method}"
                            )

                    imputation_params[col] = col_params

            # Sonuçları kaydet
            self.cleaning_results["missing_value_imputation"] = {
                "method": method,
                "columns": columns,
                "imputation_params": imputation_params,
            }

            total_missing = sum(int(p.get("missing_values_filled", 0)) for p in imputation_params.values())  # type: ignore
            self.logger.info(
                f"Missing value imputation tamamlandı: {total_missing} değer dolduruldu"
            )
            return result_data

        except Exception as e:
            self.logger.error(f"Missing value imputation hatası: {str(e)}")
            raise

    def smooth_data(
        self,
        columns: Optional[List[str]] = None,
        method: str = "moving_average",
        window: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Veriyi yumuşatır.

        Args:
            columns (List[str], optional): Yumuşatılacak sütunlar
            method (str): Smoothing yöntemi
            window (int, optional): Pencere boyutu

        Returns:
            pd.DataFrame: Yumuşatılmış veri
        """
        try:
            if columns is None:
                columns = ["open", "high", "low", "close", "volume"]

            if window is None:
                window = int(CLEANING_CONFIG.get("moving_average_window", 5))  # type: ignore

            self.logger.info(
                f"Data smoothing başlatılıyor (method: {method}, window: {window})"
            )

            # Null safety kontrolü ekstra
            if self.data is None:
                return {"error": "Data is None"}  # type: ignore

            result_data = self.data.copy()
            smoothing_params = {}

            for col in columns:
                if col in self.data.columns:
                    col_params = {"method": method, "window": window}

                    if method == "moving_average":
                        # Hareketli ortalama
                        smoothed_col = f"{col}_smoothed"
                        result_data[smoothed_col] = (
                            result_data[col].rolling(window=window).mean()
                        )
                        col_params["smoothed_column"] = smoothed_col

                    elif method == "exponential":
                        # Exponential smoothing
                        alpha = float(CLEANING_CONFIG.get("exponential_alpha", 0.3))  # type: ignore
                        smoothed_col = f"{col}_smoothed"
                        result_data[smoothed_col] = (
                            result_data[col].ewm(alpha=alpha).mean()
                        )
                        col_params["alpha"] = alpha
                        col_params["smoothed_column"] = smoothed_col

                    elif method == "gaussian":
                        # Gaussian smoothing
                        smoothed_col = f"{col}_smoothed"
                        result_data[smoothed_col] = (
                            result_data[col]
                            .rolling(window=window)
                            .apply(lambda x: np.mean(x), raw=True)
                        )
                        col_params["smoothed_column"] = smoothed_col

                    else:
                        raise ValueError(f"Desteklenmeyen smoothing yöntemi: {method}")

                    smoothing_params[col] = col_params

            # Sonuçları kaydet
            self.cleaning_results["data_smoothing"] = {
                "method": method,
                "window": window,
                "columns": columns,
                "smoothing_params": smoothing_params,
            }

            self.logger.info(f"Data smoothing tamamlandı: {method}")
            return result_data

        except Exception as e:
            self.logger.error(f"Data smoothing hatası: {str(e)}")
            raise

    def remove_duplicates(self) -> pd.DataFrame:
        """
        Duplikasyonları kaldırır.

        Returns:
            pd.DataFrame: Duplikasyonları kaldırılmış veri
        """
        try:
            # Null safety kontrolü
            if self.data is None:
                return {"error": "Data is None"}

            self.logger.info("Duplicate removal başlatılıyor")

            # Duplikasyon sayısını hesapla
            duplicate_count = self.data.duplicated().sum()

            # Duplikasyonları kaldır
            result_data = self.data.drop_duplicates()

            # Sonuçları kaydet
            self.cleaning_results["duplicate_removal"] = {
                "duplicates_removed": duplicate_count,
                "original_shape": self.data.shape,
                "cleaned_shape": result_data.shape,
            }

            self.logger.info(
                f"Duplicate removal tamamlandı: {duplicate_count} duplikasyon kaldırıldı"
            )
            return result_data

        except Exception as e:
            self.logger.error(f"Duplicate removal hatası: {str(e)}")
            raise

    def clean_ohlc_consistency(self) -> pd.DataFrame:
        """
        OHLC tutarlılığını düzeltir.

        Returns:
            pd.DataFrame: OHLC tutarlılığı düzeltilmiş veri
        """
        try:
            # Null safety kontrolü
            if self.data is None:
                return {"error": "Data is None"}

            self.logger.info("OHLC consistency cleaning başlatılıyor")

            result_data = self.data.copy()
            consistency_fixes = {}

            # High >= Low kontrolü
            if "high" in result_data.columns and "low" in result_data.columns:
                high_low_violations = result_data["high"] < result_data["low"]
                if high_low_violations.any():
                    # High ve low'u değiştir
                    result_data.loc[high_low_violations, ["high", "low"]] = (
                        result_data.loc[high_low_violations, ["low", "high"]].values
                    )
                    consistency_fixes["high_low_swaps"] = high_low_violations.sum()

            # High >= Open kontrolü
            if "high" in result_data.columns and "open" in result_data.columns:
                high_open_violations = result_data["high"] < result_data["open"]
                if high_open_violations.any():
                    result_data.loc[high_open_violations, "high"] = result_data.loc[
                        high_open_violations, "open"
                    ]
                    consistency_fixes["high_open_fixes"] = high_open_violations.sum()

            # High >= Close kontrolü
            if "high" in result_data.columns and "close" in result_data.columns:
                high_close_violations = result_data["high"] < result_data["close"]
                if high_close_violations.any():
                    result_data.loc[high_close_violations, "high"] = result_data.loc[
                        high_close_violations, "close"
                    ]
                    consistency_fixes["high_close_fixes"] = high_close_violations.sum()

            # Low <= Open kontrolü
            if "low" in result_data.columns and "open" in result_data.columns:
                low_open_violations = result_data["low"] > result_data["open"]
                if low_open_violations.any():
                    result_data.loc[low_open_violations, "low"] = result_data.loc[
                        low_open_violations, "open"
                    ]
                    consistency_fixes["low_open_fixes"] = low_open_violations.sum()

            # Low <= Close kontrolü
            if "low" in result_data.columns and "close" in result_data.columns:
                low_close_violations = result_data["low"] > result_data["close"]
                if low_close_violations.any():
                    result_data.loc[low_close_violations, "low"] = result_data.loc[
                        low_close_violations, "close"
                    ]
                    consistency_fixes["low_close_fixes"] = low_close_violations.sum()

            # Sonuçları kaydet
            self.cleaning_results["ohlc_consistency"] = {
                "consistency_fixes": consistency_fixes,
                "total_fixes": sum(consistency_fixes.values()),
            }

            total_fixes = sum(consistency_fixes.values())
            self.logger.info(
                f"OHLC consistency cleaning tamamlandı: {total_fixes} düzeltme"
            )
            return result_data

        except Exception as e:
            self.logger.error(f"OHLC consistency cleaning hatası: {str(e)}")
            raise

    def run_all_cleaning(self, config: Optional[Dict[str, Any]] = None, outlier_indices: Optional[Dict[str, List[int]]] = None) -> pd.DataFrame:
        """
        Tüm temizleme işlemlerini çalıştırır.

        Args:
            config (Dict[str, Any], optional): Temizleme konfigürasyonu
            outlier_indices (Dict[str, List[int]], optional): Outlier indeksleri

        Returns:
            pd.DataFrame: Tüm temizleme işlemleri uygulanmış veri
        """
        try:
            # Null safety kontrolü
            if self.data is None:
                return {"error": "Data is None"}

            if config is None:
                config = {
                    "outlier_handling": {"method": "cap"},
                    "missing_value_imputation": {"method": "forward_fill"},
                    "data_smoothing": {"method": "moving_average", "window": 5},
                    "duplicate_removal": True,
                    "ohlc_consistency": True,
                }

            self.logger.info("Kapsamlı veri temizleme başlatılıyor")

            result_data = self.data.copy()

            # Duplikasyonları kaldır
            if config.get("duplicate_removal", True):
                result_data = self.remove_duplicates()

            # OHLC tutarlılığını düzelt
            if config.get("ohlc_consistency", True):
                result_data = self.clean_ohlc_consistency()

            # Outlier'ları işle
            if "outlier_handling" in config:
                result_data = self.handle_outliers(
                    method=config["outlier_handling"]["method"],
                    outlier_indices=outlier_indices
                )

            # Eksik değerleri doldur
            if "missing_value_imputation" in config:
                result_data = self.impute_missing_values(
                    method=config["missing_value_imputation"]["method"]
                )

            # Veriyi yumuşat
            if "data_smoothing" in config:
                result_data = self.smooth_data(
                    method=config["data_smoothing"]["method"],
                    window=config["data_smoothing"].get("window", 5),
                )

            # Genel özet
            self.cleaning_results["summary"] = {
                "total_cleaning_operations": len(self.cleaning_results),
                "timestamp": pd.Timestamp.now().isoformat(),
                "config_used": config,
                "final_shape": result_data.shape,
            }

            self.logger.info("Veri temizleme tamamlandı")
            return result_data

        except Exception as e:
            self.logger.error(f"Veri temizleme hatası: {str(e)}")
            raise

    def get_cleaning_report(self) -> Dict[str, Any]:
        """
        Temizleme raporunu döndürür.

        Returns:
            Dict[str, Any]: Detaylı temizleme raporu
        """
        return {
            "timestamp": pd.Timestamp.now().isoformat(),
            "data_shape": self.data.shape if self.data is not None else None,
            "cleaning_results": self.cleaning_results,
            "summary": {
                "total_cleaning_operations": len(self.cleaning_results),
                "data_quality_improvement": self._assess_quality_improvement(),
            },
        }

    def _assess_quality_improvement(self) -> str:
        """
        Veri kalitesi iyileştirmesini değerlendirir.

        Returns:
            str: Kalite iyileştirmesi değerlendirmesi
        """
        try:
            if not self.cleaning_results:
                return "No cleaning performed"

            # Temizleme işlemlerini say
            total_operations = len(self.cleaning_results)
            successful_operations = sum(
                1
                for result in self.cleaning_results.values()
                if isinstance(result, dict)
                and "total_fixes" in result
                or "outliers_handled" in result
            )

            if total_operations == 0:
                return "No operations performed"
            elif successful_operations / total_operations >= 0.8:
                return "Excellent improvement"
            elif successful_operations / total_operations >= 0.6:
                return "Good improvement"
            elif successful_operations / total_operations >= 0.4:
                return "Moderate improvement"
            else:
                return "Limited improvement"

        except Exception as e:
            self.logger.error(f"Kalite iyileştirmesi değerlendirme hatası: {str(e)}")
            return "Unknown"
