"""
OutlierDetector - İleri seviye anomali tespiti

Bu modül Z-score, IQR, MAD, Isolation Forest, LOF ve domain-specific
yöntemlerle anomali tespiti gerçekleştirir.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from scipy import stats
from scipy.stats import median_abs_deviation

from .config import OUTLIER_CONFIG


class OutlierDetector:
    """
    Bitcoin fiyat verilerinde anomali tespiti yapar.

    Attributes:
        data (pd.DataFrame): Analiz edilecek veri
        outlier_results (Dict): Anomali tespit sonuçları
        logger (logging.Logger): Logging objesi
    """

    def __init__(self, log_level: str = "INFO"):
        """
        OutlierDetector'ı başlatır.

        Args:
            log_level (str): Logging seviyesi
        """
        self.data: Optional[pd.DataFrame] = None
        self.outlier_results: Dict[str, Any] = {}
        self.logger = self._setup_logger(log_level)

    def _setup_logger(self, log_level: str) -> logging.Logger:
        """Logger'ı kurar."""
        logger = logging.getLogger(f"{__name__}.OutlierDetector")
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
        Analiz edilecek veriyi ayarlar.

        Args:
            data (pd.DataFrame): Analiz edilecek veri
        """
        self.data = data.copy()
        self.logger.info(f"Anomali tespiti için veri ayarlandı: {self.data.shape}")

    def detect_zscore_outliers(
        self, columns: Optional[List[str]] = None, threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Z-score yöntemiyle outlier tespiti yapar.

        Args:
            columns (List[str], optional): Analiz edilecek sütunlar
            threshold (float, optional): Z-score threshold

        Returns:
            Dict[str, Any]: Z-score outlier sonuçları
        """
        try:
            if threshold is None:
                threshold = OUTLIER_CONFIG["z_score_threshold"]

            if columns is None:
                columns = ["open", "high", "low", "close", "volume"]

            # Null safety kontrolü
            if self.data is None:
                return {"error": "Data is None"}

            self.logger.info(
                f"Z-score outlier tespiti başlatılıyor (threshold: {threshold})"
            )

            results = {}

            for col in columns:
                if col in self.data.columns:
                    # Z-score hesapla
                    z_scores = np.abs(stats.zscore(self.data[col].dropna()))

                    # Outlier'ları tespit et
                    outliers = z_scores > threshold
                    outlier_count = outliers.sum()
                    outlier_pct = (outlier_count / len(self.data)) * 100

                    results[col] = {
                        "outlier_count": outlier_count,
                        "outlier_percentage": outlier_pct,
                        "outlier_indices": self.data[col].index[outliers].tolist(),
                        "z_scores": z_scores.tolist(),
                        "max_z_score": z_scores.max(),
                    }

            self.outlier_results["z_score"] = results
            total_outliers = sum(r["outlier_count"] for r in results.values())

            self.logger.info(
                f"Z-score outlier tespiti tamamlandı: {total_outliers} outlier"
            )
            return results

        except Exception as e:
            self.logger.error(f"Z-score outlier tespiti hatası: {str(e)}")
            raise

    def detect_iqr_outliers(
        self, columns: Optional[List[str]] = None, multiplier: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        IQR yöntemiyle outlier tespiti yapar.

        Args:
            columns (List[str], optional): Analiz edilecek sütunlar
            multiplier (float, optional): IQR çarpanı

        Returns:
            Dict[str, Any]: IQR outlier sonuçları
        """
        try:
            if multiplier is None:
                multiplier = OUTLIER_CONFIG["iqr_multiplier"]

            if columns is None:
                columns = ["open", "high", "low", "close", "volume"]

            # Null safety kontrolü
            if self.data is None:
                return {"error": "Data is None"}

            self.logger.info(
                f"IQR outlier tespiti başlatılıyor (multiplier: {multiplier})"
            )

            results = {}

            for col in columns:
                if col in self.data.columns:
                    # IQR hesapla
                    Q1 = self.data[col].quantile(0.25)
                    Q3 = self.data[col].quantile(0.75)
                    IQR = Q3 - Q1

                    # Outlier sınırları
                    lower_bound = Q1 - multiplier * IQR
                    upper_bound = Q3 + multiplier * IQR

                    # Outlier'ları tespit et
                    outliers = (self.data[col] < lower_bound) | (
                        self.data[col] > upper_bound
                    )
                    outlier_count = outliers.sum()
                    outlier_pct = (outlier_count / len(self.data)) * 100

                    results[col] = {
                        "outlier_count": outlier_count,
                        "outlier_percentage": outlier_pct,
                        "outlier_indices": self.data[col].index[outliers].tolist(),
                        "Q1": Q1,
                        "Q3": Q3,
                        "IQR": IQR,
                        "lower_bound": lower_bound,
                        "upper_bound": upper_bound,
                    }

            self.outlier_results["iqr"] = results
            total_outliers = sum(r["outlier_count"] for r in results.values())

            self.logger.info(
                f"IQR outlier tespiti tamamlandı: {total_outliers} outlier"
            )
            return results

        except Exception as e:
            self.logger.error(f"IQR outlier tespiti hatası: {str(e)}")
            raise

    def detect_mad_outliers(
        self, columns: Optional[List[str]] = None, threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Modified Z-score (MAD) yöntemiyle outlier tespiti yapar.

        Args:
            columns (List[str], optional): Analiz edilecek sütunlar
            threshold (float, optional): MAD threshold

        Returns:
            Dict[str, Any]: MAD outlier sonuçları
        """
        try:
            if threshold is None:
                threshold = OUTLIER_CONFIG["mad_threshold"]

            if columns is None:
                columns = ["open", "high", "low", "close", "volume"]

            # Null safety kontrolü
            if self.data is None:
                return {"error": "Data is None"}

            self.logger.info(
                f"MAD outlier tespiti başlatılıyor (threshold: {threshold})"
            )

            results = {}

            for col in columns:
                if col in self.data.columns:
                    # MAD hesapla
                    median = self.data[col].median()
                    mad = median_abs_deviation(self.data[col], scale="normal")

                    # Modified Z-score hesapla
                    modified_z_scores = 0.6745 * (self.data[col] - median) / mad

                    # Outlier'ları tespit et
                    outliers = np.abs(modified_z_scores) > threshold
                    outlier_count = outliers.sum()
                    outlier_pct = (outlier_count / len(self.data)) * 100

                    results[col] = {
                        "outlier_count": outlier_count,
                        "outlier_percentage": outlier_pct,
                        "outlier_indices": self.data[col].index[outliers].tolist(),
                        "modified_z_scores": modified_z_scores.tolist(),
                        "median": median,
                        "mad": mad,
                        "max_modified_z_score": np.abs(modified_z_scores).max(),
                    }

            self.outlier_results["mad"] = results
            total_outliers = sum(r["outlier_count"] for r in results.values())

            self.logger.info(
                f"MAD outlier tespiti tamamlandı: {total_outliers} outlier"
            )
            return results

        except Exception as e:
            self.logger.error(f"MAD outlier tespiti hatası: {str(e)}")
            raise

    def detect_isolation_forest_outliers(
        self, columns: Optional[List[str]] = None, contamination: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Isolation Forest yöntemiyle outlier tespiti yapar.

        Args:
            columns (List[str], optional): Analiz edilecek sütunlar
            contamination (float, optional): Contamination oranı

        Returns:
            Dict[str, Any]: Isolation Forest outlier sonuçları
        """
        try:
            if contamination is None:
                contamination = OUTLIER_CONFIG["isolation_forest_contamination"]

            if columns is None:
                columns = ["open", "high", "low", "close", "volume"]

            # Null safety kontrolü
            if self.data is None:
                return {"error": "Data is None"}

            self.logger.info(
                f"Isolation Forest outlier tespiti başlatılıyor (contamination: {contamination})"
            )

            # Veriyi hazırla
            data_subset = self.data[columns].dropna()

            if len(data_subset) == 0:
                self.logger.warning("Analiz edilecek veri bulunamadı")
                return {}

            # Isolation Forest modeli
            iso_forest = IsolationForest(
                contamination=contamination, random_state=42, n_estimators=100
            )

            # Outlier tespiti
            outlier_labels = iso_forest.fit_predict(data_subset)
            outlier_scores = iso_forest.decision_function(data_subset)

            # Sonuçları hazırla
            outlier_count = (outlier_labels == -1).sum()
            outlier_pct = (outlier_count / len(data_subset)) * 100

            results = {
                "outlier_count": outlier_count,
                "outlier_percentage": outlier_pct,
                "outlier_indices": data_subset.index[outlier_labels == -1].tolist(),
                "outlier_scores": outlier_scores.tolist(),
                "contamination": contamination,
                "columns_used": columns,
            }

            self.outlier_results["isolation_forest"] = results

            self.logger.info(
                f"Isolation Forest outlier tespiti tamamlandı: {outlier_count} outlier"
            )
            return results

        except Exception as e:
            self.logger.error(f"Isolation Forest outlier tespiti hatası: {str(e)}")
            raise

    def detect_lof_outliers(
        self,
        columns: Optional[List[str]] = None,
        n_neighbors: Optional[int] = None,
        contamination: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Local Outlier Factor (LOF) yöntemiyle outlier tespiti yapar.

        Args:
            columns (List[str], optional): Analiz edilecek sütunlar
            n_neighbors (int, optional): Komşu sayısı
            contamination (float, optional): Contamination oranı

        Returns:
            Dict[str, Any]: LOF outlier sonuçları
        """
        try:
            if n_neighbors is None:
                n_neighbors = int(OUTLIER_CONFIG.get("lof_neighbors", 20))
            if contamination is None:
                contamination = float(OUTLIER_CONFIG.get("lof_contamination", 0.1))

            if columns is None:
                columns = ["open", "high", "low", "close", "volume"]

            # Null safety kontrolü
            if self.data is None:
                return {"error": "Data is None"}

            self.logger.info(
                f"LOF outlier tespiti başlatılıyor (n_neighbors: {n_neighbors})"
            )

            # Veriyi hazırla
            data_subset = self.data[columns].dropna()

            if len(data_subset) == 0:
                self.logger.warning("Analiz edilecek veri bulunamadı")
                return {}

            # LOF modeli
            lof = LocalOutlierFactor(
                n_neighbors=n_neighbors, contamination=contamination
            )

            # Outlier tespiti
            outlier_labels = lof.fit_predict(data_subset)
            outlier_scores = lof.negative_outlier_factor_

            # Sonuçları hazırla
            outlier_count = (outlier_labels == -1).sum()
            outlier_pct = (outlier_count / len(data_subset)) * 100

            results = {
                "outlier_count": outlier_count,
                "outlier_percentage": outlier_pct,
                "outlier_indices": data_subset.index[outlier_labels == -1].tolist(),
                "outlier_scores": outlier_scores.tolist(),
                "n_neighbors": n_neighbors,
                "contamination": contamination,
                "columns_used": columns,
            }

            self.outlier_results["lof"] = results

            self.logger.info(f"LOF outlier tespiti tamamlandı: {outlier_count} outlier")
            return results

        except Exception as e:
            self.logger.error(f"LOF outlier tespiti hatası: {str(e)}")
            raise

    def detect_volume_spikes(self, threshold: Optional[float] = None) -> Dict[str, Any]:
        """
        Volume spike'larını tespit eder.

        Args:
            threshold (float, optional): Volume spike threshold

        Returns:
            Dict[str, Any]: Volume spike sonuçları
        """
        try:
            # Null safety kontrolü
            if self.data is None:
                return {"error": "Data is None"}

            if "volume" not in self.data.columns:
                self.logger.warning("Volume sütunu bulunamadı")
                return {}

            if threshold is None:
                threshold = OUTLIER_CONFIG["volume_spike_threshold"]

            self.logger.info(
                f"Volume spike tespiti başlatılıyor (threshold: {threshold})"
            )

            # Volume istatistikleri
            volume_mean = self.data["volume"].mean()
            volume_std = self.data["volume"].std()

            # Volume spike tespiti
            volume_spikes = self.data["volume"] > (volume_mean + threshold * volume_std)
            spike_count = volume_spikes.sum()
            spike_pct = (spike_count / len(self.data)) * 100

            results = {
                "spike_count": spike_count,
                "spike_percentage": spike_pct,
                "spike_indices": self.data["volume"].index[volume_spikes].tolist(),
                "volume_mean": volume_mean,
                "volume_std": volume_std,
                "threshold": threshold,
                "spike_volumes": self.data["volume"][volume_spikes].tolist(),
            }

            self.outlier_results["volume_spikes"] = results

            self.logger.info(f"Volume spike tespiti tamamlandı: {spike_count} spike")
            return results

        except Exception as e:
            self.logger.error(f"Volume spike tespiti hatası: {str(e)}")
            raise

    def detect_price_jumps(self, threshold: Optional[float] = None) -> Dict[str, Any]:
        """
        Fiyat sıçramalarını tespit eder.

        Args:
            threshold (float, optional): Fiyat sıçrama threshold

        Returns:
            Dict[str, Any]: Fiyat sıçrama sonuçları
        """
        try:
            # Null safety kontrolü
            if self.data is None:
                return {"error": "Data is None"}

            if "close" not in self.data.columns:
                self.logger.warning("Close sütunu bulunamadı")
                return {}

            if threshold is None:
                threshold = OUTLIER_CONFIG["price_jump_threshold"]

            self.logger.info(
                f"Fiyat sıçrama tespiti başlatılıyor (threshold: {threshold})"
            )

            # Günlük fiyat değişimleri
            price_changes = self.data["close"].pct_change()

            # Fiyat sıçramaları
            price_jumps = abs(price_changes) > threshold
            jump_count = price_jumps.sum()
            jump_pct = (jump_count / len(self.data)) * 100

            results = {
                "jump_count": jump_count,
                "jump_percentage": jump_pct,
                "jump_indices": self.data["close"].index[price_jumps].tolist(),
                "price_changes": price_changes[price_jumps].tolist(),
                "max_positive_change": price_changes.max(),
                "max_negative_change": price_changes.min(),
                "threshold": threshold,
            }

            self.outlier_results["price_jumps"] = results

            self.logger.info(f"Fiyat sıçrama tespiti tamamlandı: {jump_count} sıçrama")
            return results

        except Exception as e:
            self.logger.error(f"Fiyat sıçrama tespiti hatası: {str(e)}")
            raise

    def detect_flash_crashes(self, threshold: Optional[float] = None) -> Dict[str, Any]:
        """
        Flash crash'leri tespit eder.

        Args:
            threshold (float, optional): Flash crash threshold

        Returns:
            Dict[str, Any]: Flash crash sonuçları
        """
        try:
            # Null safety kontrolü
            if self.data is None:
                return {"error": "Data is None"}

            if "close" not in self.data.columns:
                self.logger.warning("Close sütunu bulunamadı")
                return {}

            if threshold is None:
                threshold = OUTLIER_CONFIG["flash_crash_threshold"]

            self.logger.info(
                f"Flash crash tespiti başlatılıyor (threshold: {threshold})"
            )

            # Günlük fiyat değişimleri
            price_changes = self.data["close"].pct_change()

            # Flash crash'ler
            flash_crashes = price_changes < threshold
            crash_count = flash_crashes.sum()
            crash_pct = (crash_count / len(self.data)) * 100

            results = {
                "crash_count": crash_count,
                "crash_percentage": crash_pct,
                "crash_indices": self.data["close"].index[flash_crashes].tolist(),
                "crash_changes": price_changes[flash_crashes].tolist(),
                "max_crash": price_changes.min(),
                "threshold": threshold,
            }

            self.outlier_results["flash_crashes"] = results

            self.logger.info(f"Flash crash tespiti tamamlandı: {crash_count} crash")
            return results

        except Exception as e:
            self.logger.error(f"Flash crash tespiti hatası: {str(e)}")
            raise

    def run_all_detections(self) -> Dict[str, Any]:
        """
        Tüm anomali tespit yöntemlerini çalıştırır.

        Returns:
            Dict[str, Any]: Tüm anomali tespit sonuçları
        """
        try:
            self.logger.info("Kapsamlı anomali tespiti başlatılıyor")

            # İstatistiksel yöntemler
            self.detect_zscore_outliers()
            self.detect_iqr_outliers()
            self.detect_mad_outliers()

            # Makine öğrenmesi yöntemleri
            self.detect_isolation_forest_outliers()
            self.detect_lof_outliers()

            # Domain-specific yöntemler
            self.detect_volume_spikes()
            self.detect_price_jumps()
            self.detect_flash_crashes()

            # Genel özet
            total_outliers = self._calculate_total_outliers()
            self.outlier_results["summary"] = {
                "total_outliers": total_outliers,
                "detection_methods": list(self.outlier_results.keys()),
                "timestamp": pd.Timestamp.now().isoformat(),
            }

            self.logger.info(
                f"Anomali tespiti tamamlandı: {total_outliers} toplam outlier"
            )
            return self.outlier_results

        except Exception as e:
            self.logger.error(f"Anomali tespiti hatası: {str(e)}")
            raise

    def _calculate_total_outliers(self) -> int:
        """
        Toplam outlier sayısını hesaplar.

        Returns:
            int: Toplam outlier sayısı
        """
        try:
            total = 0

            for method, results in self.outlier_results.items():
                if isinstance(results, dict):
                    if "outlier_count" in results:
                        total += results["outlier_count"]
                    elif "spike_count" in results:
                        total += results["spike_count"]
                    elif "jump_count" in results:
                        total += results["jump_count"]
                    elif "crash_count" in results:
                        total += results["crash_count"]

            return total

        except Exception as e:
            self.logger.error(f"Toplam outlier hesaplama hatası: {str(e)}")
            return 0

    def get_outlier_report(self) -> Dict[str, Any]:
        """
        Anomali tespit raporunu döndürür.

        Returns:
            Dict[str, Any]: Detaylı anomali tespit raporu
        """
        return {
            "timestamp": pd.Timestamp.now().isoformat(),
            "data_shape": self.data.shape if self.data is not None else None,
            "outlier_results": self.outlier_results,
            "summary": {
                "total_outliers": self._calculate_total_outliers(),
                "detection_methods_used": len(self.outlier_results),
                "data_quality": (
                    "Good"
                    if self._calculate_total_outliers()
                    < (len(self.data) if self.data is not None else 0) * 0.05
                    else "Needs Attention"
                ),
            },
        }
