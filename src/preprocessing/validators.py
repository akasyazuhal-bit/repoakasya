"""
DataValidator - Kapsamlı veri validasyonu

Bu modül OHLC mantıksal tutarlılık, eksik değer analizi,
duplikasyon kontrolü ve zaman serisi sürekliliği kontrollerini gerçekleştirir.
"""

import pandas as pd
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from .config import VALIDATION_CONFIG


class DataValidator:
    """
    Bitcoin fiyat verilerini kapsamlı olarak doğrular.

    Attributes:
        data (pd.DataFrame): Doğrulanacak veri
        validation_results (Dict): Validasyon sonuçları
        logger (logging.Logger): Logging objesi
    """

    def __init__(self, log_level: str = "INFO"):
        """
        DataValidator'ı başlatır.

        Args:
            log_level (str): Logging seviyesi
        """
        self.data: Optional[pd.DataFrame] = None
        self.validation_results: Dict[str, Any] = {}
        self.logger = self._setup_logger(log_level)

    def _setup_logger(self, log_level: str) -> logging.Logger:
        """Logger'ı kurar."""
        logger = logging.getLogger(f"{__name__}.DataValidator")
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
        Doğrulanacak veriyi ayarlar.

        Args:
            data (pd.DataFrame): Doğrulanacak veri
        """
        self.data = data.copy()
        self.logger.info(f"Validasyon için veri ayarlandı: {self.data.shape}")

    def validate_ohlc_consistency(self) -> Dict[str, Any]:
        """
        OHLC mantıksal tutarlılığını kontrol eder.

        Returns:
            Dict[str, Any]: OHLC validasyon sonuçları
        """
        try:
            self.logger.info("OHLC tutarlılık kontrolü başlatılıyor")

            # Null safety kontrolü
            if self.data is None:
                return {"error": "Data is None"}

            results: Dict[str, Any] = {
                "high_ge_low": True,
                "high_ge_open": True,
                "high_ge_close": True,
                "low_le_open": True,
                "low_le_close": True,
                "volume_positive": True,
                "price_positive": True,
                "violations": [],
            }

            # Temel OHLC kontrolleri
            if "high" in self.data.columns and "low" in self.data.columns:
                high_ge_low = (self.data["high"] >= self.data["low"]).all()
                results["high_ge_low"] = high_ge_low
                if not high_ge_low:
                    results["violations"].append("High < Low ihlalleri")

            if "high" in self.data.columns and "open" in self.data.columns:
                high_ge_open = (self.data["high"] >= self.data["open"]).all()
                results["high_ge_open"] = high_ge_open
                if not high_ge_open:
                    results["violations"].append("High < Open ihlalleri")

            if "high" in self.data.columns and "close" in self.data.columns:
                high_ge_close = (self.data["high"] >= self.data["close"]).all()
                results["high_ge_close"] = high_ge_close
                if not high_ge_close:
                    results["violations"].append("High < Close ihlalleri")

            if "low" in self.data.columns and "open" in self.data.columns:
                low_le_open = (self.data["low"] <= self.data["open"]).all()
                results["low_le_open"] = low_le_open
                if not low_le_open:
                    results["violations"].append("Low > Open ihlalleri")

            if "low" in self.data.columns and "close" in self.data.columns:
                low_le_close = (self.data["low"] <= self.data["close"]).all()
                results["low_le_close"] = low_le_close
                if not low_le_close:
                    results["violations"].append("Low > Close ihlalleri")

            # Volume pozitiflik kontrolü
            if "volume" in self.data.columns:
                volume_positive = (self.data["volume"] >= 0).all()
                results["volume_positive"] = volume_positive
                if not volume_positive:
                    results["violations"].append("Negatif volume değerleri")

            # Fiyat pozitiflik kontrolü
            price_columns = ["open", "high", "low", "close"]
            price_positive = True
            for col in price_columns:
                if col in self.data.columns:
                    if not (self.data[col] > 0).all():
                        price_positive = False
                        results["violations"].append(f"Negatif {col} değerleri")
                        break
            results["price_positive"] = price_positive

            # Genel tutarlılık
            results["is_consistent"] = all(
                [
                    results["high_ge_low"],
                    results["high_ge_open"],
                    results["high_ge_close"],
                    results["low_le_open"],
                    results["low_le_close"],
                    results["volume_positive"],
                    results["price_positive"],
                ]
            )

            self.validation_results["ohlc_consistency"] = results

            if results["is_consistent"]:
                self.logger.info("OHLC tutarlılık kontrolü başarılı")
            else:
                self.logger.warning(
                    f"OHLC tutarlılık ihlalleri: {results['violations']}"
                )

            return results

        except Exception as e:
            self.logger.error(f"OHLC tutarlılık kontrolü hatası: {str(e)}")
            raise

    def validate_missing_values(self) -> Dict[str, Any]:
        """
        Eksik değerleri analiz eder.

        Returns:
            Dict[str, Any]: Eksik değer analiz sonuçları
        """
        try:
            self.logger.info("Eksik değer analizi başlatılıyor")

            # Null safety kontrolü
            if self.data is None:
                return {"error": "Data is None"}

            missing_counts = self.data.isnull().sum()
            missing_percentages = (missing_counts / len(self.data)) * 100

            results = {
                "missing_counts": missing_counts.to_dict(),
                "missing_percentages": missing_percentages.to_dict(),
                "total_missing": missing_counts.sum(),
                "max_missing_pct": missing_percentages.max(),
                "is_acceptable": missing_percentages.max()
                <= VALIDATION_CONFIG["max_missing_pct"],
            }

            self.validation_results["missing_values"] = results

            if results["is_acceptable"]:
                self.logger.info("Eksik değer analizi başarılı")
            else:
                self.logger.warning(
                    f"Yüksek eksik değer oranı: {results['max_missing_pct']:.2f}%"
                )

            return results

        except Exception as e:
            self.logger.error(f"Eksik değer analizi hatası: {str(e)}")
            raise

    def validate_duplicates(self) -> Dict[str, Any]:
        """
        Duplikasyonları kontrol eder.

        Returns:
            Dict[str, Any]: Duplikasyon analiz sonuçları
        """
        try:
            self.logger.info("Duplikasyon kontrolü başlatılıyor")

            # Null safety kontrolü
            if self.data is None:
                return {"error": "Data is None"}

            # Tüm satır duplikasyonları
            total_duplicates = self.data.duplicated().sum()

            # Timestamp duplikasyonları (eğer index datetime ise)
            timestamp_duplicates = 0
            if isinstance(self.data.index, pd.DatetimeIndex):
                timestamp_duplicates = self.data.index.duplicated().sum()

            results = {
                "total_duplicates": total_duplicates,
                "timestamp_duplicates": timestamp_duplicates,
                "is_clean": total_duplicates == 0 and timestamp_duplicates == 0,
            }

            self.validation_results["duplicates"] = results

            if results["is_clean"]:
                self.logger.info("Duplikasyon kontrolü başarılı")
            else:
                self.logger.warning(
                    f"Duplikasyon tespit edildi: {total_duplicates} satır, {timestamp_duplicates} timestamp"
                )

            return results

        except Exception as e:
            self.logger.error(f"Duplikasyon kontrolü hatası: {str(e)}")
            raise

    def validate_timestamp_continuity(self) -> Dict[str, Any]:
        """
        Zaman serisi sürekliliğini kontrol eder.

        Returns:
            Dict[str, Any]: Zaman serisi süreklilik sonuçları
        """
        try:
            self.logger.info("Zaman serisi süreklilik kontrolü başlatılıyor")

            # Null safety kontrolü
            if self.data is None:
                return {"error": "Data is None"}

            if not isinstance(self.data.index, pd.DatetimeIndex):
                raise ValueError("Index datetime tipinde olmalıdır")

            # Sıralama kontrolü
            is_sorted = self.data.index.is_monotonic_increasing

            # Beklenen tarih aralığı
            start_date = self.data.index.min()
            end_date = self.data.index.max()
            expected_days = (end_date - start_date).days + 1
            actual_days = len(self.data)

            # Eksik tarihleri bul
            full_range = pd.date_range(start=start_date, end=end_date, freq="D")
            missing_dates = set(full_range) - set(self.data.index)
            extra_dates = set(self.data.index) - set(full_range)

            results = {
                "is_sorted": is_sorted,
                "expected_days": expected_days,
                "actual_days": actual_days,
                "missing_dates": len(missing_dates),
                "extra_dates": len(extra_dates),
                "missing_dates_list": sorted(list(missing_dates)),
                "is_continuous": len(missing_dates) == 0 and len(extra_dates) == 0,
            }

            self.validation_results["timestamp_continuity"] = results

            if results["is_continuous"]:
                self.logger.info("Zaman serisi süreklilik kontrolü başarılı")
            else:
                self.logger.warning(
                    f"Süreklilik ihlalleri: {len(missing_dates)} eksik, {len(extra_dates)} fazla tarih"
                )

            return results

        except Exception as e:
            self.logger.error(f"Zaman serisi süreklilik kontrolü hatası: {str(e)}")
            raise

    def validate_price_changes(self) -> Dict[str, Any]:
        """
        Fiyat değişimlerini kontrol eder.

        Returns:
            Dict[str, Any]: Fiyat değişim analiz sonuçları
        """
        try:
            self.logger.info("Fiyat değişim kontrolü başlatılıyor")

            # Null safety kontrolü
            if self.data is None:
                return {"error": "Data is None"}

            results: Dict[str, Any] = {
                "price_jumps": [],
                "flash_crashes": [],
                "extreme_changes": [],
            }

            if "close" in self.data.columns:
                # Günlük fiyat değişimleri
                price_changes = self.data["close"].pct_change()

                # Aşırı fiyat değişimleri
                extreme_threshold = VALIDATION_CONFIG["price_change_limit_pct"] / 100
                extreme_changes = price_changes[abs(price_changes) > extreme_threshold]

                # Flash crash tespiti
                flash_crash_threshold = VALIDATION_CONFIG.get(
                    "flash_crash_threshold", -0.10
                )
                flash_crashes = price_changes[price_changes < flash_crash_threshold]

                results["extreme_changes"] = extreme_changes.to_dict()
                results["flash_crashes"] = flash_crashes.to_dict()
                results["max_change"] = price_changes.max()
                results["min_change"] = price_changes.min()
                results["mean_change"] = price_changes.mean()
                results["std_change"] = price_changes.std()

            self.validation_results["price_changes"] = results

            if len(results["extreme_changes"]) == 0:
                self.logger.info("Fiyat değişim kontrolü başarılı")
            else:
                self.logger.warning(
                    f"Aşırı fiyat değişimleri tespit edildi: {len(results['extreme_changes'])} adet"
                )

            return results

        except Exception as e:
            self.logger.error(f"Fiyat değişim kontrolü hatası: {str(e)}")
            raise

    def validate_volume_anomalies(self) -> Dict[str, Any]:
        """
        Volume anomalilerini kontrol eder.

        Returns:
            Dict[str, Any]: Volume anomali analiz sonuçları
        """
        try:
            self.logger.info("Volume anomali kontrolü başlatılıyor")

            # Null safety kontrolü
            if self.data is None:
                return {"error": "Data is None"}

            results: Dict[str, Any] = {
                "volume_spikes": [],
                "volume_drops": [],
                "anomalies": [],
            }

            if "volume" in self.data.columns:
                # Volume istatistikleri
                volume_mean = self.data["volume"].mean()
                volume_std = self.data["volume"].std()
                volume_threshold = VALIDATION_CONFIG["volume_spike_threshold"]

                # Volume spike tespiti
                volume_spikes = self.data["volume"] > (
                    volume_mean + volume_threshold * volume_std
                )
                results["volume_spikes"] = volume_spikes.sum()

                # Volume drop tespiti
                volume_drops = self.data["volume"] < (
                    volume_mean - volume_threshold * volume_std
                )
                results["volume_drops"] = volume_drops.sum()

                # Genel anomaliler
                results["anomalies"] = (volume_spikes | volume_drops).sum()
                results["volume_mean"] = volume_mean
                results["volume_std"] = volume_std
                results["volume_min"] = self.data["volume"].min()
                results["volume_max"] = self.data["volume"].max()

            self.validation_results["volume_anomalies"] = results

            if results["anomalies"] == 0:
                self.logger.info("Volume anomali kontrolü başarılı")
            else:
                self.logger.warning(
                    f"Volume anomalileri tespit edildi: {results['anomalies']} adet"
                )

            return results

        except Exception as e:
            self.logger.error(f"Volume anomali kontrolü hatası: {str(e)}")
            raise

    def run_all_validations(self) -> Dict[str, Any]:
        """
        Tüm validasyonları çalıştırır.

        Returns:
            Dict[str, Any]: Tüm validasyon sonuçları
        """
        try:
            self.logger.info("Kapsamlı validasyon başlatılıyor")

            # Tüm validasyonları çalıştır
            self.validate_ohlc_consistency()
            self.validate_missing_values()
            self.validate_duplicates()
            self.validate_timestamp_continuity()
            self.validate_price_changes()
            self.validate_volume_anomalies()

            # Genel kalite skoru hesapla
            quality_score = self._calculate_quality_score()
            self.validation_results["quality_score"] = quality_score

            # Genel durum
            all_passed = all(
                [
                    self.validation_results["ohlc_consistency"]["is_consistent"],
                    self.validation_results["missing_values"]["is_acceptable"],
                    self.validation_results["duplicates"]["is_clean"],
                    self.validation_results["timestamp_continuity"]["is_continuous"],
                ]
            )

            self.validation_results["overall_status"] = "PASS" if all_passed else "FAIL"

            self.logger.info(
                f"Validasyon tamamlandı - Durum: {self.validation_results['overall_status']}"
            )
            return self.validation_results

        except Exception as e:
            self.logger.error(f"Validasyon hatası: {str(e)}")
            raise

    def _calculate_quality_score(self) -> float:
        """
        Veri kalite skorunu hesaplar.

        Returns:
            float: Kalite skoru (0-100)
        """
        try:
            score = 100.0

            # OHLC tutarlılık
            if not self.validation_results["ohlc_consistency"]["is_consistent"]:
                score -= 20

            # Eksik değerler
            missing_pct = self.validation_results["missing_values"]["max_missing_pct"]
            if missing_pct > 0:
                score -= min(missing_pct * 2, 30)

            # Duplikasyonlar
            if not self.validation_results["duplicates"]["is_clean"]:
                score -= 15

            # Zaman serisi süreklilik
            if not self.validation_results["timestamp_continuity"]["is_continuous"]:
                score -= 25

            # Volume anomalileri
            if self.validation_results["volume_anomalies"]["anomalies"] > 0:
                score -= 10

            return max(0, score)

        except Exception as e:
            self.logger.error(f"Kalite skoru hesaplama hatası: {str(e)}")
            return 0.0

    def get_validation_report(self) -> Dict[str, Any]:
        """
        Validasyon raporunu döndürür.

        Returns:
            Dict[str, Any]: Detaylı validasyon raporu
        """
        return {
            "timestamp": datetime.now().isoformat(),
            "data_shape": self.data.shape if self.data is not None else None,
            "validation_results": self.validation_results,
            "summary": {
                "total_checks": len(self.validation_results),
                "passed_checks": sum(
                    1
                    for v in self.validation_results.values()
                    if isinstance(v, dict)
                    and v.get(
                        "is_consistent",
                        v.get(
                            "is_acceptable",
                            v.get("is_clean", v.get("is_continuous", False)),
                        ),
                    )
                ),
            },
        }
