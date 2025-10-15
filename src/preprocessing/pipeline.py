"""
DataPipeline - Ana orkestrasyon sınıfı

Bu modül tüm preprocessing sınıflarını koordine eder,
pipeline workflow yönetimi yapar ve sonuç raporlama gerçekleştirir.
"""

import pandas as pd
import numpy as np
import logging
import json
import warnings
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime
from .data_loader import DataLoader
from .validators import DataValidator
from .outlier_detector import OutlierDetector
from .statistical_analyzer import StatisticalAnalyzer
from .transformers import DataTransformer
from .cleaners import DataCleaner
from .config import PIPELINE_CONFIG

warnings.filterwarnings("ignore")


class DataPipeline:
    """
    Bitcoin fiyat verileri için kapsamlı preprocessing pipeline'ı.

    Attributes:
        input_path (str): Giriş dosya yolu
        output_path (str): Çıkış dosya yolu
        config (Dict): Pipeline konfigürasyonu
        logger (logging.Logger): Logging objesi
        results (Dict): Pipeline sonuçları
    """

    def __init__(
        self, input_path: str, output_path: str, config: Optional[Dict[str, Any]] = None
    ):
        """
        DataPipeline'ı başlatır.

        Args:
            input_path (str): Giriş CSV dosya yolu
            output_path (str): Çıkış CSV dosya yolu
            config (Dict[str, Any], optional): Pipeline konfigürasyonu
        """
        self.input_path = input_path
        self.output_path = output_path
        self.config = config or {}
        self.logger = self._setup_logger()
        self.results: Dict[str, Any] = {}

        # Pipeline bileşenlerini başlat
        self._initialize_components()

    def _setup_logger(self) -> logging.Logger:
        """Logger'ı kurar."""
        logger = logging.getLogger(f"{__name__}.DataPipeline")
        log_level = self.config.get("log_level", PIPELINE_CONFIG["log_level"])
        logger.setLevel(getattr(logging, log_level.upper()))

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _initialize_components(self) -> None:
        """Pipeline bileşenlerini başlatır."""
        try:
            log_level = self.config.get("log_level", PIPELINE_CONFIG["log_level"])

            self.data_loader = DataLoader(log_level)
            self.validator = DataValidator(log_level)
            self.outlier_detector = OutlierDetector(log_level)
            self.statistical_analyzer = StatisticalAnalyzer(log_level)
            self.transformer = DataTransformer(log_level)
            self.cleaner = DataCleaner(log_level)

            self.logger.info("Pipeline bileşenleri başlatıldı")

        except Exception as e:
            self.logger.error(f"Pipeline bileşenleri başlatma hatası: {str(e)}")
            raise

    def run(self) -> pd.DataFrame:
        """
        Pipeline'ı çalıştırır.

        Returns:
            pd.DataFrame: İşlenmiş veri
        """
        try:
            self.logger.info("Bitcoin Data Preprocessing Pipeline başlatılıyor")
            start_time = datetime.now()

            # 1. Veri yükleme ve temel validasyon
            self.logger.info("=== AŞAMA 1: Veri Yükleme ve Temel Validasyon ===")
            data = self._load_and_validate_data()

            # 2. İstatistiksel analiz
            self.logger.info("=== AŞAMA 2: İstatistiksel Analiz ===")
            self._perform_statistical_analysis(data)

            # 3. Outlier tespiti
            self.logger.info("=== AŞAMA 3: Outlier Tespiti ===")
            outlier_results = self._detect_outliers(data)

            # 4. Veri temizleme
            self.logger.info("=== AŞAMA 4: Veri Temizleme ===")
            cleaned_data = self._clean_data(data, outlier_results)

            # 5. Veri dönüşümü
            self.logger.info("=== AŞAMA 5: Veri Dönüşümü ===")
            transformed_data = self._transform_data(cleaned_data)

            # 6. Final validasyon ve kaydetme
            self.logger.info("=== AŞAMA 6: Final Validasyon ve Kaydetme ===")
            final_data = self._final_validation_and_save(transformed_data)

            # Pipeline sonuçlarını kaydet
            end_time = datetime.now()
            self.results["pipeline_summary"] = {
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_seconds": (end_time - start_time).total_seconds(),
                "input_shape": data.shape if data is not None else None,
                "output_shape": final_data.shape if final_data is not None else None,
                "status": "SUCCESS",
            }

            self.logger.info(
                f"Pipeline başarıyla tamamlandı: {self.results['pipeline_summary']['duration_seconds']:.2f} saniye"
            )
            return final_data

        except Exception as e:
            self.logger.error(f"Pipeline çalıştırma hatası: {str(e)}")
            self.results["pipeline_summary"] = {
                "status": "FAILED",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }
            raise

    def _load_and_validate_data(self) -> pd.DataFrame:
        """Veriyi yükler ve temel validasyonları gerçekleştirir."""
        try:
            # Veriyi yükle
            data = self.data_loader.load_and_validate(self.input_path)

            # Temel validasyonları çalıştır
            self.validator.set_data(data)
            validation_results = self.validator.run_all_validations()

            # Sonuçları kaydet
            self.results["data_loading"] = {
                "input_path": self.input_path,
                "data_shape": data.shape,
                "validation_results": validation_results,
            }

            return data

        except Exception as e:
            self.logger.error(f"Veri yükleme ve validasyon hatası: {str(e)}")
            raise

    def _perform_statistical_analysis(self, data: pd.DataFrame) -> None:
        """İstatistiksel analiz gerçekleştirir."""
        try:
            self.statistical_analyzer.set_data(data)
            analysis_results = self.statistical_analyzer.run_all_analyses()

            # Sonuçları kaydet
            self.results["statistical_analysis"] = analysis_results

        except Exception as e:
            self.logger.error(f"İstatistiksel analiz hatası: {str(e)}")
            raise

    def _detect_outliers(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Outlier tespiti gerçekleştirir."""
        try:
            self.outlier_detector.set_data(data)
            outlier_results = self.outlier_detector.run_all_detections()

            # Sonuçları kaydet
            self.results["outlier_detection"] = outlier_results

            return outlier_results

        except Exception as e:
            self.logger.error(f"Outlier tespiti hatası: {str(e)}")
            raise

    def _extract_outlier_indices(self, outlier_results: Dict[str, Any]) -> Dict[str, List[int]]:
        """Outlier detection sonuçlarından outlier indekslerini çıkarır."""
        try:
            outlier_indices = {}
            
            # Z-score sonuçlarından outlier indekslerini çıkar
            if "z_score" in outlier_results:
                for col, results in outlier_results["z_score"].items():
                    if "outlier_indices" in results:
                        outlier_indices[col] = results["outlier_indices"]
            
            # IQR sonuçlarından outlier indekslerini çıkar
            if "iqr" in outlier_results:
                for col, results in outlier_results["iqr"].items():
                    if "outlier_indices" in results:
                        if col not in outlier_indices:
                            outlier_indices[col] = []
                        outlier_indices[col].extend(results["outlier_indices"])
            
            # MAD sonuçlarından outlier indekslerini çıkar
            if "mad" in outlier_results:
                for col, results in outlier_results["mad"].items():
                    if "outlier_indices" in results:
                        if col not in outlier_indices:
                            outlier_indices[col] = []
                        outlier_indices[col].extend(results["outlier_indices"])
            
            # Duplikasyonları kaldır
            for col in outlier_indices:
                outlier_indices[col] = list(set(outlier_indices[col]))
            
            return outlier_indices
            
        except Exception as e:
            self.logger.error(f"Outlier indeks çıkarma hatası: {str(e)}")
            return {}

    def _clean_data(
        self, data: pd.DataFrame, outlier_results: Dict[str, Any]
    ) -> pd.DataFrame:
        """Veriyi temizler."""
        try:
            # Temizleme konfigürasyonu
            cleaning_config = self.config.get(
                "cleaning",
                {
                    "outlier_handling": {"method": "cap"},
                    "missing_value_imputation": {"method": "forward_fill"},
                    "data_smoothing": {"method": "moving_average", "window": 5},
                    "duplicate_removal": True,
                    "ohlc_consistency": True,
                },
            )

            self.cleaner.set_data(data)
            # Outlier detection sonuçlarını cleaners'a geçir
            outlier_indices = self._extract_outlier_indices(outlier_results)
            cleaned_data = self.cleaner.run_all_cleaning(cleaning_config, outlier_indices)

            # Sonuçları kaydet
            self.results["data_cleaning"] = {
                "cleaning_config": cleaning_config,
                "cleaning_results": self.cleaner.get_cleaning_report(),
            }

            return cleaned_data

        except Exception as e:
            self.logger.error(f"Veri temizleme hatası: {str(e)}")
            raise

    def _transform_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Veriyi dönüştürür."""
        try:
            # Dönüşüm konfigürasyonu
            transformation_config = self.config.get(
                "transformation",
                {
                    "scaling": {"method": "robust"},
                    "transformation": {"method": "log"},
                    "differencing": {"order": 1},
                    "technical_indicators": {"windows": [5, 10, 20]},
                    "lag_features": {"lags": [1, 2, 3, 5, 10]},
                },
            )

            self.transformer.set_data(data)
            transformed_data = self.transformer.run_all_transformations(
                transformation_config
            )

            # Sonuçları kaydet
            self.results["data_transformation"] = {
                "transformation_config": transformation_config,
                "transformation_results": self.transformer.get_transformation_report(),
            }

            return transformed_data

        except Exception as e:
            self.logger.error(f"Veri dönüşümü hatası: {str(e)}")
            raise

    def _final_validation_and_save(self, data: pd.DataFrame) -> pd.DataFrame:
        """Final validasyon ve kaydetme işlemlerini gerçekleştirir."""
        try:
            # Final validasyon
            self.validator.set_data(data)
            final_validation = self.validator.run_all_validations()

            # Çıkış dizinini oluştur
            output_path = Path(self.output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Veriyi kaydet
            data.to_csv(self.output_path, index=True)

            # Metadata'yı kaydet
            metadata_path = output_path.parent / f"{output_path.stem}_metadata.json"
            self._save_metadata(str(metadata_path))

            # Sonuçları kaydet
            self.results["final_validation"] = final_validation
            self.results["output_info"] = {
                "output_path": str(self.output_path),
                "metadata_path": str(metadata_path),
                "final_shape": data.shape,
            }

            self.logger.info(f"Veri başarıyla kaydedildi: {self.output_path}")
            return data

        except Exception as e:
            self.logger.error(f"Final validasyon ve kaydetme hatası: {str(e)}")
            raise

    def _save_metadata(self, metadata_path: str) -> None:
        """Metadata'yı JSON dosyasına kaydeder."""
        try:
            # Timestamp'ları string'e çevir
            metadata_copy = self.results.copy()
            self._convert_timestamps_to_strings(metadata_copy)

            # JSON'a kaydet
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata_copy, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Metadata kaydedildi: {metadata_path}")

        except Exception as e:
            self.logger.error(f"Metadata kaydetme hatası: {str(e)}")
            raise

    def _convert_timestamps_to_strings(self, obj: Any) -> Any:
        """Timestamp'ları string'e çevirir."""
        if isinstance(obj, dict):
            for key, value in obj.items():
                obj[key] = self._convert_timestamps_to_strings(value)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                obj[i] = self._convert_timestamps_to_strings(item)
        elif isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        elif isinstance(obj, np.datetime64):
            return pd.Timestamp(obj).isoformat()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, (np.int_, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float64, np.float32)):
            return float(obj)
        return obj

    def get_pipeline_report(self) -> Dict[str, Any]:
        """
        Pipeline raporunu döndürür.

        Returns:
            Dict[str, Any]: Detaylı pipeline raporu
        """
        return {
            "timestamp": datetime.now().isoformat(),
            "pipeline_config": self.config,
            "results": self.results,
            "summary": {
                "total_stages": len(self.results),
                "status": self.results.get("pipeline_summary", {}).get(
                    "status", "UNKNOWN"
                ),
                "input_path": self.input_path,
                "output_path": self.output_path,
            },
        }

    def save_pipeline_report(self, report_path: str) -> None:
        """
        Pipeline raporunu kaydeder.

        Args:
            report_path (str): Rapor dosya yolu
        """
        try:
            report = self.get_pipeline_report()

            # Timestamp'ları string'e çevir
            self._convert_timestamps_to_strings(report)

            # JSON'a kaydet
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Pipeline raporu kaydedildi: {report_path}")

        except Exception as e:
            self.logger.error(f"Pipeline raporu kaydetme hatası: {str(e)}")
            raise

    def validate_pipeline(self) -> bool:
        """
        Pipeline'ın doğru çalışıp çalışmadığını kontrol eder.

        Returns:
            bool: Pipeline geçerli mi
        """
        try:
            # Giriş dosyası kontrolü
            if not Path(self.input_path).exists():
                self.logger.error(f"Giriş dosyası bulunamadı: {self.input_path}")
                return False

            # Çıkış dizini kontrolü
            output_dir = Path(self.output_path).parent
            if not output_dir.exists():
                output_dir.mkdir(parents=True, exist_ok=True)

            # Konfigürasyon kontrolü
            if not isinstance(self.config, dict):
                self.logger.error("Konfigürasyon dict tipinde olmalıdır")
                return False

            self.logger.info("Pipeline validasyonu başarılı")
            return True

        except Exception as e:
            self.logger.error(f"Pipeline validasyon hatası: {str(e)}")
            return False

    def get_component_status(self) -> Dict[str, str]:
        """
        Pipeline bileşenlerinin durumunu döndürür.

        Returns:
            Dict[str, str]: Bileşen durumları
        """
        try:
            status = {}

            # Her bileşenin durumunu kontrol et
            components = {
                "data_loader": self.data_loader,
                "validator": self.validator,
                "outlier_detector": self.outlier_detector,
                "statistical_analyzer": self.statistical_analyzer,
                "transformer": self.transformer,
                "cleaner": self.cleaner,
            }

            for name, component in components.items():
                try:
                    # Bileşenin temel özelliklerini kontrol et
                    if hasattr(component, "logger"):
                        status[name] = "READY"
                    else:
                        status[name] = "ERROR"
                except Exception:
                    status[name] = "ERROR"

            return status

        except Exception as e:
            self.logger.error(f"Bileşen durumu kontrol hatası: {str(e)}")
            return {}

    def reset_pipeline(self) -> None:
        """Pipeline'ı sıfırlar."""
        try:
            self.results = {}
            self._initialize_components()
            self.logger.info("Pipeline sıfırlandı")

        except Exception as e:
            self.logger.error(f"Pipeline sıfırlama hatası: {str(e)}")
            raise
