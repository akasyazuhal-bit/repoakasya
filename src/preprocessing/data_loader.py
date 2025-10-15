"""
DataLoader - Veri yükleme ve başlangıç validasyonu

Bu modül CSV dosyalarını yükler, temel tip kontrolü yapar,
timestamp parsing işlemlerini gerçekleştirir ve veri şemasını doğrular.
"""

import pandas as pd
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path

from .config import PATH_CONFIG, ERROR_MESSAGES, SUCCESS_MESSAGES


class DataLoader:
    """
    Bitcoin fiyat verilerini yükler ve temel validasyonları gerçekleştirir.

    Attributes:
        data (pd.DataFrame): Yüklenen veri
        metadata (Dict): Veri hakkında metadata
        logger (logging.Logger): Logging objesi
    """

    def __init__(
        self,
        log_level: str = "INFO",
        data_reader=None,
        timestamp_parser=None,
        schema_validator=None,
        metadata_manager=None,
        config_provider=None,
        logger=None,
    ):
        """
        DataLoader'ı başlatır.

        Args:
            log_level (str): Logging seviyesi
            data_reader: Veri okuyucu
            timestamp_parser: Timestamp parser
            schema_validator: Şema validator
            metadata_manager: Metadata yöneticisi
            config_provider: Konfigürasyon sağlayıcı
            logger: Logger
        """
        self.data: Optional[pd.DataFrame] = None
        self.metadata: Dict[str, Any] = {}

        # Logger'ı ayarla
        if logger is not None:
            self.logger = logger
        else:
            self.logger = self._setup_logger(log_level)

        # Diğer bileşenleri sakla
        self.data_reader = data_reader
        self.timestamp_parser = timestamp_parser
        self.schema_validator = schema_validator
        self.metadata_manager = metadata_manager
        self.config_provider = config_provider

    def _setup_logger(self, log_level: str) -> logging.Logger:
        """Logger'ı kurar."""
        logger = logging.getLogger(f"{__name__}.DataLoader")
        logger.setLevel(getattr(logging, log_level.upper()))

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def load_csv(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        CSV dosyasını yükler.

        Args:
            file_path (str): CSV dosya yolu
            **kwargs: pandas.read_csv için ek parametreler

        Returns:
            pd.DataFrame: Yüklenen veri

        Raises:
            FileNotFoundError: Dosya bulunamadığında
            pd.errors.EmptyDataError: Dosya boş olduğunda
            pd.errors.ParserError: CSV parsing hatası
        """
        try:
            self.logger.info(f"CSV dosyası yükleniyor: {file_path}")

            # Dosya varlığını kontrol et
            if not Path(file_path).exists():
                raise FileNotFoundError(f"Dosya bulunamadı: {file_path}")

            # CSV okuma parametrelerini ayarla
            read_params = {
                "sep": PATH_CONFIG["csv_separator"],
                "decimal": PATH_CONFIG["decimal_separator"],
                "encoding": PATH_CONFIG["output_encoding"],
                **kwargs,
            }

            # CSV'yi yükle
            self.data = pd.read_csv(file_path, **read_params)

            # Temel bilgileri logla
            self.logger.info(
                SUCCESS_MESSAGES["data_loaded"].format(shape=self.data.shape)
            )

            return self.data

        except Exception as e:
            self.logger.error(f"CSV yükleme hatası: {str(e)}")
            raise

    def parse_timestamps(
        self, timestamp_column: str = "timestamp", format: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Timestamp sütununu datetime'a çevirir.

        Args:
            timestamp_column (str): Timestamp sütun adı
            format (str, optional): Timestamp formatı

        Returns:
            pd.DataFrame: Timestamp'ları parse edilmiş veri

        Raises:
            KeyError: Timestamp sütunu bulunamadığında
            ValueError: Timestamp parsing hatası
        """
        try:
            if self.data is None:
                raise ValueError("Önce veri yüklenmelidir")

            if timestamp_column not in self.data.columns:
                raise KeyError(f"Timestamp sütunu bulunamadı: {timestamp_column}")

            self.logger.info(f"Timestamp parsing başlatılıyor: {timestamp_column}")

            # Timestamp formatını belirle
            if format is None:
                format = str(PATH_CONFIG["timestamp_format"])

            # Timestamp'ları parse et
            self.data[timestamp_column] = pd.to_datetime(
                self.data[timestamp_column], format=format, errors="coerce"
            )

            # Datetime index oluştur
            self.data.set_index(timestamp_column, inplace=True)
            self.data.sort_index(inplace=True)

            # Metadata güncelle
            self.metadata["timestamp_range"] = {
                "start": self.data.index.min(),
                "end": self.data.index.max(),
                "days": len(self.data),
            }

            self.logger.info("Timestamp parsing tamamlandı")
            return self.data

        except Exception as e:
            self.logger.error(
                ERROR_MESSAGES["timestamp_parsing_error"].format(error=str(e))
            )
            raise

    def validate_schema(self, required_columns: Optional[List[str]] = None) -> bool:
        """
        Veri şemasını doğrular.

        Args:
            required_columns (List[str], optional): Gerekli sütunlar

        Returns:
            bool: Şema geçerli mi

        Raises:
            ValueError: Şema geçersiz olduğunda
        """
        try:
            if self.data is None:
                raise ValueError("Önce veri yüklenmelidir")

            if required_columns is None:
                required_columns = list(PATH_CONFIG["required_columns"])

            self.logger.info("Şema validasyonu başlatılıyor")

            # Timestamp index olarak ayarlandıysa, sütun listesinden çıkar
            available_columns = list(self.data.columns)
            if isinstance(self.data.index, pd.DatetimeIndex):
                # Timestamp index olarak ayarlandı, sütun listesinden çıkar
                required_columns = [
                    col for col in required_columns if col != "timestamp"
                ]

            # Eksik sütunları kontrol et
            missing_columns = set(required_columns) - set(available_columns)
            if missing_columns:
                error_msg = ERROR_MESSAGES["missing_columns"].format(
                    missing_columns=list(missing_columns)
                )
                self.logger.error(error_msg)
                raise ValueError(error_msg)

            # Veri tiplerini kontrol et
            invalid_types = []
            for col in required_columns:
                if not pd.api.types.is_numeric_dtype(self.data[col]):
                    invalid_types.append(f"{col}: numeric bekleniyor")

            if invalid_types:
                error_msg = ERROR_MESSAGES["invalid_data_types"].format(
                    invalid_types=invalid_types
                )
                self.logger.error(error_msg)
                raise ValueError(error_msg)

            # Metadata güncelle
            self.metadata["schema"] = {
                "columns": list(self.data.columns),
                "dtypes": dict(self.data.dtypes),
                "shape": self.data.shape,
                "index_type": type(self.data.index).__name__,
            }

            self.logger.info("Şema validasyonu başarılı")
            return True

        except Exception as e:
            self.logger.error(f"Şema validasyon hatası: {str(e)}")
            raise

    def get_basic_info(self) -> Dict[str, Any]:
        """
        Veri hakkında temel bilgileri döndürür.

        Returns:
            Dict[str, Any]: Temel veri bilgileri
        """
        if self.data is None:
            return {}

        info = {
            "shape": self.data.shape,
            "columns": list(self.data.columns),
            "dtypes": dict(self.data.dtypes),
            "memory_usage": self.data.memory_usage(deep=True).sum(),
            "date_range": self.metadata.get("timestamp_range", {}),
            "missing_values": dict(self.data.isnull().sum()),
            "duplicates": self.data.duplicated().sum(),
        }

        return info

    def load_and_validate(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        Veriyi yükler ve temel validasyonları gerçekleştirir.

        Args:
            file_path (str): CSV dosya yolu
            **kwargs: Ek parametreler

        Returns:
            pd.DataFrame: Yüklenmiş ve validasyonu yapılmış veri
        """
        try:
            # Veriyi yükle
            self.load_csv(file_path, **kwargs)

            # Timestamp'ları parse et
            self.parse_timestamps()

            # Şemayı doğrula
            self.validate_schema()

            # Temel bilgileri al
            self.metadata["basic_info"] = self.get_basic_info()

            self.logger.info("Veri yükleme ve validasyon tamamlandı")
            return self.data

        except Exception as e:
            self.logger.error(f"Veri yükleme ve validasyon hatası: {str(e)}")
            raise

    def save_metadata(self, output_path: str) -> None:
        """
        Metadata'yı JSON dosyasına kaydeder.

        Args:
            output_path (str): Çıktı dosya yolu
        """
        try:
            import json

            # Timestamp'ları string'e çevir
            metadata_copy = self.metadata.copy()
            if "timestamp_range" in metadata_copy:
                timestamp_range = metadata_copy["timestamp_range"]
                if "start" in timestamp_range:
                    timestamp_range["start"] = str(timestamp_range["start"])
                if "end" in timestamp_range:
                    timestamp_range["end"] = str(timestamp_range["end"])

            # JSON'a kaydet
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(metadata_copy, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Metadata kaydedildi: {output_path}")

        except Exception as e:
            self.logger.error(f"Metadata kaydetme hatası: {str(e)}")
            raise


class ConfigurationProvider:
    """Genel konfigürasyon sağlayıcı"""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def get_config(self) -> Dict[str, Any]:
        """Genel konfigürasyonu döndürür"""
        return PATH_CONFIG


class BitcoinConfigProvider(ConfigurationProvider):
    """Bitcoin verileri için konfigürasyon sağlayıcı"""

    def __init__(self, logger: logging.Logger):
        super().__init__(logger)

    def get_config(self) -> Dict[str, Any]:
        """Bitcoin konfigürasyonunu döndürür"""
        base_config = super().get_config()
        bitcoin_specific = {
            "required_columns": ["timestamp", "open", "high", "low", "close", "volume"],
            "timestamp_format": "%Y-%m-%d %H:%M:%S",
            "validation_rules": {
                "price_positive": True,
                "volume_positive": True,
                "ohlc_consistency": True,
            },
        }
        return {**base_config, **bitcoin_specific}
