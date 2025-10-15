"""
SchemaValidator - Şema validasyonu sorumluluğu

Bu modül sadece şema validasyonu işlemlerini yönetir.
SRP prensibi gereği tek sorumluluğa sahiptir.
"""

import pandas as pd
import logging
from typing import Optional, List
from abc import ABC, abstractmethod

from .config import PATH_CONFIG, ERROR_MESSAGES


class ValidationResult:
    """Validasyon sonucu için data class"""

    def __init__(
        self,
        is_valid: bool,
        errors: Optional[List[str]] = None,
        warnings: Optional[List[str]] = None,
    ):
        self.is_valid = is_valid
        self.errors = errors or []
        self.warnings = warnings or []

    def add_error(self, error: str):
        """Hata ekle"""
        self.errors.append(error)
        self.is_valid = False

    def add_warning(self, warning: str):
        """Uyarı ekle"""
        self.warnings.append(warning)


class ValidationStrategy(ABC):
    """Validasyon stratejisi için abstract base class"""

    @abstractmethod
    def validate(self, data: pd.DataFrame) -> ValidationResult:
        """Validasyon işlemi"""
        pass


class BitcoinValidationStrategy(ValidationStrategy):
    """Bitcoin verisi için özel validasyon stratejisi"""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def validate(self, data: pd.DataFrame) -> ValidationResult:
        """Bitcoin verisi validasyonu"""
        result = ValidationResult(is_valid=True)

        # OHLC tutarlılık kontrolü
        if not self._validate_ohlc_consistency(data, result):
            pass

        # Volume pozitiflik kontrolü
        if not self._validate_volume_positive(data, result):
            pass

        # Timestamp sıralama kontrolü
        if not self._validate_timestamp_order(data, result):
            pass

        return result

    def _validate_ohlc_consistency(
        self, data: pd.DataFrame, result: ValidationResult
    ) -> bool:
        """OHLC tutarlılık kontrolü"""
        try:
            ohlc_columns = ["open", "high", "low", "close"]
            if not all(col in data.columns for col in ohlc_columns):
                result.add_error("OHLC sütunları eksik")
                return False

            # High >= max(open, close) kontrolü
            high_check = data["high"] >= data[["open", "close"]].max(axis=1)
            if not high_check.all():
                result.add_error("High değeri open/close'dan küçük")
                return False

            # Low <= min(open, close) kontrolü
            low_check = data["low"] <= data[["open", "close"]].min(axis=1)
            if not low_check.all():
                result.add_error("Low değeri open/close'dan büyük")
                return False

            return True
        except Exception as e:
            result.add_error(f"OHLC tutarlılık kontrolü hatası: {str(e)}")
            return False

    def _validate_volume_positive(
        self, data: pd.DataFrame, result: ValidationResult
    ) -> bool:
        """Volume pozitiflik kontrolü"""
        try:
            if "volume" not in data.columns:
                result.add_error("Volume sütunu eksik")
                return False

            if (data["volume"] < 0).any():
                result.add_error("Volume değerleri negatif olamaz")
                return False

            return True
        except Exception as e:
            result.add_error(f"Volume kontrolü hatası: {str(e)}")
            return False

    def _validate_timestamp_order(
        self, data: pd.DataFrame, result: ValidationResult
    ) -> bool:
        """Timestamp sıralama kontrolü"""
        try:
            if not data.index.is_monotonic_increasing:
                result.add_warning("Timestamp'lar sıralı değil")
                return False

            return True
        except Exception as e:
            result.add_error(f"Timestamp sıralama kontrolü hatası: {str(e)}")
            return False


class SchemaValidator:
    """
    Şema validasyonu işlemlerini yöneten sınıf.
    SRP prensibi gereği sadece şema validasyonu sorumluluğu taşır.
    """

    def __init__(self, validation_strategy: ValidationStrategy, logger: logging.Logger):
        """
        SchemaValidator'ı başlatır.

        Args:
            validation_strategy (ValidationStrategy): Validasyon stratejisi
            logger (logging.Logger): Logger instance
        """
        self.validation_strategy = validation_strategy
        self.logger = logger

    def validate_schema(
        self, data: pd.DataFrame, required_columns: Optional[List[str]] = None
    ) -> ValidationResult:
        """
        Veri şemasını doğrular.

        Args:
            data (pd.DataFrame): Validasyon yapılacak veri
            required_columns (List[str], optional): Gerekli sütunlar

        Returns:
            ValidationResult: Validasyon sonucu
        """
        try:
            if data is None or data.empty:
                raise ValueError("Veri boş veya None")

            if required_columns is None:
                required_columns = list(PATH_CONFIG["required_columns"])

            self.logger.info("Şema validasyonu başlatılıyor")

            result = ValidationResult(is_valid=True)

            # Eksik sütunları kontrol et
            missing_columns = set(required_columns) - set(data.columns)
            if missing_columns:
                result.add_error(
                    ERROR_MESSAGES["missing_columns"].format(
                        missing_columns=list(missing_columns)
                    )
                )

            # Veri tiplerini kontrol et
            self._validate_data_types(data, required_columns, result)

            # Özel validasyon stratejisini uygula
            if result.is_valid:
                strategy_result = self.validation_strategy.validate(data)
                result.is_valid = strategy_result.is_valid
                result.errors.extend(strategy_result.errors)
                result.warnings.extend(strategy_result.warnings)

            if result.is_valid:
                self.logger.info("Şema validasyonu başarılı")
            else:
                self.logger.error(f"Şema validasyonu başarısız: {result.errors}")

            return result

        except Exception as e:
            self.logger.error(f"Şema validasyon hatası: {str(e)}")
            raise

    def _validate_data_types(
        self, data: pd.DataFrame, required_columns: List[str], result: ValidationResult
    ):
        """Veri tiplerini kontrol et"""
        invalid_types = []

        for col in required_columns:
            if col not in data.columns:
                continue

            if col == "timestamp":
                if not pd.api.types.is_datetime64_any_dtype(data[col]):
                    invalid_types.append(f"{col}: datetime bekleniyor")
            else:
                if not pd.api.types.is_numeric_dtype(data[col]):
                    invalid_types.append(f"{col}: numeric bekleniyor")

        if invalid_types:
            result.add_error(
                ERROR_MESSAGES["invalid_data_types"].format(invalid_types=invalid_types)
            )
