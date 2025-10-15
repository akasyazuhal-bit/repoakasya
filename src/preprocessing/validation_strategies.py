"""
ValidationStrategies - Farklı validasyon stratejileri

Bu modül farklı veri türleri için özelleşmiş validasyon stratejileri sağlar.
Strategy pattern uygular.
"""

import pandas as pd
import logging
from typing import List

from .schema_validator import ValidationStrategy, ValidationResult


class StockValidationStrategy(ValidationStrategy):
    """Hisse senedi verisi için validasyon stratejisi"""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def validate(self, data: pd.DataFrame) -> ValidationResult:
        """Hisse senedi verisi validasyonu"""
        result = ValidationResult(is_valid=True)

        # OHLC tutarlılık kontrolü
        if not self._validate_ohlc_consistency(data, result):
            pass

        # Volume pozitiflik kontrolü
        if not self._validate_volume_positive(data, result):
            pass

        # Fiyat pozitiflik kontrolü
        if not self._validate_price_positive(data, result):
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

    def _validate_price_positive(
        self, data: pd.DataFrame, result: ValidationResult
    ) -> bool:
        """Fiyat pozitiflik kontrolü"""
        try:
            price_columns = ["open", "high", "low", "close"]
            for col in price_columns:
                if col in data.columns and (data[col] <= 0).any():
                    result.add_error(f"{col} değerleri pozitif olmalı")
                    return False

            return True
        except Exception as e:
            result.add_error(f"Fiyat pozitiflik kontrolü hatası: {str(e)}")
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


class CryptoValidationStrategy(ValidationStrategy):
    """Kripto para verisi için validasyon stratejisi"""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def validate(self, data: pd.DataFrame) -> ValidationResult:
        """Kripto para verisi validasyonu"""
        result = ValidationResult(is_valid=True)

        # OHLC tutarlılık kontrolü
        if not self._validate_ohlc_consistency(data, result):
            pass

        # Volume pozitiflik kontrolü
        if not self._validate_volume_positive(data, result):
            pass

        # Fiyat pozitiflik kontrolü
        if not self._validate_price_positive(data, result):
            pass

        # Aşırı volatilite kontrolü
        if not self._validate_volatility(data, result):
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

    def _validate_price_positive(
        self, data: pd.DataFrame, result: ValidationResult
    ) -> bool:
        """Fiyat pozitiflik kontrolü"""
        try:
            price_columns = ["open", "high", "low", "close"]
            for col in price_columns:
                if col in data.columns and (data[col] <= 0).any():
                    result.add_error(f"{col} değerleri pozitif olmalı")
                    return False

            return True
        except Exception as e:
            result.add_error(f"Fiyat pozitiflik kontrolü hatası: {str(e)}")
            return False

    def _validate_volatility(
        self, data: pd.DataFrame, result: ValidationResult
    ) -> bool:
        """Aşırı volatilite kontrolü"""
        try:
            if "close" not in data.columns:
                return True

            # Günlük değişim oranını hesapla
            daily_returns = data["close"].pct_change().dropna()

            # Aşırı volatilite threshold'u (%50)
            volatility_threshold = 0.5
            extreme_volatility = abs(daily_returns) > volatility_threshold

            if extreme_volatility.any():
                extreme_count = extreme_volatility.sum()
                result.add_warning(
                    f"Aşırı volatilite tespit edildi: {extreme_count} gün"
                )

            return True
        except Exception as e:
            result.add_error(f"Volatilite kontrolü hatası: {str(e)}")
            return False


class ForexValidationStrategy(ValidationStrategy):
    """Forex verisi için validasyon stratejisi"""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def validate(self, data: pd.DataFrame) -> ValidationResult:
        """Forex verisi validasyonu"""
        result = ValidationResult(is_valid=True)

        # OHLC tutarlılık kontrolü
        if not self._validate_ohlc_consistency(data, result):
            pass

        # Spread kontrolü
        if not self._validate_spread(data, result):
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

    def _validate_spread(self, data: pd.DataFrame, result: ValidationResult) -> bool:
        """Spread kontrolü"""
        try:
            if "bid" in data.columns and "ask" in data.columns:
                # Spread hesapla
                spread = data["ask"] - data["bid"]

                # Negatif spread kontrolü
                if (spread < 0).any():
                    result.add_error("Negatif spread tespit edildi")
                    return False

                # Aşırı yüksek spread kontrolü
                spread_pct = spread / data["bid"] * 100
                if (spread_pct > 1.0).any():  # %1'den fazla spread
                    result.add_warning("Aşırı yüksek spread tespit edildi")

            return True
        except Exception as e:
            result.add_error(f"Spread kontrolü hatası: {str(e)}")
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


class ValidationStrategyFactory:
    """Validasyon stratejisi factory sınıfı"""

    @staticmethod
    def create_strategy(
        strategy_type: str, logger: logging.Logger
    ) -> ValidationStrategy:
        """
        Validasyon stratejisi oluşturur.

        Args:
            strategy_type (str): Strateji türü ('bitcoin', 'stock', 'crypto', 'forex')
            logger (logging.Logger): Logger instance

        Returns:
            ValidationStrategy: Oluşturulan strateji

        Raises:
            ValueError: Desteklenmeyen strateji türü
        """
        strategy_type = strategy_type.lower()

        if strategy_type == "bitcoin":
            from .schema_validator import BitcoinValidationStrategy

            return BitcoinValidationStrategy(logger)
        elif strategy_type == "stock":
            return StockValidationStrategy(logger)
        elif strategy_type == "crypto":
            return CryptoValidationStrategy(logger)
        elif strategy_type == "forex":
            return ForexValidationStrategy(logger)
        else:
            raise ValueError(f"Desteklenmeyen validasyon stratejisi: {strategy_type}")

    @staticmethod
    def get_available_strategies() -> List[str]:
        """Kullanılabilir stratejileri döndürür"""
        return ["bitcoin", "stock", "crypto", "forex"]
