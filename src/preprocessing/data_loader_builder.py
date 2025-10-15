"""
DataLoaderBuilder - DataLoader oluşturmak için builder pattern

Bu modül karmaşık DataLoader objelerini adım adım oluşturmak için builder pattern uygular.
"""

import logging
from typing import Optional

from .data_loader import DataLoader, BitcoinConfigProvider, ConfigurationProvider
from .data_reader import DataReader, DataSource, CSVDataSource, JSONDataSource
from .timestamp_parser import TimestampParser
from .schema_validator import (
    SchemaValidator,
    ValidationStrategy,
    BitcoinValidationStrategy,
)
from .metadata_manager import MetadataManager


class DataLoaderBuilder:
    """DataLoader oluşturmak için builder sınıfı"""

    def __init__(self):
        """Builder'ı başlatır"""
        self._config_provider: Optional[ConfigurationProvider] = None
        self._data_source: Optional[DataSource] = None
        self._logger: Optional[logging.Logger] = None
        self._validation_strategy: Optional[ValidationStrategy] = None
        self._log_level: str = "INFO"
        self._logger_name: str = "DataLoader"

    def with_config(
        self, config_provider: ConfigurationProvider
    ) -> "DataLoaderBuilder":
        """
        Konfigürasyon sağlayıcı ekler.

        Args:
            config_provider (ConfigurationProvider): Konfigürasyon sağlayıcı

        Returns:
            DataLoaderBuilder: Builder instance (method chaining)
        """
        self._config_provider = config_provider
        return self

    def with_data_source(self, data_source: DataSource) -> "DataLoaderBuilder":
        """
        Veri kaynağı ekler.

        Args:
            data_source (DataSource): Veri kaynağı

        Returns:
            DataLoaderBuilder: Builder instance (method chaining)
        """
        self._data_source = data_source
        return self

    def with_logger(self, logger: logging.Logger) -> "DataLoaderBuilder":
        """
        Logger ekler.

        Args:
            logger (logging.Logger): Logger instance

        Returns:
            DataLoaderBuilder: Builder instance (method chaining)
        """
        self._logger = logger
        return self

    def with_validation_strategy(
        self, validation_strategy: ValidationStrategy
    ) -> "DataLoaderBuilder":
        """
        Validasyon stratejisi ekler.

        Args:
            validation_strategy (ValidationStrategy): Validasyon stratejisi

        Returns:
            DataLoaderBuilder: Builder instance (method chaining)
        """
        self._validation_strategy = validation_strategy
        return self

    def with_log_level(self, log_level: str) -> "DataLoaderBuilder":
        """
        Log seviyesi ayarlar.

        Args:
            log_level (str): Log seviyesi

        Returns:
            DataLoaderBuilder: Builder instance (method chaining)
        """
        self._log_level = log_level
        return self

    def with_logger_name(self, logger_name: str) -> "DataLoaderBuilder":
        """
        Logger adı ayarlar.

        Args:
            logger_name (str): Logger adı

        Returns:
            DataLoaderBuilder: Builder instance (method chaining)
        """
        self._logger_name = logger_name
        return self

    def build(self) -> DataLoader:
        """
        DataLoader'ı oluşturur.

        Returns:
            DataLoader: Oluşturulan DataLoader instance

        Raises:
            ValueError: Gerekli bileşenler eksikse
        """
        # Gerekli bileşenleri kontrol et ve oluştur
        config_provider = self._get_config_provider()
        logger = self._get_logger()
        data_source = self._get_data_source(logger)
        data_reader = DataReader(data_source)
        timestamp_parser = TimestampParser(logger)
        validation_strategy = self._get_validation_strategy(logger)
        schema_validator = SchemaValidator(validation_strategy, logger)
        metadata_manager = MetadataManager(logger)

        # DataLoader oluştur
        return DataLoader(
            data_reader=data_reader,
            timestamp_parser=timestamp_parser,
            schema_validator=schema_validator,
            metadata_manager=metadata_manager,
            config_provider=config_provider,
            logger=logger,
        )

    def _get_config_provider(self) -> ConfigurationProvider:
        """Konfigürasyon sağlayıcı alır veya varsayılan oluşturur"""
        if self._config_provider is None:
            # Geçici logger oluştur
            temp_logger = logging.getLogger("DefaultConfigProvider")
            return BitcoinConfigProvider(temp_logger)
        return self._config_provider

    def _get_logger(self) -> logging.Logger:
        """Logger alır veya oluşturur"""
        if self._logger is not None:
            return self._logger

        logger = logging.getLogger(self._logger_name)
        logger.setLevel(getattr(logging, self._log_level.upper()))

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _get_data_source(self, logger: logging.Logger) -> DataSource:
        """Veri kaynağı alır veya varsayılan oluşturur"""
        if self._data_source is not None:
            return self._data_source
        return CSVDataSource(logger)

    def _get_validation_strategy(self, logger: logging.Logger) -> ValidationStrategy:
        """Validasyon stratejisi alır veya varsayılan oluşturur"""
        if self._validation_strategy is not None:
            return self._validation_strategy
        return BitcoinValidationStrategy(logger)


# Convenience methods for common use cases
class DataLoaderBuilderFactory:
    """DataLoaderBuilder oluşturmak için factory"""

    @staticmethod
    def create_bitcoin_builder() -> DataLoaderBuilder:
        """Bitcoin verisi için builder oluşturur"""
        # Geçici logger oluştur
        temp_logger = logging.getLogger("BitcoinDataLoader")
        return (
            DataLoaderBuilder()
            .with_config(BitcoinConfigProvider(temp_logger))
            .with_data_source(CSVDataSource(temp_logger))
            .with_validation_strategy(BitcoinValidationStrategy(temp_logger))
            .with_logger_name("BitcoinDataLoader")
        )

    @staticmethod
    def create_json_builder() -> DataLoaderBuilder:
        """JSON verisi için builder oluşturur"""
        # Geçici logger oluştur
        temp_logger = logging.getLogger("JSONDataLoader")
        return (
            DataLoaderBuilder()
            .with_config(BitcoinConfigProvider(temp_logger))
            .with_data_source(JSONDataSource(temp_logger))
            .with_validation_strategy(BitcoinValidationStrategy(temp_logger))
            .with_logger_name("JSONDataLoader")
        )

    @staticmethod
    def create_custom_builder() -> DataLoaderBuilder:
        """Özel konfigürasyon için builder oluşturur"""
        return DataLoaderBuilder()
