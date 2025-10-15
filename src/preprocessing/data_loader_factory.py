"""
DataLoaderFactory - DataLoader oluşturma factory'si

Bu modül farklı türde DataLoader'lar oluşturmak için factory pattern uygular.
"""

import logging
from typing import Union

from .data_loader import DataLoader, BitcoinConfigProvider
from .data_reader import DataReader, CSVDataSource, JSONDataSource
from .timestamp_parser import TimestampParser
from .schema_validator import SchemaValidator, BitcoinValidationStrategy
from .metadata_manager import MetadataManager


class DataLoaderFactory:
    """DataLoader oluşturmak için factory sınıfı"""

    @staticmethod
    def create_bitcoin_loader(log_level: str = "INFO") -> DataLoader:
        """
        Bitcoin verisi için DataLoader oluşturur.

        Args:
            log_level (str): Logging seviyesi

        Returns:
            DataLoader: Bitcoin DataLoader instance
        """
        # Logger oluştur
        logger = DataLoaderFactory._create_logger("BitcoinDataLoader", log_level)

        # Bileşenleri oluştur
        config_provider = BitcoinConfigProvider(logger)
        data_source = CSVDataSource(logger)
        data_reader = DataReader(data_source)
        timestamp_parser = TimestampParser(logger)
        validation_strategy = BitcoinValidationStrategy(logger)
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

    @staticmethod
    def create_stock_loader(log_level: str = "INFO") -> DataLoader:
        """
        Hisse senedi verisi için DataLoader oluşturur.

        Args:
            log_level (str): Logging seviyesi

        Returns:
            DataLoader: Stock DataLoader instance
        """
        # Logger oluştur
        logger = DataLoaderFactory._create_logger("StockDataLoader", log_level)

        # Bileşenleri oluştur
        config_provider = BitcoinConfigProvider(logger)  # Şimdilik aynı config kullan
        data_source = CSVDataSource(logger)
        data_reader = DataReader(data_source)
        timestamp_parser = TimestampParser(logger)
        validation_strategy = BitcoinValidationStrategy(
            logger
        )  # Şimdilik aynı validation
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

    @staticmethod
    def create_json_loader(log_level: str = "INFO") -> DataLoader:
        """
        JSON verisi için DataLoader oluşturur.

        Args:
            log_level (str): Logging seviyesi

        Returns:
            DataLoader: JSON DataLoader instance
        """
        # Logger oluştur
        logger = DataLoaderFactory._create_logger("JSONDataLoader", log_level)

        # Bileşenleri oluştur
        config_provider = BitcoinConfigProvider(logger)
        data_source = JSONDataSource(logger)
        data_reader = DataReader(data_source)
        timestamp_parser = TimestampParser(logger)
        validation_strategy = BitcoinValidationStrategy(logger)
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

    @staticmethod
    def create_custom_loader(
        data_source_type: str, log_level: str = "INFO"
    ) -> DataLoader:
        """
        Özel veri kaynağı için DataLoader oluşturur.

        Args:
            data_source_type (str): Veri kaynağı türü ('csv', 'json')
            log_level (str): Logging seviyesi

        Returns:
            DataLoader: Custom DataLoader instance
        """
        # Logger oluştur
        logger = DataLoaderFactory._create_logger(
            f"CustomDataLoader_{data_source_type}", log_level
        )

        # Veri kaynağını seç
        data_source: Union[CSVDataSource, JSONDataSource]
        if data_source_type.lower() == "csv":
            data_source = CSVDataSource(logger)
        elif data_source_type.lower() == "json":
            data_source = JSONDataSource(logger)
        else:
            raise ValueError(f"Desteklenmeyen veri kaynağı türü: {data_source_type}")

        # Bileşenleri oluştur
        config_provider = BitcoinConfigProvider(logger)
        data_reader = DataReader(data_source)
        timestamp_parser = TimestampParser(logger)
        validation_strategy = BitcoinValidationStrategy(logger)
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

    @staticmethod
    def _create_logger(name: str, log_level: str) -> logging.Logger:
        """Logger oluşturur"""
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, log_level.upper()))

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger
