"""
DataRepository - Veri erişimi için repository pattern

Bu modül veri erişim işlemlerini soyutlar ve farklı veri kaynakları için
tutarlı bir interface sağlar.
"""

import pandas as pd
import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from pathlib import Path

from .data_reader import CSVDataSource, JSONDataSource
from .metadata_manager import MetadataManager


class DataRepository(ABC):
    """Veri repository'si için abstract base class"""

    @abstractmethod
    def load_data(self, source: str, **kwargs) -> pd.DataFrame:
        """Veri yükler"""
        pass

    @abstractmethod
    def save_data(self, data: pd.DataFrame, destination: str, **kwargs) -> None:
        """Veri kaydeder"""
        pass

    @abstractmethod
    def exists(self, path: str) -> bool:
        """Dosya/veri kaynağı var mı kontrol eder"""
        pass

    @abstractmethod
    def get_metadata(self, path: str) -> Optional[Dict[str, Any]]:
        """Veri kaynağı metadata'sını alır"""
        pass


class CSVRepository(DataRepository):
    """CSV dosyaları için repository"""

    def __init__(
        self, logger: logging.Logger, metadata_manager: Optional[MetadataManager] = None
    ):
        """
        CSVRepository'yi başlatır.

        Args:
            logger (logging.Logger): Logger instance
            metadata_manager (MetadataManager, optional): Metadata manager
        """
        self.logger = logger
        self.metadata_manager = metadata_manager
        self.data_source = CSVDataSource(logger)

    def load_data(self, source: str, **kwargs) -> pd.DataFrame:
        """
        CSV dosyasından veri yükler.

        Args:
            source (str): CSV dosya yolu
            **kwargs: Ek parametreler

        Returns:
            pd.DataFrame: Yüklenen veri
        """
        try:
            self.logger.info(f"CSV verisi yükleniyor: {source}")
            data = self.data_source.load_data(source, **kwargs)

            # Metadata'ya bilgi ekle
            if self.metadata_manager:
                self.metadata_manager.add_metadata(
                    "data_source",
                    {
                        "type": "csv",
                        "path": source,
                        "shape": data.shape,
                        "columns": list(data.columns),
                    },
                )

            return data

        except Exception as e:
            self.logger.error(f"CSV yükleme hatası: {str(e)}")
            raise

    def save_data(self, data: pd.DataFrame, destination: str, **kwargs) -> None:
        """
        Veriyi CSV dosyasına kaydeder.

        Args:
            data (pd.DataFrame): Kaydedilecek veri
            destination (str): Hedef dosya yolu
            **kwargs: Ek parametreler
        """
        try:
            self.logger.info(f"CSV verisi kaydediliyor: {destination}")

            # Dizin oluştur
            Path(destination).parent.mkdir(parents=True, exist_ok=True)

            # CSV parametrelerini ayarla
            save_params = {"index": False, "encoding": "utf-8", **kwargs}

            # CSV'ye kaydet
            data.to_csv(destination, **save_params)

            self.logger.info(f"CSV verisi başarıyla kaydedildi: {destination}")

        except Exception as e:
            self.logger.error(f"CSV kaydetme hatası: {str(e)}")
            raise

    def exists(self, path: str) -> bool:
        """CSV dosyası var mı kontrol eder"""
        return Path(path).exists()

    def get_metadata(self, path: str) -> Optional[Dict[str, Any]]:
        """CSV dosyası metadata'sını alır"""
        try:
            if not self.exists(path):
                return None

            file_path = Path(path)
            return {
                "type": "csv",
                "path": str(path),
                "size": file_path.stat().st_size,
                "modified": file_path.stat().st_mtime,
                "exists": True,
            }
        except Exception as e:
            self.logger.error(f"CSV metadata alma hatası: {str(e)}")
            return None


class JSONRepository(DataRepository):
    """JSON dosyaları için repository"""

    def __init__(
        self, logger: logging.Logger, metadata_manager: Optional[MetadataManager] = None
    ):
        """
        JSONRepository'yi başlatır.

        Args:
            logger (logging.Logger): Logger instance
            metadata_manager (MetadataManager, optional): Metadata manager
        """
        self.logger = logger
        self.metadata_manager = metadata_manager
        self.data_source = JSONDataSource(logger)

    def load_data(self, source: str, **kwargs) -> pd.DataFrame:
        """
        JSON dosyasından veri yükler.

        Args:
            source (str): JSON dosya yolu
            **kwargs: Ek parametreler

        Returns:
            pd.DataFrame: Yüklenen veri
        """
        try:
            self.logger.info(f"JSON verisi yükleniyor: {source}")
            data = self.data_source.load_data(source, **kwargs)

            # Metadata'ya bilgi ekle
            if self.metadata_manager:
                self.metadata_manager.add_metadata(
                    "data_source",
                    {
                        "type": "json",
                        "path": source,
                        "shape": data.shape,
                        "columns": list(data.columns),
                    },
                )

            return data

        except Exception as e:
            self.logger.error(f"JSON yükleme hatası: {str(e)}")
            raise

    def save_data(self, data: pd.DataFrame, destination: str, **kwargs) -> None:
        """
        Veriyi JSON dosyasına kaydeder.

        Args:
            data (pd.DataFrame): Kaydedilecek veri
            destination (str): Hedef dosya yolu
            **kwargs: Ek parametreler
        """
        try:
            self.logger.info(f"JSON verisi kaydediliyor: {destination}")

            # Dizin oluştur
            Path(destination).parent.mkdir(parents=True, exist_ok=True)

            # JSON parametrelerini ayarla
            save_params = {
                "orient": "records",
                "indent": 2,
                "ensure_ascii": False,
                **kwargs,
            }

            # JSON'a kaydet
            data.to_json(destination, **save_params)

            self.logger.info(f"JSON verisi başarıyla kaydedildi: {destination}")

        except Exception as e:
            self.logger.error(f"JSON kaydetme hatası: {str(e)}")
            raise

    def exists(self, path: str) -> bool:
        """JSON dosyası var mı kontrol eder"""
        return Path(path).exists()

    def get_metadata(self, path: str) -> Optional[Dict[str, Any]]:
        """JSON dosyası metadata'sını alır"""
        try:
            if not self.exists(path):
                return None

            file_path = Path(path)
            return {
                "type": "json",
                "path": str(path),
                "size": file_path.stat().st_size,
                "modified": file_path.stat().st_mtime,
                "exists": True,
            }
        except Exception as e:
            self.logger.error(f"JSON metadata alma hatası: {str(e)}")
            return None


class RepositoryFactory:
    """Repository factory sınıfı"""

    @staticmethod
    def create_csv_repository(
        logger: logging.Logger, metadata_manager: Optional[MetadataManager] = None
    ) -> CSVRepository:
        """CSV repository oluşturur"""
        return CSVRepository(logger, metadata_manager)

    @staticmethod
    def create_json_repository(
        logger: logging.Logger, metadata_manager: Optional[MetadataManager] = None
    ) -> JSONRepository:
        """JSON repository oluşturur"""
        return JSONRepository(logger, metadata_manager)

    @staticmethod
    def create_repository_by_extension(
        file_path: str,
        logger: logging.Logger,
        metadata_manager: Optional[MetadataManager] = None,
    ) -> DataRepository:
        """Dosya uzantısına göre repository oluşturur"""
        extension = Path(file_path).suffix.lower()

        if extension == ".csv":
            return RepositoryFactory.create_csv_repository(logger, metadata_manager)
        elif extension == ".json":
            return RepositoryFactory.create_json_repository(logger, metadata_manager)
        else:
            raise ValueError(f"Desteklenmeyen dosya uzantısı: {extension}")
