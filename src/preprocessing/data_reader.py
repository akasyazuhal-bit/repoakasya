"""
DataReader - Veri okuma sorumluluğu

Bu modül sadece veri okuma işlemlerini yönetir.
SRP prensibi gereği tek sorumluluğa sahiptir.
"""

import pandas as pd
import logging
from pathlib import Path
from abc import ABC, abstractmethod

from .config import PATH_CONFIG, SUCCESS_MESSAGES


class DataSource(ABC):
    """Veri kaynağı için abstract base class"""

    @abstractmethod
    def load_data(self, source: str, **kwargs) -> pd.DataFrame:
        """Veri yükleme işlemi"""
        pass


class CSVDataSource(DataSource):
    """CSV dosyalarından veri okuma"""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def load_data(self, source: str, **kwargs) -> pd.DataFrame:
        """
        CSV dosyasını yükler.

        Args:
            source (str): CSV dosya yolu
            **kwargs: pandas.read_csv için ek parametreler

        Returns:
            pd.DataFrame: Yüklenen veri

        Raises:
            FileNotFoundError: Dosya bulunamadığında
            pd.errors.EmptyDataError: Dosya boş olduğunda
            pd.errors.ParserError: CSV parsing hatası
        """
        try:
            self.logger.info(f"CSV dosyası yükleniyor: {source}")

            # Dosya varlığını kontrol et
            if not Path(source).exists():
                raise FileNotFoundError(f"Dosya bulunamadı: {source}")

            # CSV okuma parametrelerini ayarla
            read_params = {
                "sep": PATH_CONFIG["csv_separator"],
                "decimal": PATH_CONFIG["decimal_separator"],
                "encoding": PATH_CONFIG["output_encoding"],
                **kwargs,
            }

            # CSV'yi yükle
            data = pd.read_csv(source, **read_params)

            # Temel bilgileri logla
            self.logger.info(SUCCESS_MESSAGES["data_loaded"].format(shape=data.shape))

            return data

        except Exception as e:
            self.logger.error(f"CSV yükleme hatası: {str(e)}")
            raise


class JSONDataSource(DataSource):
    """JSON dosyalarından veri okuma"""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def load_data(self, source: str, **kwargs) -> pd.DataFrame:
        """
        JSON dosyasını yükler.

        Args:
            source (str): JSON dosya yolu
            **kwargs: pandas.read_json için ek parametreler

        Returns:
            pd.DataFrame: Yüklenen veri
        """
        try:
            self.logger.info(f"JSON dosyası yükleniyor: {source}")

            # Dosya varlığını kontrol et
            if not Path(source).exists():
                raise FileNotFoundError(f"Dosya bulunamadı: {source}")

            # JSON'u yükle
            data = pd.read_json(source, **kwargs)

            self.logger.info(SUCCESS_MESSAGES["data_loaded"].format(shape=data.shape))

            return data

        except Exception as e:
            self.logger.error(f"JSON yükleme hatası: {str(e)}")
            raise


class DataReader:
    """
    Veri okuma işlemlerini yöneten sınıf.
    SRP prensibi gereği sadece veri okuma sorumluluğu taşır.
    """

    def __init__(self, data_source: DataSource):
        """
        DataReader'ı başlatır.

        Args:
            data_source (DataSource): Veri kaynağı implementasyonu
        """
        self.data_source = data_source

    def load_data(self, source: str, **kwargs) -> pd.DataFrame:
        """
        Veri kaynağından veri yükler.

        Args:
            source (str): Veri kaynağı yolu
            **kwargs: Ek parametreler

        Returns:
            pd.DataFrame: Yüklenen veri
        """
        return self.data_source.load_data(source, **kwargs)
