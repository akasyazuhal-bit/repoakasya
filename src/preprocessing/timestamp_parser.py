"""
TimestampParser - Timestamp parsing sorumluluğu

Bu modül sadece timestamp parsing işlemlerini yönetir.
SRP prensibi gereği tek sorumluluğa sahiptir.
"""

import pandas as pd
import logging
from typing import Optional

from .config import PATH_CONFIG, ERROR_MESSAGES


class TimestampParser:
    """
    Timestamp parsing işlemlerini yöneten sınıf.
    SRP prensibi gereği sadece timestamp parsing sorumluluğu taşır.
    """

    def __init__(self, logger: logging.Logger):
        """
        TimestampParser'ı başlatır.

        Args:
            logger (logging.Logger): Logger instance
        """
        self.logger = logger

    def parse_timestamps(
        self,
        data: pd.DataFrame,
        timestamp_column: str = "timestamp",
        format: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Timestamp sütununu datetime'a çevirir.

        Args:
            data (pd.DataFrame): İşlenecek veri
            timestamp_column (str): Timestamp sütun adı
            format (str, optional): Timestamp formatı

        Returns:
            pd.DataFrame: Timestamp'ları parse edilmiş veri

        Raises:
            KeyError: Timestamp sütunu bulunamadığında
            ValueError: Timestamp parsing hatası
        """
        try:
            if data is None or data.empty:
                raise ValueError("Veri boş veya None")

            if timestamp_column not in data.columns:
                raise KeyError(f"Timestamp sütunu bulunamadı: {timestamp_column}")

            self.logger.info(f"Timestamp parsing başlatılıyor: {timestamp_column}")

            # Timestamp formatını belirle
            if format is None:
                format = str(PATH_CONFIG["timestamp_format"])

            # Verinin kopyasını oluştur
            processed_data = data.copy()

            # Timestamp'ları parse et
            processed_data[timestamp_column] = pd.to_datetime(
                processed_data[timestamp_column], format=format, errors="coerce"
            )

            # Datetime index oluştur
            processed_data.set_index(timestamp_column, inplace=True)
            processed_data.sort_index(inplace=True)

            # Metadata oluştur
            metadata = {
                "timestamp_range": {
                    "start": processed_data.index.min(),
                    "end": processed_data.index.max(),
                    "days": len(processed_data),
                }
            }

            self.logger.info("Timestamp parsing tamamlandı")
            return processed_data, metadata

        except Exception as e:
            self.logger.error(
                ERROR_MESSAGES["timestamp_parsing_error"].format(error=str(e))
            )
            raise
