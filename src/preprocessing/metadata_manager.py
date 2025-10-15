"""
MetadataManager - Metadata yönetimi sorumluluğu

Bu modül sadece metadata yönetimi işlemlerini yönetir.
SRP prensibi gereği tek sorumluluğa sahiptir.
"""

import json
import logging
from typing import Dict, Any
from pathlib import Path
from datetime import datetime


class MetadataManager:
    """
    Metadata yönetimi işlemlerini yöneten sınıf.
    SRP prensibi gereği sadece metadata yönetimi sorumluluğu taşır.
    """

    def __init__(self, logger: logging.Logger):
        """
        MetadataManager'ı başlatır.

        Args:
            logger (logging.Logger): Logger instance
        """
        self.logger = logger
        self._metadata: Dict[str, Any] = {}

    @property
    def metadata(self) -> Dict[str, Any]:
        """Metadata'ya read-only erişim"""
        return self._metadata.copy()

    def add_metadata(self, key: str, value: Any) -> None:
        """
        Metadata'ya yeni veri ekler.

        Args:
            key (str): Metadata anahtarı
            value (Any): Metadata değeri
        """
        self._metadata[key] = value
        self.logger.debug(f"Metadata eklendi: {key}")

    def update_metadata(self, metadata_dict: Dict[str, Any]) -> None:
        """
        Metadata'yı günceller.

        Args:
            metadata_dict (Dict[str, Any]): Güncellenecek metadata
        """
        self._metadata.update(metadata_dict)
        self.logger.debug(f"Metadata güncellendi: {len(metadata_dict)} öğe")

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """
        Metadata'dan değer alır.

        Args:
            key (str): Metadata anahtarı
            default (Any): Varsayılan değer

        Returns:
            Any: Metadata değeri
        """
        return self._metadata.get(key, default)

    def save_metadata(self, output_path: str) -> None:
        """
        Metadata'yı JSON dosyasına kaydeder.

        Args:
            output_path (str): Çıktı dosya yolu
        """
        try:
            # Timestamp'ları string'e çevir
            metadata_copy = self._serialize_metadata()

            # Dizin oluştur
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            # JSON'a kaydet
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(metadata_copy, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Metadata kaydedildi: {output_path}")

        except Exception as e:
            self.logger.error(f"Metadata kaydetme hatası: {str(e)}")
            raise

    def load_metadata(self, input_path: str) -> None:
        """
        Metadata'yı JSON dosyasından yükler.

        Args:
            input_path (str): Giriş dosya yolu
        """
        try:
            if not Path(input_path).exists():
                raise FileNotFoundError(f"Metadata dosyası bulunamadı: {input_path}")

            with open(input_path, "r", encoding="utf-8") as f:
                self._metadata = json.load(f)

            self.logger.info(f"Metadata yüklendi: {input_path}")

        except Exception as e:
            self.logger.error(f"Metadata yükleme hatası: {str(e)}")
            raise

    def clear_metadata(self) -> None:
        """Metadata'yı temizler"""
        self._metadata.clear()
        self.logger.debug("Metadata temizlendi")

    def _serialize_metadata(self) -> Dict[str, Any]:
        """Metadata'yı JSON serializable hale getirir"""
        metadata_copy = self._metadata.copy()

        # Timestamp'ları string'e çevir
        if "timestamp_range" in metadata_copy:
            timestamp_range = metadata_copy["timestamp_range"]
            if "start" in timestamp_range and timestamp_range["start"] is not None:
                timestamp_range["start"] = str(timestamp_range["start"])
            if "end" in timestamp_range and timestamp_range["end"] is not None:
                timestamp_range["end"] = str(timestamp_range["end"])

        # Diğer datetime objelerini string'e çevir
        for key, value in metadata_copy.items():
            if isinstance(value, datetime):
                metadata_copy[key] = str(value)

        return metadata_copy
