"""
Bitcoin Data Preprocessing Package

Bu paket Bitcoin fiyat verilerini makine öğrenmesi için hazırlamak üzere
geliştirilmiş kapsamlı bir veri ön işleme sistemidir.

Ana Sınıflar:
- DataLoader: Veri yükleme ve başlangıç validasyonu
- DataValidator: Kapsamlı veri validasyonu
- OutlierDetector: İleri seviye anomali tespiti
- StatisticalAnalyzer: Detaylı istatistiksel analiz
- DataTransformer: Veri dönüşümleri
- DataCleaner: Veri temizleme operasyonları
- DataPipeline: Ana orkestrasyon sınıfı
"""

from .data_loader import DataLoader
from .validators import DataValidator
from .outlier_detector import OutlierDetector
from .statistical_analyzer import StatisticalAnalyzer
from .transformers import DataTransformer
from .cleaners import DataCleaner
from .pipeline import DataPipeline

__version__ = "1.0.0"
__author__ = "Bitcoin Data Preprocessing Team"

__all__ = [
    "DataLoader",
    "DataValidator",
    "OutlierDetector",
    "StatisticalAnalyzer",
    "DataTransformer",
    "DataCleaner",
    "DataPipeline",
]
