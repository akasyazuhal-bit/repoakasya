"""
Example Usage - OOP iyileştirmeleri sonrası kullanım örnekleri

Bu dosya yeni OOP yapısının nasıl kullanılacağını gösterir.
"""

import logging
from pathlib import Path

# Yeni OOP yapısından import'lar
from .data_loader_factory import DataLoaderFactory
from .data_loader_builder import DataLoaderBuilder, DataLoaderBuilderFactory
from .data_repository import RepositoryFactory
from .validation_strategies import ValidationStrategyFactory


def example_factory_usage():
    """Factory pattern kullanım örneği"""
    print("=== Factory Pattern Kullanım Örneği ===")

    # Bitcoin verisi için DataLoader oluştur
    DataLoaderFactory.create_bitcoin_loader(log_level="INFO")

    # JSON verisi için DataLoader oluştur
    DataLoaderFactory.create_json_loader(log_level="DEBUG")

    # Özel veri kaynağı için DataLoader oluştur
    DataLoaderFactory.create_custom_loader("csv", log_level="WARNING")

    print("Factory pattern ile DataLoader'lar oluşturuldu")


def example_builder_usage():
    """Builder pattern kullanım örneği"""
    print("\n=== Builder Pattern Kullanım Örneği ===")

    # Bitcoin verisi için builder kullan
    (DataLoaderBuilderFactory.create_bitcoin_builder().with_log_level("INFO").build())

    # Özel konfigürasyon ile builder kullan
    (
        DataLoaderBuilder()
        .with_log_level("DEBUG")
        .with_logger_name("CustomLoader")
        .build()
    )

    print("Builder pattern ile DataLoader'lar oluşturuldu")


def example_repository_usage():
    """Repository pattern kullanım örneği"""
    print("\n=== Repository Pattern Kullanım Örneği ===")

    # Logger oluştur
    logger = logging.getLogger("RepositoryExample")
    logger.setLevel(logging.INFO)

    # CSV repository oluştur
    RepositoryFactory.create_csv_repository(logger)

    # JSON repository oluştur
    RepositoryFactory.create_json_repository(logger)

    # Dosya uzantısına göre repository oluştur
    RepositoryFactory.create_repository_by_extension("data.csv", logger)

    print("Repository pattern ile veri erişimi sağlandı")


def example_validation_strategies():
    """Validation strategies kullanım örneği"""
    print("\n=== Validation Strategies Kullanım Örneği ===")

    # Logger oluştur
    logger = logging.getLogger("ValidationExample")
    logger.setLevel(logging.INFO)

    # Farklı validasyon stratejileri oluştur
    ValidationStrategyFactory.create_strategy("bitcoin", logger)
    ValidationStrategyFactory.create_strategy("stock", logger)
    ValidationStrategyFactory.create_strategy("crypto", logger)
    ValidationStrategyFactory.create_strategy("forex", logger)

    # Kullanılabilir stratejileri listele
    available_strategies = ValidationStrategyFactory.get_available_strategies()
    print(f"Kullanılabilir stratejiler: {available_strategies}")

    print("Validation strategies oluşturuldu")


def example_complete_workflow():
    """Tam workflow örneği"""
    print("\n=== Tam Workflow Örneği ===")

    try:
        # 1. Factory ile DataLoader oluştur
        loader = DataLoaderFactory.create_bitcoin_loader(log_level="INFO")

        # 2. Veri dosyası yolu
        data_file = "data/raw/dataraw.csv"

        # 3. Veriyi yükle ve validasyon yap
        if Path(data_file).exists():
            data = loader.load_and_validate(data_file)
            print(f"Veri başarıyla yüklendi: {data.shape}")

            # 4. Metadata'yı kaydet
            metadata_file = "reports/data_metadata.json"
            loader.save_metadata(metadata_file)
            print(f"Metadata kaydedildi: {metadata_file}")

            # 5. Temel bilgileri al
            basic_info = loader.get_basic_info()
            print(f"Temel bilgiler: {list(basic_info.keys())}")

        else:
            print(f"Veri dosyası bulunamadı: {data_file}")

    except Exception as e:
        print(f"Workflow hatası: {str(e)}")


def example_advanced_usage():
    """Gelişmiş kullanım örneği"""
    print("\n=== Gelişmiş Kullanım Örneği ===")

    try:
        # 1. Özel validasyon stratejisi ile builder kullan
        logger = logging.getLogger("AdvancedExample")
        logger.setLevel(logging.INFO)

        # 2. Özel konfigürasyon ile DataLoader oluştur
        (
            DataLoaderBuilder()
            .with_log_level("INFO")
            .with_logger_name("AdvancedLoader")
            .build()
        )

        # 3. Repository ile veri erişimi
        csv_repo = RepositoryFactory.create_csv_repository(logger)

        # 4. Veri dosyası var mı kontrol et
        data_file = "data/raw/dataraw.csv"
        if csv_repo.exists(data_file):
            print(f"Veri dosyası mevcut: {data_file}")

            # 5. Metadata al
            metadata = csv_repo.get_metadata(data_file)
            if metadata:
                print(f"Dosya boyutu: {metadata['size']} bytes")
        else:
            print(f"Veri dosyası bulunamadı: {data_file}")

    except Exception as e:
        print(f"Gelişmiş kullanım hatası: {str(e)}")


if __name__ == "__main__":
    # Logging ayarla
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Örnekleri çalıştır
    example_factory_usage()
    example_builder_usage()
    example_repository_usage()
    example_validation_strategies()
    example_complete_workflow()
    example_advanced_usage()

    print("\n=== Tüm örnekler tamamlandı ===")
