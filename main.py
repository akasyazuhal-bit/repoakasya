"""
Bitcoin Data Preprocessing Pipeline - Ana Executor

Bu script Bitcoin fiyat verilerini makine öğrenmesi için hazırlamak üzere
geliştirilmiş kapsamlı preprocessing pipeline'ını çalıştırır.
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
import argparse
import json

# Proje root dizinini Python path'e ekle
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.preprocessing.pipeline import DataPipeline  # noqa: E402
from src.preprocessing.config import (  # noqa: E402
    OUTLIER_CONFIG,
    VALIDATION_CONFIG,
    STATISTICAL_CONFIG,
)


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """
    Logging sistemini kurar.

    Args:
        log_level (str): Logging seviyesi

    Returns:
        logging.Logger: Logger objesi
    """
    # Ana logger'ı kur
    logger = logging.getLogger("BitcoinDataPreprocessing")
    logger.setLevel(getattr(logging, log_level.upper()))

    # Handler'ları temizle
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    log_file = (
        project_root
        / "logs"
        / f'preprocessing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    )
    log_file.parent.mkdir(exist_ok=True)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def create_default_config() -> dict:
    """
    Varsayılan pipeline konfigürasyonunu oluşturur.

    Returns:
        dict: Varsayılan konfigürasyon
    """
    return {
        "log_level": "INFO",
        "outlier_handling": {
            "method": "cap",
            "z_score_threshold": OUTLIER_CONFIG["z_score_threshold"],
            "iqr_multiplier": OUTLIER_CONFIG["iqr_multiplier"],
        },
        "missing_value_imputation": {"method": "forward_fill"},
        "data_smoothing": {"method": "moving_average", "window": 5},
        "scaling": {"method": "robust"},
        "transformation": {"method": "log"},
        "differencing": {"order": 1},
        "technical_indicators": {"windows": [5, 10, 20]},
        "lag_features": {"lags": [1, 2, 3, 5, 10]},
        "validation": {
            "max_missing_pct": VALIDATION_CONFIG["max_missing_pct"],
            "price_change_limit_pct": VALIDATION_CONFIG["price_change_limit_pct"],
        },
        "statistical_analysis": {
            "normality_alpha": STATISTICAL_CONFIG["normality_alpha"],
            "stationarity_alpha": STATISTICAL_CONFIG["stationarity_alpha"],
        },
    }


def load_config(config_path: str) -> dict:
    """
    Konfigürasyon dosyasını yükler.

    Args:
        config_path (str): Konfigürasyon dosya yolu

    Returns:
        dict: Konfigürasyon
    """
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        return config
    except Exception as e:
        print(f"Konfigürasyon yükleme hatası: {str(e)}")
        print("Varsayılan konfigürasyon kullanılacak")
        return create_default_config()


def save_config(config: dict, config_path: str) -> None:
    """
    Konfigürasyonu dosyaya kaydeder.

    Args:
        config (dict): Konfigürasyon
        config_path (str): Konfigürasyon dosya yolu
    """
    try:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"Konfigürasyon kaydedildi: {config_path}")
    except Exception as e:
        print(f"Konfigürasyon kaydetme hatası: {str(e)}")


def run_pipeline(
    input_path: str, output_path: str, config: dict, logger: logging.Logger
) -> bool:
    """
    Pipeline'ı çalıştırır.

    Args:
        input_path (str): Giriş dosya yolu
        output_path (str): Çıkış dosya yolu
        config (dict): Pipeline konfigürasyonu
        logger (logging.Logger): Logger objesi

    Returns:
        bool: Pipeline başarılı mı
    """
    try:
        logger.info("=" * 60)
        logger.info("BITCOIN DATA PREPROCESSING PIPELINE")
        logger.info("=" * 60)
        logger.info(f"Giriş dosyası: {input_path}")
        logger.info(f"Çıkış dosyası: {output_path}")
        logger.info(
            f"Konfigürasyon: {json.dumps(config, indent=2, ensure_ascii=False)}"
        )
        logger.info("=" * 60)

        # Pipeline'ı oluştur
        pipeline = DataPipeline(
            input_path=input_path, output_path=output_path, config=config
        )

        # Pipeline validasyonu
        if not pipeline.validate_pipeline():
            logger.error("Pipeline validasyonu başarısız")
            return False

        # Bileşen durumlarını kontrol et
        component_status = pipeline.get_component_status()
        logger.info(f"Bileşen durumları: {component_status}")

        # Pipeline'ı çalıştır
        logger.info("Pipeline başlatılıyor...")
        start_time = datetime.now()

        result_data = pipeline.run()

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        logger.info(f"Pipeline başarıyla tamamlandı: {duration:.2f} saniye")
        logger.info(f"Sonuç veri boyutu: {result_data.shape}")

        # Pipeline raporunu kaydet
        report_path = (
            Path(output_path).parent / f"{Path(output_path).stem}_pipeline_report.json"
        )
        pipeline.save_pipeline_report(str(report_path))
        logger.info(f"Pipeline raporu kaydedildi: {report_path}")

        return True

    except Exception as e:
        logger.error(f"Pipeline çalıştırma hatası: {str(e)}")
        return False


def main():
    """Ana fonksiyon."""
    parser = argparse.ArgumentParser(
        description="Bitcoin Data Preprocessing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnek kullanım:
  python main.py -i data/raw/dataraw.csv -o data/processed/cleaned_data.csv
  python main.py -i data/raw/dataraw.csv -o data/processed/cleaned_data.csv -c config.json
  python main.py -i data/raw/dataraw.csv -o data/processed/cleaned_data.csv --create-config
        """,
    )

    parser.add_argument("-i", "--input", required=True, help="Giriş CSV dosya yolu")

    parser.add_argument("-o", "--output", required=True, help="Çıkış CSV dosya yolu")

    parser.add_argument("-c", "--config", help="Konfigürasyon JSON dosya yolu")

    parser.add_argument(
        "--create-config",
        action="store_true",
        help="Varsayılan konfigürasyon dosyası oluştur",
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging seviyesi",
    )

    args = parser.parse_args()

    # Logging'i kur
    logger = setup_logging(args.log_level)

    try:
        # Konfigürasyon dosyası oluştur
        if args.create_config:
            config = create_default_config()
            config_path = "config.json"
            save_config(config, config_path)
            logger.info(f"Varsayılan konfigürasyon oluşturuldu: {config_path}")
            return

        # Konfigürasyonu yükle
        if args.config:
            config = load_config(args.config)
        else:
            config = create_default_config()

        # Logging seviyesini güncelle
        config["log_level"] = args.log_level

        # Pipeline'ı çalıştır
        success = run_pipeline(
            input_path=args.input, output_path=args.output, config=config, logger=logger
        )

        if success:
            logger.info("Pipeline başarıyla tamamlandı!")
            sys.exit(0)
        else:
            logger.error("Pipeline başarısız!")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("Pipeline kullanıcı tarafından durduruldu")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Beklenmeyen hata: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
