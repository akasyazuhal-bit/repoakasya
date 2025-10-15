"""
Bitcoin Data Preprocessing Configuration

Bu modül tüm preprocessing işlemleri için gerekli konfigürasyon
sabitlerini ve threshold değerlerini içerir.
"""

# Outlier Detection Configuration
OUTLIER_CONFIG = {
    "z_score_threshold": 3.0,  # Z-score için threshold
    "iqr_multiplier": 1.5,  # IQR yöntemi için çarpan
    "mad_threshold": 3.5,  # Modified Z-score için threshold
    "isolation_forest_contamination": 0.05,  # Isolation Forest contamination oranı
    "lof_neighbors": 20,  # LOF için komşu sayısı
    "lof_contamination": 0.05,  # LOF contamination oranı
    "volume_spike_threshold": 3.0,  # Volume spike tespiti için threshold
    "price_jump_threshold": 0.20,  # Fiyat sıçrama tespiti için threshold (%20)
    "flash_crash_threshold": -0.10,  # Flash crash tespiti için threshold (%10)
}

# Data Validation Configuration
VALIDATION_CONFIG = {
    "max_missing_pct": 5.0,  # Maksimum eksik değer yüzdesi
    "price_change_limit_pct": 20.0,  # Maksimum günlük fiyat değişimi
    "volume_spike_threshold": 3.0,  # Volume spike threshold
    "ohlc_tolerance": 0.001,  # OHLC tutarlılık toleransı
    "min_volume": 1000.0,  # Minimum volume değeri
    "max_volume_multiplier": 10.0,  # Maksimum volume çarpanı (ortalama * çarpan)
}

# Statistical Analysis Configuration
STATISTICAL_CONFIG = {
    "normality_alpha": 0.05,  # Normallik testleri için alpha
    "stationarity_alpha": 0.05,  # Stationarity testleri için alpha
    "correlation_threshold": 0.7,  # Yüksek korelasyon threshold'u
    "autocorr_lags": 40,  # Autocorrelation için lag sayısı
    "adf_max_lags": 12,  # ADF testi için maksimum lag
    "kpss_regression": "c",  # KPSS testi regression tipi ('c', 'ct')
}

# Data Transformation Configuration
TRANSFORMATION_CONFIG = {
    "scaling_methods": [
        "minmax",
        "standard",
        "robust",
    ],  # Kullanılabilir scaling yöntemleri
    "transformation_methods": [
        "log",
        "boxcox",
        "yeojohnson",
    ],  # Transformation yöntemleri
    "differencing_orders": [1, 2],  # Differencing dereceleri
    "boxcox_lambda_range": (-2, 2),  # Box-Cox lambda aralığı
    "yeojohnson_lambda_range": (-2, 2),  # Yeo-Johnson lambda aralığı
}

# Data Cleaning Configuration
CLEANING_CONFIG = {
    "outlier_handling": ["remove", "cap", "interpolate"],  # Outlier handling yöntemleri
    "missing_value_methods": [
        "forward_fill",
        "backward_fill",
        "interpolate",
        "mean",
    ],  # Missing value yöntemleri
    "smoothing_methods": ["moving_average", "exponential"],  # Smoothing yöntemleri
    "moving_average_window": 5,  # Moving average pencere boyutu
    "exponential_alpha": 0.3,  # Exponential smoothing alpha
    "interpolation_method": "linear",  # Interpolation yöntemi
}

# Pipeline Configuration
PIPELINE_CONFIG = {
    "log_level": "INFO",  # Logging seviyesi
    "save_intermediate": False,  # Ara sonuçları kaydet
    "parallel_processing": False,  # Paralel işleme
    "memory_efficient": True,  # Bellek verimli mod
    "progress_bar": True,  # Progress bar göster
    "validation_strict": True,  # Strict validation modu
}

# File Paths Configuration
PATH_CONFIG = {
    "input_columns": ["timestamp", "open", "high", "low", "close", "volume"],
    "required_columns": ["timestamp", "open", "high", "low", "close", "volume"],
    "timestamp_format": "%Y-%m-%d",
    "output_encoding": "utf-8",
    "csv_separator": ",",
    "decimal_separator": ".",
}

# Error Messages
ERROR_MESSAGES = {
    "missing_columns": "Gerekli sütunlar eksik: {missing_columns}",
    "invalid_data_types": "Geçersiz veri tipleri: {invalid_types}",
    "timestamp_parsing_error": "Timestamp parsing hatası: {error}",
    "ohlc_consistency_error": "OHLC tutarlılık hatası: {error}",
    "outlier_detection_error": "Outlier tespit hatası: {error}",
    "statistical_test_error": "İstatistiksel test hatası: {error}",
    "transformation_error": "Dönüşüm hatası: {error}",
    "cleaning_error": "Temizleme hatası: {error}",
}

# Success Messages
SUCCESS_MESSAGES = {
    "data_loaded": "Veri başarıyla yüklendi: {shape}",
    "validation_passed": "Validasyon başarılı",
    "outliers_detected": "Outlier tespit edildi: {count} adet",
    "statistical_analysis_complete": "İstatistiksel analiz tamamlandı",
    "transformation_complete": "Dönüşüm tamamlandı",
    "cleaning_complete": "Temizleme tamamlandı",
    "pipeline_complete": "Pipeline başarıyla tamamlandı",
}

# Default Values
DEFAULT_VALUES = {
    "fill_method": "forward_fill",
    "outlier_method": "isolation_forest",
    "scaling_method": "robust",
    "transformation_method": "log",
    "cleaning_method": "cap",
    "smoothing_method": "moving_average",
}
