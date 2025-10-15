# OHLCV Veri Temizleme ve FE Hazırlık Planı

## 📋 Genel Bakış
Bu plan, Binance API'den çekilen BTCUSDT OHLCV verisini Feature Engineering sürecine hazırlamak için kapsamlı bir temizleme stratejisi sunar.

## 🎯 Amaç
- Veri kalitesini artırmak
- FE sürecine uygun veri yapısı oluşturmak
- Outlier'ları tespit etmek ve yönetmek
- Eksik değerleri analiz etmek ve doldurmak
- Veri dönüşümlerini optimize etmek

## 📊 Veri Analizi (Mevcut Durum)
- **Dosya:** `data/raw/dataraw.csv`
- **Periyot:** 2024 Ocak - Haziran (6 ay)
- **Kayıt Sayısı:** 181 günlük veri
- **Sütunlar:** timestamp, open, high, low, close, volume
- **Veri Tipi:** Günlük OHLCV

## 🔍 1. VERİ KALİTESİ KONTROLÜ

### 1.1 Temel Veri Analizi
- [ ] Veri boyutları ve tiplerini kontrol et
- [ ] Duplicate kayıtları tespit et
- [ ] Timestamp sıralamasını kontrol et
- [ ] Veri aralığını analiz et

### 1.2 Veri Tutarlılığı Kontrolleri
- [ ] OHLC mantıksal tutarlılığı (High >= Low, High >= Open, High >= Close, Low <= Open, Low <= Close)
- [ ] Volume değerlerinin pozitif olup olmadığını kontrol et
- [ ] Timestamp sürekliliğini kontrol et (eksik günler)
- [ ] Weekend/holiday verilerini analiz et

### 1.3 Veri Tipi Kontrolleri
- [ ] Numeric sütunların doğru tip kontrolü
- [ ] Timestamp format kontrolü
- [ ] Decimal precision kontrolü

## 🔍 2. EKSİK DEĞER ANALİZİ

### 2.1 Eksik Değer Tespiti
- [ ] Her sütun için eksik değer sayısı
- [ ] Eksik değer pattern analizi
- [ ] Eksik değerlerin zaman serisi dağılımı

### 2.2 Eksik Değer Stratejileri
- [ ] **Forward Fill:** Son geçerli değerle doldurma
- [ ] **Backward Fill:** Sonraki geçerli değerle doldurma
- [ ] **Interpolation:** Lineer interpolasyon
- [ ] **Market Close:** Piyasa kapanış değeri ile doldurma
- [ ] **Drop:** Eksik kayıtları silme

## 🔍 3. OUTLIER TESPİTİ VE YÖNETİMİ

### 3.1 İstatistiksel Outlier Tespiti
- [ ] **Z-Score:** |z| > 3 değerleri
- [ ] **IQR Method:** Q1 - 1.5*IQR ve Q3 + 1.5*IQR dışındaki değerler
- [ ] **Modified Z-Score:** Median Absolute Deviation (MAD) kullanarak

### 3.2 Domain-Specific Outlier Tespiti
- [ ] **Price Outliers:** Aşırı fiyat hareketleri
- [ ] **Volume Outliers:** Anormal hacim artışları
- [ ] **Volatility Outliers:** Aşırı volatilite dönemleri
- [ ] **Market Events:** Önemli piyasa olayları ile karşılaştırma

### 3.3 Outlier Yönetim Stratejileri
- [ ] **Capping:** Maksimum/minimum değerlerle sınırlama
- [ ] **Winsorizing:** %1-5 extreme değerleri değiştirme
- [ ] **Transformation:** Log, sqrt dönüşümleri
- [ ] **Removal:** Aşırı outlier'ları silme
- [ ] **Flagging:** Outlier'ları işaretleme ama koruma

## 🔍 4. VERİ DÖNÜŞÜMLERİ

### 4.1 Temel Dönüşümler
- [ ] **Returns:** Günlük getiri hesaplama (Close/Close_prev - 1)
- [ ] **Log Returns:** Log(Close/Close_prev)
- [ ] **Price Ratios:** High/Low, Close/Open oranları
- [ ] **Volume Ratios:** Volume/Volume_avg oranları

### 4.2 Teknik İndikatörler
- [ ] **Moving Averages:** 5, 10, 20, 50 günlük ortalamalar
- [ ] **Volatility:** Rolling standard deviation
- [ ] **RSI:** Relative Strength Index
- [ ] **MACD:** Moving Average Convergence Divergence
- [ ] **Bollinger Bands:** Upper, middle, lower bands

### 4.3 Zaman Serisi Özellikleri
- [ ] **Lag Features:** 1, 2, 3, 5, 10 günlük gecikmeler
- [ ] **Rolling Statistics:** Min, max, mean, std (5, 10, 20 gün)
- [ ] **Seasonality:** Haftalık, aylık pattern'ler
- [ ] **Trend:** Linear trend slope

## 🔍 5. FE HAZIRLIK ADIMLARI

### 5.1 Hedef Değişken Tanımlama
- [ ] **Binary Classification:** Fiyat artış/azalış (1/0)
- [ ] **Multi-class:** Büyük artış, küçük artış, değişim yok, küçük azalış, büyük azalış
- [ ] **Regression:** Gelecek fiyat tahmini
- [ ] **Volatility Prediction:** Gelecek volatilite tahmini

### 5.2 Feature Engineering
- [ ] **Price Features:** OHLC kombinasyonları, price ratios
- [ ] **Volume Features:** Volume patterns, volume-price relationships
- [ ] **Technical Features:** Momentum, trend, volatility indicators
- [ ] **Time Features:** Day of week, month, quarter, seasonality
- [ ] **Market Features:** Market regime, volatility regime

### 5.3 Feature Selection
- [ ] **Correlation Analysis:** Yüksek korelasyonlu feature'ları tespit et
- [ ] **Variance Analysis:** Düşük varyanslı feature'ları tespit et
- [ ] **Feature Importance:** Model-based feature importance
- [ ] **Mutual Information:** Feature-target ilişkisi

## 🔍 6. VERİ DOĞRULAMA VE TEST

### 6.1 Veri Kalitesi Metrikleri
- [ ] **Completeness:** Eksik değer oranı < %5
- [ ] **Consistency:** OHLC mantıksal tutarlılığı %100
- [ ] **Accuracy:** Outlier oranı < %2
- [ ] **Timeliness:** Güncel veri güncelliği

### 6.2 İstatistiksel Testler
- [ ] **Normality Tests:** Shapiro-Wilk, Kolmogorov-Smirnov
- [ ] **Stationarity Tests:** ADF, KPSS
- [ ] **Autocorrelation:** Ljung-Box test
- [ ] **Heteroscedasticity:** Breusch-Pagan test

### 6.3 Cross-Validation
- [ ] **Time Series Split:** Chronological train/test split
- [ ] **Walk-Forward Analysis:** Rolling window validation
- [ ] **Out-of-Sample Testing:** Son 30 gün test seti

## 🔍 7. ÇIKTI DOSYALARI

### 7.1 Temizlenmiş Veri
- [ ] `data/processed/cleaned_data.csv`
- [ ] `data/processed/feature_engineered_data.csv`
- [ ] `data/processed/train_data.csv`
- [ ] `data/processed/test_data.csv`

### 7.2 Raporlar
- [ ] `reports/data_quality_report.html`
- [ ] `reports/outlier_analysis_report.html`
- [ ] `reports/feature_analysis_report.html`
- [ ] `reports/validation_report.html`

### 7.3 Metadata
- [ ] `data/processed/data_dictionary.json`
- [ ] `data/processed/transformation_log.json`
- [ ] `data/processed/quality_metrics.json`

## 🔍 8. KALİTE KONTROL CHECKLİSTESİ

### 8.1 Veri Kalitesi
- [ ] Tüm sütunlar doğru veri tipinde
- [ ] Eksik değer oranı kabul edilebilir seviyede
- [ ] Outlier'lar tespit edildi ve yönetildi
- [ ] OHLC mantıksal tutarlılığı sağlandı

### 8.2 Feature Engineering
- [ ] Hedef değişken doğru tanımlandı
- [ ] Feature'lar anlamlı ve yorumlanabilir
- [ ] Feature correlation'ları kontrol edildi
- [ ] Feature scaling uygulandı

### 8.3 Model Hazırlığı
- [ ] Train/test split doğru yapıldı
- [ ] Time series özellikleri korundu
- [ ] Data leakage önlendi
- [ ] Cross-validation stratejisi belirlendi

## 🔍 9. RİSK YÖNETİMİ

### 9.1 Veri Kaybı Riski
- [ ] Backup veri setleri oluşturuldu
- [ ] Transformation log'ları tutuldu
- [ ] Rollback stratejisi hazırlandı

### 9.2 Model Performans Riski
- [ ] Overfitting kontrolü
- [ ] Data leakage kontrolü
- [ ] Out-of-sample validation

### 9.3 İş Sürekliliği
- [ ] Otomatik pipeline oluşturuldu
- [ ] Monitoring sistemleri kuruldu
- [ ] Alert mekanizmaları hazırlandı

## 🔍 10. BAŞARI KRİTERLERİ

### 10.1 Veri Kalitesi
- ✅ Eksik değer oranı < %1
- ✅ Outlier oranı < %2
- ✅ OHLC tutarlılığı %100
- ✅ Timestamp sürekliliği %100

### 10.2 Feature Quality
- ✅ Feature correlation < 0.95
- ✅ Feature variance > 0.01
- ✅ Feature-target correlation > 0.1
- ✅ Feature stability > 0.8

### 10.3 Model Readiness
- ✅ Train/test split 80/20
- ✅ Time series order korundu
- ✅ No data leakage
- ✅ Cross-validation ready

---

## 📝 Notlar
- Bu plan iterative olarak uygulanacak
- Her adım sonrası kalite kontrolü yapılacak
- Gerekirse plan güncellenecek
- Tüm adımlar dokümante edilecek

## 🚀 Sonraki Adımlar
1. Veri kalitesi kontrolü başlat
2. Eksik değer analizi yap
3. Outlier tespiti uygula
4. Feature engineering başlat
5. Model hazırlığı tamamla
