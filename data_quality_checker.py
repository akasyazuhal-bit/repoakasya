#!/usr/bin/env python3
"""
OHLCV Veri Kalitesi Kontrol Script'i
Plan.md'deki checklist'i otomatik olarak kontrol eder
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from typing import Dict, List, Tuple, Any

class DataQualityChecker:
    """OHLCV veri kalitesi kontrol sinifi"""
    
    def __init__(self, data_path: str):
        """
        Args:
            data_path (str): Veri dosyasi yolu
        """
        self.data_path = data_path
        self.df = None
        self.quality_report = {}
        self.load_data()
    
    def load_data(self):
        """Veriyi yukle"""
        try:
            self.df = pd.read_csv(self.data_path)
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
            print(f"[OK] Veri yuklendi: {len(self.df)} kayit")
        except Exception as e:
            print(f"[ERROR] Veri yukleme hatasi: {e}")
            return False
        return True
    
    def check_basic_info(self) -> Dict[str, Any]:
        """Temel veri bilgilerini kontrol et"""
        print("\n1. TEMEL VERI ANALIZI")
        print("=" * 50)
        
        info = {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'dtypes': self.df.dtypes.to_dict(),
            'memory_usage': self.df.memory_usage(deep=True).sum(),
            'date_range': {
                'start': self.df['timestamp'].min(),
                'end': self.df['timestamp'].max(),
                'days': (self.df['timestamp'].max() - self.df['timestamp'].min()).days
            }
        }
        
        print(f"Veri boyutu: {info['shape']}")
        print(f"Tarih araligi: {info['date_range']['start']} - {info['date_range']['end']}")
        print(f"Toplam gun: {info['date_range']['days']}")
        
        return info
    
    def check_duplicates(self) -> Dict[str, Any]:
        """Duplicate kayitlari kontrol et"""
        print("\n2. DUPLICATE KONTROLU")
        print("=" * 50)
        
        duplicate_count = self.df.duplicated().sum()
        duplicate_timestamps = self.df['timestamp'].duplicated().sum()
        
        result = {
            'total_duplicates': duplicate_count,
            'timestamp_duplicates': duplicate_timestamps,
            'is_clean': duplicate_count == 0 and duplicate_timestamps == 0
        }
        
        if result['is_clean']:
            print("[OK] Duplicate kayit bulunamadi")
        else:
            print(f"[WARNING] {duplicate_count} duplicate kayit bulundu")
            print(f"[WARNING] {duplicate_timestamps} duplicate timestamp bulundu")
        
        return result
    
    def check_missing_values(self) -> Dict[str, Any]:
        """Eksik degerleri kontrol et"""
        print("\n3. EKSIK DEGER ANALIZI")
        print("=" * 50)
        
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        
        result = {
            'missing_counts': missing.to_dict(),
            'missing_percentages': missing_pct.to_dict(),
            'total_missing': missing.sum(),
            'max_missing_pct': missing_pct.max(),
            'is_acceptable': missing_pct.max() < 5.0  # %5'ten az eksik deger
        }
        
        print("Eksik deger analizi:")
        for col in self.df.columns:
            count = missing[col]
            pct = missing_pct[col]
            status = "[OK]" if pct < 5 else "[WARNING]" if pct < 10 else "[ERROR]"
            print(f"  {status} {col}: {count} ({pct:.2f}%)")
        
        if result['is_acceptable']:
            print("[OK] Eksik deger orani kabul edilebilir")
        else:
            print("[ERROR] Eksik deger orani yuksek")
        
        return result
    
    def check_ohlc_consistency(self) -> Dict[str, Any]:
        """OHLC mantiksal tutarliligini kontrol et"""
        print("\n4. OHLC TUTARLILIK KONTROLU")
        print("=" * 50)
        
        # OHLC mantiksal kontrolleri
        high_ge_low = (self.df['high'] >= self.df['low']).all()
        high_ge_open = (self.df['high'] >= self.df['open']).all()
        high_ge_close = (self.df['high'] >= self.df['close']).all()
        low_le_open = (self.df['low'] <= self.df['open']).all()
        low_le_close = (self.df['low'] <= self.df['close']).all()
        
        # Volume pozitif kontrolu
        volume_positive = (self.df['volume'] >= 0).all()
        
        # Fiyat pozitif kontrolu
        price_positive = (
            (self.df['open'] > 0).all() and
            (self.df['high'] > 0).all() and
            (self.df['low'] > 0).all() and
            (self.df['close'] > 0).all()
        )
        
        result = {
            'high_ge_low': high_ge_low,
            'high_ge_open': high_ge_open,
            'high_ge_close': high_ge_close,
            'low_le_open': low_le_open,
            'low_le_close': low_le_close,
            'volume_positive': volume_positive,
            'price_positive': price_positive,
            'is_consistent': all([
                high_ge_low, high_ge_open, high_ge_close,
                low_le_open, low_le_close, volume_positive, price_positive
            ])
        }
        
        checks = [
            ("High >= Low", high_ge_low),
            ("High >= Open", high_ge_open),
            ("High >= Close", high_ge_close),
            ("Low <= Open", low_le_open),
            ("Low <= Close", low_le_close),
            ("Volume >= 0", volume_positive),
            ("Prices > 0", price_positive)
        ]
        
        for check_name, check_result in checks:
            status = "[OK]" if check_result else "[ERROR]"
            print(f"  {status} {check_name}")
        
        if result['is_consistent']:
            print("[OK] OHLC tutarliligi saglandi")
        else:
            print("[ERROR] OHLC tutarlilik sorunlari var")
        
        return result
    
    def check_timestamp_continuity(self) -> Dict[str, Any]:
        """Timestamp surekliligini kontrol et"""
        print("\n5. TİMESTAMP SUREKLILIK KONTROLU")
        print("=" * 50)
        
        # Timestamp siralamasi
        is_sorted = self.df['timestamp'].is_monotonic_increasing
        
        # Eksik gunleri bul
        expected_dates = pd.date_range(
            start=self.df['timestamp'].min(),
            end=self.df['timestamp'].max(),
            freq='D'
        )
        actual_dates = set(self.df['timestamp'].dt.date)
        expected_dates_set = set(expected_dates.date)
        missing_dates = expected_dates_set - actual_dates
        extra_dates = actual_dates - expected_dates_set
        
        result = {
            'is_sorted': is_sorted,
            'expected_days': len(expected_dates),
            'actual_days': len(actual_dates),
            'missing_dates': len(missing_dates),
            'extra_dates': len(extra_dates),
            'missing_dates_list': sorted(list(missing_dates)),
            'is_continuous': len(missing_dates) == 0 and len(extra_dates) == 0
        }
        
        print(f"Beklenen gun sayisi: {result['expected_days']}")
        print(f"Mevcut gun sayisi: {result['actual_days']}")
        print(f"Eksik gun sayisi: {result['missing_dates']}")
        print(f"Fazla gun sayisi: {result['extra_dates']}")
        
        if result['is_sorted']:
            print("[OK] Timestamp siralamasi dogru")
        else:
            print("[ERROR] Timestamp siralamasi bozuk")
        
        if result['is_continuous']:
            print("[OK] Timestamp surekliligi saglandi")
        else:
            print("[WARNING] Timestamp sureklilik sorunlari var")
            if missing_dates:
                print(f"   Eksik gunler: {sorted(list(missing_dates))[:10]}...")
        
        return result
    
    def detect_outliers(self) -> Dict[str, Any]:
        """Outlier'lari tespit et"""
        print("\n6. OUTLIER TESPITI")
        print("=" * 50)
        
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        outlier_results = {}
        
        for col in numeric_cols:
            # Z-Score yontemi
            z_scores = np.abs((self.df[col] - self.df[col].mean()) / self.df[col].std())
            z_outliers = (z_scores > 3).sum()
            
            # IQR yontemi
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            iqr_outliers = ((self.df[col] < lower_bound) | (self.df[col] > upper_bound)).sum()
            
            outlier_results[col] = {
                'z_score_outliers': z_outliers,
                'iqr_outliers': iqr_outliers,
                'z_score_pct': (z_outliers / len(self.df)) * 100,
                'iqr_pct': (iqr_outliers / len(self.df)) * 100
            }
            
            print(f"{col.upper()}:")
            print(f"   Z-Score outliers: {z_outliers} ({outlier_results[col]['z_score_pct']:.2f}%)")
            print(f"   IQR outliers: {iqr_outliers} ({outlier_results[col]['iqr_pct']:.2f}%)")
        
        # Genel outlier degerlendirmesi
        max_outlier_pct = max([result['iqr_pct'] for result in outlier_results.values()])
        is_acceptable = max_outlier_pct < 5.0  # %5'ten az outlier
        
        result = {
            'outlier_details': outlier_results,
            'max_outlier_pct': max_outlier_pct,
            'is_acceptable': is_acceptable
        }
        
        if result['is_acceptable']:
            print("[OK] Outlier orani kabul edilebilir")
        else:
            print("[WARNING] Outlier orani yuksek")
        
        return result
    
    def calculate_basic_statistics(self) -> Dict[str, Any]:
        """Temel istatistikleri hesapla"""
        print("\n7. TEMEL ISTATISTIKLER")
        print("=" * 50)
        
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        stats = {}
        
        for col in numeric_cols:
            stats[col] = {
                'mean': self.df[col].mean(),
                'median': self.df[col].median(),
                'std': self.df[col].std(),
                'min': self.df[col].min(),
                'max': self.df[col].max(),
                'skewness': self.df[col].skew(),
                'kurtosis': self.df[col].kurtosis()
            }
            
            print(f"{col.upper()}:")
            print(f"   Ortalama: {stats[col]['mean']:.2f}")
            print(f"   Medyan: {stats[col]['median']:.2f}")
            print(f"   Std: {stats[col]['std']:.2f}")
            print(f"   Min-Max: {stats[col]['min']:.2f} - {stats[col]['max']:.2f}")
        
        return stats
    
    def generate_quality_report(self) -> Dict[str, Any]:
        """Kapsamli kalite raporu olustur"""
        print("\n" + "="*60)
        print("KAPSAMLI VERI KALITE RAPORU")
        print("="*60)
        
        # Tum kontrolleri calistir
        basic_info = self.check_basic_info()
        duplicates = self.check_duplicates()
        missing = self.check_missing_values()
        ohlc = self.check_ohlc_consistency()
        timestamp = self.check_timestamp_continuity()
        outliers = self.detect_outliers()
        statistics = self.calculate_basic_statistics()
        
        # Genel kalite skoru hesapla
        quality_checks = [
            duplicates['is_clean'],
            missing['is_acceptable'],
            ohlc['is_consistent'],
            timestamp['is_sorted'],
            outliers['is_acceptable']
        ]
        
        quality_score = sum(quality_checks) / len(quality_checks) * 100
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'data_path': self.data_path,
            'basic_info': basic_info,
            'duplicates': duplicates,
            'missing_values': missing,
            'ohlc_consistency': ohlc,
            'timestamp_continuity': timestamp,
            'outliers': outliers,
            'statistics': statistics,
            'quality_score': quality_score,
            'quality_checks': quality_checks,
            'overall_status': 'PASS' if quality_score >= 80 else 'FAIL'
        }
        
        print(f"\nGENEL KALITE SKORU: {quality_score:.1f}/100")
        print(f"DURUM: {report['overall_status']}")
        
        if quality_score >= 80:
            print("[OK] Veri kalitesi yuksek - FE surecine hazir")
        elif quality_score >= 60:
            print("[WARNING] Veri kalitesi orta - Bazi duzeltmeler gerekli")
        else:
            print("[ERROR] Veri kalitesi dusuk - Kapsamli temizleme gerekli")
        
        return report
    
    def save_report(self, report: Dict[str, Any], output_path: str = "reports/quality_report.json"):
        """Raporu dosyaya kaydet"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\nRapor kaydedildi: {output_path}")

def main():
    """Ana fonksiyon"""
    print("OHLCV VERI KALITESI KONTROL SISTEMI")
    print("="*60)
    
    # Veri yolu
    data_path = "data/raw/dataraw.csv"
    
    # Kontrol sistemi baslat
    checker = DataQualityChecker(data_path)
    
    if checker.df is not None:
        # Kalite raporu olustur
        report = checker.generate_quality_report()
        
        # Raporu kaydet
        checker.save_report(report)
        
        print("\n[OK] Veri kalitesi kontrolu tamamlandi!")
    else:
        print("[ERROR] Veri yuklenemedi!")

if __name__ == "__main__":
    main()