#!/usr/bin/env python3
"""
Binance API'den OHLCV verisi çeken script
2024 Ocak'tan itibaren 6 aylık 1 günlük zaman diliminde veri çeker
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import os

def fetch_binance_klines(symbol, interval, start_time, end_time):
    """
    Binance API'den kline verisi çeker
    
    Args:
        symbol (str): Trading pair (örn: 'BTCUSDT')
        interval (str): Zaman dilimi (örn: '1d')
        start_time (int): Başlangıç zamanı (millisecond)
        end_time (int): Bitiş zamanı (millisecond)
    
    Returns:
        list: Kline verileri
    """
    base_url = "https://api.binance.com/api/v3/klines"
    
    params = {
        'symbol': symbol,
        'interval': interval,
        'startTime': start_time,
        'endTime': end_time,
        'limit': 1000  # Maksimum 1000 kayıt
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"API isteği başarısız: {e}")
        return None

def convert_klines_to_dataframe(klines):
    """
    Kline verilerini pandas DataFrame'e dönüştürür
    
    Args:
        klines (list): Binance kline verileri
    
    Returns:
        pd.DataFrame: OHLCV verileri
    """
    if not klines:
        return pd.DataFrame()
    
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    
    # Sadece OHLCV sütunlarını seç
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    
    # Veri tiplerini dönüştür
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['open'] = df['open'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['close'] = df['close'].astype(float)
    df['volume'] = df['volume'].astype(float)
    
    return df

def fetch_historical_data(symbol='BTCUSDT', interval='1d', months=6):
    """
    Belirtilen süre için tarihsel veri çeker
    
    Args:
        symbol (str): Trading pair
        interval (str): Zaman dilimi
        months (int): Kaç aylık veri çekileceği
    """
    # 2024 Ocak 1'den başla
    start_date = datetime(2024, 1, 1)
    end_date = start_date + timedelta(days=months * 30)  # Yaklaşık 6 ay
    
    print(f"Veri çekiliyor: {symbol} - {interval}")
    print(f"Başlangıç: {start_date.strftime('%Y-%m-%d')}")
    print(f"Bitiş: {end_date.strftime('%Y-%m-%d')}")
    
    # Tarihleri millisecond'a çevir
    start_timestamp = int(start_date.timestamp() * 1000)
    end_timestamp = int(end_date.timestamp() * 1000)
    
    all_data = []
    current_start = start_timestamp
    
    while current_start < end_timestamp:
        current_end = min(current_start + (1000 * 24 * 60 * 60 * 1000), end_timestamp)  # 1000 gün
        
        print(f"Veri çekiliyor: {datetime.fromtimestamp(current_start/1000).strftime('%Y-%m-%d')} - {datetime.fromtimestamp(current_end/1000).strftime('%Y-%m-%d')}")
        
        klines = fetch_binance_klines(symbol, interval, current_start, current_end)
        
        if klines:
            all_data.extend(klines)
            print(f"  {len(klines)} kayıt alındı")
        else:
            print("  Veri alınamadı")
        
        # API rate limit için bekle
        time.sleep(0.1)
        
        # Sonraki batch için başlangıç zamanını güncelle
        if klines:
            current_start = klines[-1][6] + 1  # Son kline'ın close_time + 1
        else:
            current_start = current_end
    
    if all_data:
        df = convert_klines_to_dataframe(all_data)
        return df
    else:
        return pd.DataFrame()

def save_to_csv(df, symbol, interval):
    """
    DataFrame'i CSV dosyasına kaydeder
    
    Args:
        df (pd.DataFrame): Kaydedilecek veri
        symbol (str): Trading pair
        interval (str): Zaman dilimi
    """
    # data/raw klasörünü oluştur
    os.makedirs('data/raw', exist_ok=True)
    
    # Dosya adı oluştur
    filename = f"data/raw/{symbol}_{interval}_2024_6months.csv"
    
    # CSV'ye kaydet
    df.to_csv(filename, index=False)
    print(f"Veri kaydedildi: {filename}")
    print(f"Toplam kayıt sayısı: {len(df)}")

def main():
    """Ana fonksiyon"""
    print("Binance OHLCV Veri Çekici")
    print("=" * 40)
    
    # Veri çek
    df = fetch_historical_data(symbol='BTCUSDT', interval='1d', months=6)
    
    if not df.empty:
        # CSV'ye kaydet
        save_to_csv(df, 'BTCUSDT', '1d')
        
        # İlk ve son birkaç kaydı göster
        print("\nİlk 5 kayıt:")
        print(df.head())
        print("\nSon 5 kayıt:")
        print(df.tail())
        
        print(f"\nVeri aralığı: {df['timestamp'].min()} - {df['timestamp'].max()}")
    else:
        print("Veri çekilemedi!")

if __name__ == "__main__":
    main()
