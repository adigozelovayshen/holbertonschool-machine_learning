#!/usr/bin/env python3
""" Preprocessing script for BTC raw time series data """
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def preprocess_btc_data(file_path):
    """
    Preprocesses raw BTC dataset:
    - Converts Timestamp to datetime
    - Resamples minute-level data to 1-hour windows
    - Drops unnecessary columns and handles missing values
    - Scales features using MinMaxScaler
    """
    # Datadan istifadə olunan sütunları yükləyirik
    df = pd.read_csv(file_path)
    
    # Unix timestamp-i datetime formatına keçiririk
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
    df = df.set_index('Timestamp')
    
    # 2017-ci ildən sonrakı məlumatları götürürük (daha stabil dövr)
    df = df.loc['2017-01-01':]
    
    # Dəqiqəlik datanı 1 saatlıq (1H) intervala çeviririk
    df_resampled = df.resample('1h').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume_(BTC)': 'sum',
        'Volume_(Currency)': 'sum',
        'Weighted_Price': 'mean'
    })
    
    # Eksik (NaN) dəyərləri əvvəlki dəyərlə doldururuq (forward fill)
    df_resampled = df_resampled.ffill().bfill()
    
    # MinMaxScaler ilə məlumatları [0, 1] aralığına gətiririk
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df_resampled)
    
    # Qramatik olaraq hazır datanı saxlayırıq
    np.savez('preprocessed_btc.npz', data=scaled_data)
    print("Data preprocessing completed successfully! Saved as preprocessed_btc.npz")
    return scaled_data, scaler

if __name__ == '__main__':
    # Nümunə olaraq işə salmaq üçün
    preprocess_btc_data('coinbaseUSD_1-min_data_2012-01-01_to_2021-03-31.csv')
