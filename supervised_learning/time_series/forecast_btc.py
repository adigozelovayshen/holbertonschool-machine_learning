#!/usr/bin/env python3
""" Script to build, train, and validate an RNN model for BTC forecasting """
import tensorflow as tf
import numpy as np


def create_dataset(data, window_size=24, batch_size=64):
    """ Creates a tf.data.Dataset for sequence prediction """
    # Close qiyməti target olaraq (sütun indeksi 3) götürülür
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size, 3])  # Close price index
        
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.shuffle(buffer_size=1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def build_and_train_model():
    """ Trains the Keras LSTM model for time series forecasting """
    # İşlənmiş məlumatı yükləyirik
    loaded = np.load('preprocessed_btc.npz')
    data = loaded['data']
    
    # Train / Validation bölgüsü (80/20)
    split = int(len(data) * 0.8)
    train_data = data[:split]
    val_data = data[split:]
    
    train_ds = create_dataset(train_data)
    val_ds = create_dataset(val_data)
    
    # Keras Model Arxitekturası
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(24, data.shape[1])),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1)
    ])
    
    # MSE ilə compile edirik
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.summary()
    
    # Təlim
    model.fit(train_ds, validation_data=val_ds, epochs=10)
    model.save('btc_forecast_model.h5')
    print("Model training complete and saved as btc_forecast_model.h5")

if __name__ == '__main__':
    build_and_train_model()
