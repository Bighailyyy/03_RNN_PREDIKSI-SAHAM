import json
import os
import random

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import yfinance as yf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Input, SimpleRNN
from tensorflow.keras.models import Sequential

# ---------------- Konfigurasi ----------------
TICKER = "BBCA.JK"
START_DATE = "2015-01-01"
END_DATE = "2025-12-31"
WINDOW = 60          # jumlah hari historis sebagai input
TRAIN_RATIO = 0.8    # 80% latih, 20% uji (kronologis)
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001
SEED = 42
KODE = TICKER.split(".")[0]          # "BBRI" atau "BBCA"
BASE = f"hasil-{KODE}"               # "hasil-BBRI" atau "hasil-BBCA"
OUT_DIR = os.path.join(BASE, "outputs")

os.makedirs(OUT_DIR, exist_ok=True)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def load_data():
    """1. Data Collecting: unduh data dari Yahoo Finance."""
    df = yf.download(TICKER, start=START_DATE, end=END_DATE,
                     auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.sort_index()
    df.to_csv(os.path.join(OUT_DIR, f"data_{KODE.lower()}.csv"))
    return df


def run_eda(df):
    """2. Exploratory Data Analysis."""
    print("=== EDA ===")
    print("Jumlah baris   :", len(df))
    print("Rentang tanggal:", df.index.min().date(), "s.d.", df.index.max().date())
    print("Nilai hilang   :\n", df.isnull().sum())
    print(df["Close"].describe())

    plt.figure(figsize=(12, 5))
    plt.plot(df.index, df["Close"])
    plt.title(f"Harga Penutupan {TICKER}")
    plt.xlabel("Tanggal")
    plt.ylabel("Harga (Rp)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "eda_close.png"), dpi=150)
    plt.close()


def make_windows(values, window):
    X, y = [], []
    for i in range(window, len(values)):
        X.append(values[i - window:i, 0])
        y.append(values[i, 0])
    X = np.array(X).reshape(-1, window, 1)
    return X, np.array(y)


def build_model():
    """4. Arsitektur model SimpleRNN."""
    model = Sequential([
        Input(shape=(WINDOW, 1)),
        SimpleRNN(64, return_sequences=True),
        Dropout(0.2),
        SimpleRNN(32),
        Dropout(0.2),
        Dense(1),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss="mse")
    return model


def main():
    df = load_data()
    run_eda(df)

    # 3. Preprocessing (berbasis return harian)
    close = df[["Close"]].ffill().dropna()
    prices = close["Close"].values.astype("float32")
    ret = (prices[1:] / prices[:-1] - 1).reshape(-1, 1)   # return harian
    train_size = int(len(ret) * TRAIN_RATIO)

    scaler = StandardScaler()
    scaler.fit(ret[:train_size])              # fit hanya pada data latih
    scaled = scaler.transform(ret)

    train_scaled = scaled[:train_size]
    test_scaled = scaled[train_size - WINDOW:]  # sertakan 60 hari terakhir data latih

    X_train, y_train = make_windows(train_scaled, WINDOW)
    X_test, y_test = make_windows(test_scaled, WINDOW)
    print("Bentuk X_train:", X_train.shape, "| X_test:", X_test.shape)

    # 4. Model training
    model = build_model()
    model.summary()
    early_stop = EarlyStopping(monitor="val_loss", patience=5,
                               restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                        validation_split=0.1, callbacks=[early_stop], verbose=1,
                        shuffle=False)

    # 5. Evaluasi (di skala harga, supaya adil dengan baseline)
    pred_scaled = model.predict(X_test)
    pred_ret = scaler.inverse_transform(pred_scaled).flatten()
    y_true = prices[train_size + 1:]
    y_prev = prices[train_size:-1]
    y_pred = y_prev * (1 + pred_ret)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    # Baseline: harga besok = harga hari ini
    mae_b = float(mean_absolute_error(y_true, y_prev))
    rmse_b = float(np.sqrt(mean_squared_error(y_true, y_prev)))
    mape_b = float(np.mean(np.abs((y_true - y_prev) / y_true)) * 100)
    print(f"Baseline -> MAE: Rp{mae_b:,.2f} | RMSE: Rp{rmse_b:,.2f} | MAPE: {mape_b:.2f}%")
    print(f"MAE  : Rp{mae:,.2f}\nRMSE : Rp{rmse:,.2f}\nMAPE : {mape:.2f}%")

    metrics = {
        "ticker": TICKER,
        "jumlah_baris_data": int(len(prices)),
        "data_latih": int(train_size),
        "data_uji": int(len(y_true)),
        "epoch_berhenti": int(len(history.history["loss"])),
        "best_val_loss": float(min(history.history["val_loss"])),
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "MAE_baseline": mae_b,
        "RMSE_baseline": rmse_b,
        "MAPE_baseline": mape_b,
    }
    with open(os.path.join(BASE, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Grafik kurva loss
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Loss latih")
    plt.plot(history.history["val_loss"], label="Loss validasi")
    plt.title("Kurva Loss Pelatihan")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "loss_curve.png"), dpi=150)
    plt.close()

    # Grafik aktual vs prediksi
    test_dates = close.index[train_size + 1:]
    plt.figure(figsize=(12, 5))
    plt.plot(test_dates, y_true, label="Aktual")
    plt.plot(test_dates, y_pred, label="Prediksi")
    plt.title(f"Harga Aktual vs Prediksi {TICKER} (Data Uji)")
    plt.xlabel("Tanggal")
    plt.ylabel("Harga (Rp)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "actual_vs_pred.png"), dpi=150)
    plt.close()

    # 6. Simpan model dan scaler untuk deployment
    model.save(os.path.join(BASE, "model_rnn.keras"))
    joblib.dump(scaler, os.path.join(BASE, "scaler.pkl"))
    print("Model, scaler, dan metrik tersimpan.")


if __name__ == "__main__":
    main()
