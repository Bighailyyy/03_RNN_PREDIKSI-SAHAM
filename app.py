"""
Antarmuka Streamlit untuk prediksi harga saham BBCA.JK dan BBRI.JK dengan SimpleRNN.
Jalankan: streamlit run app.py
(Pastikan folder hasil-BBCA dan hasil-BBRI berisi model_rnn.keras, scaler.pkl, metrics.json.)
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from tensorflow.keras.models import load_model

WINDOW = 60

SAHAM = {
    "BBCA - Bank Central Asia": {"ticker": "BBCA.JK", "folder": "hasil-BBCA"},
    "BBRI - Bank Rakyat Indonesia": {"ticker": "BBRI.JK", "folder": "hasil-BBRI"},
}

st.set_page_config(page_title="Prediksi Saham", page_icon="📈")
st.title("📈 Prediksi Harga Saham dengan RNN")


@st.cache_resource
def load_artifacts(folder):
    model = load_model(os.path.join(folder, "model_rnn.keras"))
    scaler = joblib.load(os.path.join(folder, "scaler.pkl"))
    return model, scaler


@st.cache_data(ttl=3600)
def get_recent_data(ticker):
    df = yf.download(ticker, period="2y", auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.sort_index()


pilihan = st.selectbox("Kode saham", list(SAHAM.keys()))
ticker = SAHAM[pilihan]["ticker"]
folder = SAHAM[pilihan]["folder"]

st.info("Model tersedia untuk BBCA.JK dan BBRI.JK.")

model_path = os.path.join(folder, "model_rnn.keras")
scaler_path = os.path.join(folder, "scaler.pkl")
if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
    st.error(f"File model tidak ditemukan di folder {folder}. Jalankan training terlebih dahulu.")
    st.stop()

model, scaler = load_artifacts(folder)

if st.button("Prediksi"):
    df = get_recent_data(ticker)
    if df.empty or "Close" not in df.columns:
        st.error("Data tidak ditemukan. Periksa kode saham atau koneksi internet.")
        st.stop()

    close = df["Close"].ffill().dropna()
    if len(close) < WINDOW + 1:
        st.error(f"Data kurang dari {WINDOW + 1} hari, prediksi tidak bisa dilakukan.")
        st.stop()

    prices = close.values[-(WINDOW + 1):]
    rets = (prices[1:] / prices[:-1] - 1).reshape(-1, 1)
    scaled = scaler.transform(rets).reshape(1, WINDOW, 1)
    pred_ret = scaler.inverse_transform(model.predict(scaled)).item()
    last_price = float(prices[-1])
    pred = last_price * (1 + pred_ret)
    diff = pred - last_price

    st.subheader(f"Hasil Prediksi {pilihan}")
    col1, col2 = st.columns(2)
    col1.metric("Harga penutupan terakhir", f"Rp{last_price:,.0f}")
    col2.metric("Prediksi hari berikutnya", f"Rp{pred:,.0f}",
                f"{diff:+,.0f} ({diff / last_price * 100:+.2f}%)")

    st.subheader("Harga Penutupan 2 Tahun Terakhir")
    st.line_chart(close)

metrics_path = os.path.join(folder, "metrics.json")
if os.path.exists(metrics_path):
    with st.expander("Kinerja model pada data uji"):
        with open(metrics_path) as f:
            m = json.load(f)
        st.write(f"MAE: Rp{m['MAE']:,.2f}")
        st.write(f"RMSE: Rp{m['RMSE']:,.2f}")
        st.write(f"MAPE: {m['MAPE']:.2f}%")
        st.write("---")
        st.write("Baseline (harga besok = harga hari ini):")
        st.write(f"MAE: Rp{m['MAE_baseline']:,.2f}")
        st.write(f"RMSE: Rp{m['RMSE_baseline']:,.2f}")
        st.write(f"MAPE: {m['MAPE_baseline']:.2f}%")