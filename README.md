# Prediksi Harga Saham BBCA.JK dan BBRI.JK dengan Recurrent Neural Network (SimpleRNN)

Tugas UAS INF311 Machine Learning, **Kelompok 3: Recurrent Neural Network (RNN)**
Program Studi S1 Informatika, FMIPA, Universitas Syiah Kuala.

Anggota:
1. Putroe Fatimah Azzahra - 2408107010002
2. Fawwaz Ziyadi Ilmi - 2408107010021
3. Annisa Rahma Fathia - 2408107010027
4. Fayla Syifa Rustam - 2408107010029

## Deskripsi Singkat

Proyek ini memprediksi **harga penutupan (Close)** saham BBCA.JK dan BBRI.JK untuk satu hari perdagangan berikutnya dengan **SimpleRNN**, lalu membandingkannya dengan baseline sederhana "harga besok = harga hari ini". Ada dua skenario:

- **Skenario A (RNN harga):** model memprediksi harga langsung (normalisasi Min-Max).
- **Skenario B (RNN return):** model memprediksi return harian (StandardScaler), lalu harga dihitung dengan `harga_terakhir × (1 + return_prediksi)`. Skenario B dipakai di aplikasi.

## Konfigurasi Model

| Komponen | Konfigurasi |
|---|---|
| Data | Yahoo Finance (yfinance), 2015-01-01 s.d. 2025-12-31, 2.711 baris per saham |
| Fitur | Harga Close (Skenario B: return harian, distandardisasi) |
| Window | 60 hari |
| Pembagian data | 80% latih, 20% uji (kronologis) |
| Arsitektur | SimpleRNN(64) → Dropout(0.2) → SimpleRNN(32) → Dropout(0.2) → Dense(1) |
| Training | Adam (lr 0.001), loss MSE, batch 32, maks 50 epoch, EarlyStopping (patience 5), seed 42 |
| Evaluasi | MAE, RMSE, MAPE pada skala harga, dibandingkan dengan baseline |

## Hasil pada Data Uji

| Saham | Model | MAE (Rp) | RMSE (Rp) | MAPE (%) |
|---|---|---|---|---|
| BBCA.JK | RNN harga (A) | 221,21 | 273,84 | 2,34 |
| BBCA.JK | RNN return (B) | 107,83 | 143,33 | 1,18 |
| BBCA.JK | Baseline | 106,69 | 143,09 | 1,17 |
| BBRI.JK | RNN harga (A) | 164,56 | 196,88 | 3,84 |
| BBRI.JK | RNN return (B) | 69,17 | 92,02 | 1,54 |
| BBRI.JK | Baseline | 67,63 | 90,34 | 1,51 |

Ringkasan: RNN harga kalah dari baseline. Memprediksi return menurunkan error sekitar 50 sampai 60% dan mendekati baseline, tetapi belum mengalahkannya. Harga saham harian mendekati random walk, dan pada Skenario A harga periode uji melewati rentang scaler data latih.

## Antarmuka (Streamlit)

`app.py` menyediakan:
- Dropdown pilihan saham (BBCA dan BBRI).
- Tombol **Prediksi**: mengambil data terbaru, lalu menampilkan harga penutupan terakhir dan prediksi hari berikutnya beserta selisihnya.
- Grafik harga penutupan 2 tahun terakhir.
- Panel kinerja model (MAE, RMSE, MAPE) beserta baseline.

_(Tambahkan tangkapan layar di sini, misalnya `![UI BBCA](screenshots/ui-bbca.png)`.)_

## Struktur Proyek

```
.
├── train_model.py         # pipeline: data, EDA, preprocessing, training, evaluasi, simpan model
├── app.py                 # antarmuka Streamlit
├── requirements.txt
├── hasil-BBCA/            # hasil Skenario B untuk BBCA
│   ├── model_rnn.keras
│   ├── scaler.pkl
│   ├── metrics.json
│   └── outputs/           # grafik EDA, kurva loss, aktual vs prediksi
├── hasil-BBRI/            # hasil Skenario B untuk BBRI (struktur sama)
├── hasil-lama-BBCA/       # hasil Skenario A (pembanding)
└── hasil-lama-BBRI/
```

## Cara Menjalankan

1. Pasang dependensi (disarankan Python 3.10 sampai 3.12):
   ```bash
   pip install -r requirements.txt
   ```
2. Latih model (butuh internet untuk mengunduh data). Ubah `TICKER` di bagian konfigurasi `train_model.py` menjadi `"BBCA.JK"` atau `"BBRI.JK"`, lalu jalankan untuk masing-masing saham:
   ```bash
   python train_model.py
   ```
3. Jalankan aplikasi:
   ```bash
   streamlit run app.py
   ```

Model yang sudah dilatih tersedia di `hasil-BBCA` dan `hasil-BBRI`, jadi langkah 2 bisa dilewati jika hanya ingin mencoba aplikasi. Catatan: folder `hasil-lama-...` berisi model Skenario A dan tidak kompatibel dengan `app.py` saat ini.

## Catatan

Proyek ini bersifat akademik dan **bukan saran investasi**.
