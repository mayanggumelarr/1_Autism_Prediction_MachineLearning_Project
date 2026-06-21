# Autism Prediction

Proyek Machine Learning untuk memprediksi indikasi Autism Spectrum Disorder (ASD) pada seseorang berdasarkan data screening (kuesioner AQ-10) dan data demografis.

## 📋 Daftar Isi

- [Latar Belakang](#latar-belakang)
- [Tujuan](#tujuan)
- [Dataset](#dataset)
- [Struktur Proyek](#struktur-proyek)
- [Instalasi](#instalasi)
- [Cara Penggunaan](#cara-penggunaan)
- [Metodologi](#metodologi)
- [Hasil & Evaluasi Model](#hasil--evaluasi-model)
- [Kesimpulan](#kesimpulan)
- [Catatan Pengembangan Selanjutnya](#catatan-pengembangan-selanjutnya)

## Latar Belakang

Autisme (Autism Spectrum Disorder/ASD) merupakan salah satu gangguan perkembangan yang hingga saat ini belum memiliki metode pasti untuk diagnosis dini. Proses diagnosis konvensional umumnya membutuhkan waktu lama dan melibatkan observasi klinis oleh tenaga profesional.

Machine Learning hadir sebagai solusi pendukung untuk membantu memprediksi apakah seseorang terindikasi Autisme atau tidak, berdasarkan pola jawaban skrining dan atribut demografis, sehingga dapat menjadi alat bantu deteksi awal (bukan pengganti diagnosis klinis).

## Tujuan

Membangun model klasifikasi yang mampu memprediksi label `Class/ASD` (0 = tidak terindikasi, 1 = terindikasi) menggunakan data skor screening AQ-10 beserta atribut pendukung lainnya (usia, gender, etnis, riwayat jaundice, riwayat keluarga autisme, dll).

## Dataset

Dataset menggunakan file `train.csv` dengan total **22 kolom**, terdiri dari:

| Kategori | Kolom |
|---|---|
| Identitas | `ID` |
| Skor Screening (AQ-10) | `A1_Score` – `A10_Score` (biner 0/1) |
| Demografis | `age`, `gender`, `ethnicity`, `contry_of_res`, `age_desc` |
| Riwayat Medis | `jaundice`, `austim` (riwayat keluarga) |
| Atribut Lain | `used_app_before`, `result`, `relation` |
| **Target** | `Class/ASD` (0/1) |

> Dataset tidak disertakan dalam repositori ini. Letakkan file `train.csv` pada direktori root proyek sebelum menjalankan notebook.

## Struktur Proyek

```
.
├── 1_Autism_Prediction.ipynb   # Notebook utama (EDA, preprocessing, training, evaluasi)
├── train.csv                   # Dataset (disediakan sendiri oleh pengguna)
├── README.md                   # Dokumentasi proyek
└── requirements.txt            # Daftar dependensi Python
```

## Instalasi

1. **Clone / unduh proyek ini**

   ```bash
   git clone <url-repo-anda>
   cd <nama-folder-proyek>
   ```

2. **(Opsional, direkomendasikan) Buat virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate    # Linux/Mac
   venv\Scripts\activate       # Windows
   ```

3. **Install dependensi**

   ```bash
   pip install -r requirements.txt
   ```

4. **Siapkan dataset**

   Pastikan file `train.csv` berada pada direktori yang sama dengan notebook.

## Cara Penggunaan

1. Jalankan Jupyter Notebook atau buka di Google Colab:

   ```bash
   jupyter notebook 1_Autism_Prediction.ipynb
   ```

2. Jalankan setiap cell secara berurutan dari atas ke bawah (`Run All`), karena setiap tahap (cleaning → EDA → feature engineering → modeling) saling bergantung pada tahap sebelumnya.

## Metodologi

Notebook ini disusun dalam 5 tahapan utama:

### 1. Import Libraries & Dataset
Memuat seluruh library yang dibutuhkan serta membaca dataset `train.csv`.

### 2. Data Cleaning
- Mengganti nilai `yes`/`no` menjadi `1`/`0`.
- Menstandarkan nilai placeholder yang tidak diketahui (`?`, `others`) menjadi `Others`.

### 3. Exploratory Data Analysis (EDA)
- Analisis distribusi target (`Class/ASD`) — termasuk visualisasi pie chart.
- Visualisasi distribusi seluruh fitur integer (skor A1–A10) terhadap target.
- Visualisasi distribusi fitur kategorikal (gender, ethnicity, relation, dll).
- Analisis distribusi berdasarkan negara (`contry_of_res`).
- Analisis distribusi data numerik kontinu (`age`, `result`) menggunakan distplot & boxplot, termasuk deteksi outlier pada fitur `age`.

**Insight utama dari EDA:**
- Distribusi pria yang diprediksi terindikasi autisme lebih banyak dibandingkan perempuan.
- Etnis dengan distribusi autisme terbesar adalah *White-European*.
- Negara dengan jumlah kasus tertinggi: Amerika Serikat, disusul Inggris.
- Fitur `age` memiliki outlier (usia > 60 tahun) namun tetap dipertahankan (tidak di-drop) untuk menjaga jumlah sampel kelas minoritas, dengan catatan model yang dipilih perlu robust terhadap outlier.

### 4. Feature Engineering
- **`ageGroup`** — pengelompokan usia menjadi kategori: *Toddler, Kid, Teenager, Adult, Senior*.
- **`sum_score`** — total penjumlahan skor `A1_Score` hingga `A10_Score` sebagai representasi tingkat keparahan indikasi.
- **`ind`** — kombinasi indikator riwayat (`austim` + `used_app_before` + `jaundice`).
- **Log transform** pada fitur `age` untuk menormalkan distribusi yang skewed.
- **Label Encoding** pada seluruh kolom bertipe kategorikal (object).
- Analisis korelasi antar fitur menggunakan heatmap.

### 5. Model Training & Evaluation
- **Split data**: train-validation split (80:20) dengan `random_state=10`.
- **Penanganan imbalanced data**: menggunakan `RandomOverSampler` pada kelas minoritas (hanya diterapkan ke data train).
- **Standardisasi fitur**: menggunakan `StandardScaler`.
- **Fitur yang di-drop** sebelum training: `ID`, `age_desc`, `used_app_before`, `austim` (karena redundan/tidak relevan atau sudah terwakili oleh fitur `ind`).
- **Model yang dibandingkan**:
  - Logistic Regression
  - XGBoost Classifier
  - Decision Tree Classifier
- **Metrik evaluasi**: ROC-AUC Score dan Classification Report (precision, recall, f1-score) pada data train maupun validasi.

## Hasil & Evaluasi Model

| Model | ROC AUC (Train) | ROC AUC (Validation) |
|---|---|---|
| Logistic Regression | 0.50 | **0.81** |
| XGBoost | 0.80 | 0.73 |
| Decision Tree | 0.49 | 0.65 |

**Analisis:**
- **Logistic Regression** menunjukkan peningkatan performa yang signifikan dari train ke validasi, mengindikasikan model yang lebih general dan tidak overfitting terhadap data train.
- **XGBoost** mengalami penurunan performa (deaccuracy) sekitar 7% dari train ke validasi, mengindikasikan kecenderungan overfitting pada konfigurasi default.
- **Decision Tree** juga menunjukkan performa yang kurang stabil dibandingkan Logistic Regression.

## Kesimpulan

Berdasarkan perbandingan ROC-AUC pada data validasi, **Logistic Regression dipilih sebagai model terbaik** untuk kasus prediksi Autisme pada proyek ini, karena memberikan generalisasi yang lebih baik dibandingkan XGBoost dan Decision Tree pada konfigurasi default (tanpa hyperparameter tuning).

## Catatan Pengembangan Selanjutnya

Beberapa hal yang dapat dieksplorasi lebih lanjut untuk meningkatkan performa model:

- Melakukan **hyperparameter tuning** (misal `GridSearchCV` / `RandomizedSearchCV`) terutama untuk XGBoost dan Decision Tree agar tidak overfitting.
- Mencoba teknik balancing data lain seperti **SMOTE**.
- Melakukan **cross-validation** untuk evaluasi yang lebih robust dibandingkan single train-validation split.
- Menambahkan model lain seperti Random Forest, SVM, atau ensemble (voting/stacking).
- Melakukan **feature selection** lebih lanjut berdasarkan hasil korelasi dan feature importance.
- Menyimpan model akhir (`pickle`/`joblib`) beserta `scaler` dan `encoder` untuk keperluan deployment/inference.

---

**Disclaimer:** Model dalam proyek ini dibuat untuk tujuan edukasi dan penelitian. Hasil prediksi tidak dapat dijadikan sebagai pengganti diagnosis klinis profesional terhadap Autism Spectrum Disorder.
