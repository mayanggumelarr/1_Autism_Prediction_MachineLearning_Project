"""
main.py

FastAPI sederhana untuk prediksi Autism (ASD).
Cuma ada 2 endpoint:
  - GET  /health   -> cek apakah model sudah berhasil dimuat
  - POST /predict  -> kirim data 1 orang, dapat hasil prediksi

Cara jalanin (pastikan sudah training dulu, folder models/ sudah ada isinya):
    pip install fastapi uvicorn
    uvicorn main:app --reload

Lalu buka browser: http://127.0.0.1:8000/docs
(FastAPI otomatis bikin halaman coba-coba API di situ)
"""

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from preprocessing import clean_data, convert_age, add_clinic_feature, encode_with_saved_encoders

app = FastAPI(title="Autism Prediction API")

# --- Load semua artifacts SEKALI saja, waktu aplikasi start ---
model = joblib.load('models/model.pkl')
scaler = joblib.load('models/scaler.pkl')
encoders = joblib.load('models/encoders.pkl')
feature_columns = joblib.load('models/feature_columns.pkl')


# --- Skema input: field-field ini sesuai kolom asli di train.csv ---
class AutismInput(BaseModel):
    A1_Score: int
    A2_Score: int
    A3_Score: int
    A4_Score: int
    A5_Score: int
    A6_Score: int
    A7_Score: int
    A8_Score: int
    A9_Score: int
    A10_Score: int
    age: float
    gender: str          # 'm' atau 'f'
    ethnicity: str        # misal 'White-European'
    jaundice: str          # 'yes' atau 'no'
    austim: str            # 'yes' atau 'no' (riwayat keluarga)
    contry_of_res: str     # nama negara
    used_app_before: str   # 'yes' atau 'no'
    result: float
    age_desc: str = "18 and more"
    relation: str          # misal 'Self', 'Parent'


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(data: AutismInput):
    try:
        # 1. Ubah input jadi 1 baris DataFrame
        df = pd.DataFrame([data.model_dump()])

        # 2. Terapkan preprocessing YANG SAMA PERSIS dengan train.py
        df = clean_data(df)
        df['ageGroup'] = df['age'].apply(convert_age)
        df = add_clinic_feature(df)
        df['age'] = df['age'].apply(lambda x: np.log(x))
        df = encode_with_saved_encoders(df, encoders)

        # 3. Drop kolom yang memang tidak dipakai model (sama seperti train.py)
        removal = ['age_desc', 'used_app_before', 'austim']
        df = df.drop(columns=[c for c in removal if c in df.columns])

        # 4. Pastikan urutan kolom SAMA seperti saat training
        df = df[feature_columns]

        # 5. Scaling, lalu prediksi
        X_scaled = scaler.transform(df)
        prediction = int(model.predict(X_scaled)[0])
        probability = float(model.predict_proba(X_scaled)[0][1])

        return {
            "prediction": prediction,
            "probability_asd": round(probability, 4)
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Gagal memproses input: {e}")