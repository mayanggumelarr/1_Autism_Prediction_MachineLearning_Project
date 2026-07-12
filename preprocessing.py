import pandas as pd
import numpy as np

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace({
        'yes' : 1,
        'no' : 0,
        '?' : 'Others',
        'others' : 'Others'
    })
    return df

def convert_age(age):
    if age < 4:
        return 'Toddler'
    elif age >= 4 and age < 12:
        return 'Kid'
    elif age >= 12 and age < 18:
        return 'Teenager'
    elif age >= 18 and age < 40:
        return 'Adult'
    else:
        return 'Senior'
    
def add_clinic_feature(data: pd.DataFrame) -> pd.DataFrame:
    data['sum_score'] = 0
    for col in data.loc[:, 'A1_Score':'A10_Score'].columns:
        data['sum_score'] += data[col]

    data['ind'] = data['austim'] + data['used_app_before'] + data['jaundice']

    return data

def encode_with_saved_encoders(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    """
    1. Untuk proses encode pada data baru saat prediksi
    2. Menggunakan encoders yang sudah disimpan
    """
    df = df.copy()
    for col, le in encoders.items():
        if col not in df.columns:
            continue
        known_classes = set(le.classes)
        df[col] = df[col].apply(lambda v: v if v in known_classes else le.classes_[0])
        df[col] = le.transform(df[col])
    return df