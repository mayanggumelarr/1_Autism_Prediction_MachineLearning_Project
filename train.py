import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler

import warnings
warnings.filterwarnings('ignore')

def load_data(path: str = 'train.csv') -> pd.DataFrame:
    """Step 1: Import dataset"""
    return pd.read_csv(path)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Step 2: Data Cleaning"""
    df = df.replace({
        'yes' : 1,
        'no' : 0,
        '?' : 'Others',
        'others' : 'Others'
    })
    return df

def convert_age(age):
    """Step 3: Feature Engineering Group age"""
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
    """Step 4: Menambahkan feature add skor klinis"""
    data['sum_score'] = 0
    for col in data.loc[:, 'A1_Score':'A10_Score'].columns:
        data['sum_score'] += data[col]

    data['ind'] = data['austim'] + data['used_app_before'] + data['jaundice']

    return data

def encode_labels(data: pd.DataFrame) -> pd.DataFrame:
    """Step 5: Encoding Label"""
    encoders = {}
    for col in data.columns:
        if data[col].dtype == 'object':
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
    return data, encoders

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Step 6: Feature Engineering
    1. Buat ageGroup dari fungsi convert_age
    2. Tabah sum_score & ind dengan add_clinic_feature
    3. Log transform age
    4. Label encoding semua kolom object
    """
    df['ageGroup'] = df['age'].apply(convert_age)
    df = add_clinic_feature(df)
    df['age'] = df['age'].apply(lambda x: np.log(x))
    df = encode_labels(df)
    return df

def train_and_evaluate(df: pd.DataFrame):
    removal = ['ID', 'age_desc', 'used_app_before', 'austim']
    features = df.drop(removal + ['Class/ASD'], axis=1)
    target = df['Class/ASD']
    feature_columns = list(features.columns)

    # Split
    X_train, X_val, Y_train, Y_val = train_test_split(features, target, test_size = 0.2, random_state=10)

    # Imbalanced data handling
    ros = RandomOverSampler(sampling_strategy='minority',random_state=0)
    X, Y = ros.fit_resample(X_train,Y_train)

    # Scaling
    scaler = StandardScaler()
    X_scal = scaler.fit_transform(X)
    X_val = scaler.transform(X_val)

    # Train
    model = LogisticRegression()
    model.fit(X_scal, Y)

    # Evaluasi
    print(f'{model} : ')
    print(f'ROC AUC Score Train : ', metrics.roc_auc_score(Y, model.predict(X)))
    print(f'Classification Report Train: \n', metrics.classification_report(Y, model.predict(X)))
    print(f'ROC AUC Score Val : ', metrics.roc_auc_score(Y_val, model.predict(X_val)))
    print(f'Classification Report Val: \n', metrics.classification_report(Y_val, model.predict(X_val)))
    print('===============================================================================')

    # Saving
    best_model = model
    return best_model, scaler, feature_columns

def main():
    df = load_data('data/train.csv')
    df = clean_data(df)
    df, encoders = feature_engineering(df)
    model, scaler, feature_columns = train_and_evaluate(df)

    # Bagian Baru: Simpan model di folder models/
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(encoders, 'models/encoders.pkl')
    joblib.dump(feature_columns, 'models/feature_columns.pkl')

    print('\nModel & artifacts tersimpan di folder models/:')
    print('  - models/model.pkl            (Logistic Regression yang sudah dilatih)')
    print('  - models/scaler.pkl           (StandardScaler)')
    print('  - models/encoders.pkl         (LabelEncoder tiap kolom kategorikal)')
    print('  - models/feature_columns.pkl  (urutan kolom fitur saat training)')

if __name__== '__main__':
    main()