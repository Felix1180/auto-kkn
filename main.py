import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime
from joblib import load
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from dotenv import load_dotenv

# Load env
load_dotenv()

API_URL = "https://ensemble-api.open-meteo.com/v1/ensemble?latitude=-1.1757279780454264&longitude=116.86677769097581&hourly=temperature_2m,cloud_cover&timezone=Asia%2FSingapore&forecast_days=30&models=gfs_seamless"
DATASET_PATH = "dataset/GFS_Ensemble_Seamless.csv"
MODEL_PATH = "models/best_rf_model.joblib"
OUTPUT_CSV = "dataset/hasil_prediksi.csv"

def fetch_forecast():
    r = requests.get(API_URL)
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame({
        "time": data["hourly"]["time"],
        "temperature": data["hourly"]["temperature_2m"],
        "cloud_cover": data["hourly"]["cloud_cover"]
    })
    return df

def preprocess(df):
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)

    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month

    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)

    lags = [1, 2, 3]
    for lag in lags:
        df[f'temp_lag_{lag}'] = df['temperature'].shift(lag)
    df['temp_roll_mean_3'] = df['temperature'].rolling(3).mean()
    df['temp_roll_std_3'] = df['temperature'].rolling(3).std()

    df.dropna(inplace=True)

    feature_order = [
        'temperature', 'cloud_cover', 'hour', 'dayofweek', 'month',
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
        'temp_lag_1', 'temp_lag_2', 'temp_lag_3',
        'temp_roll_mean_3', 'temp_roll_std_3'
    ]

    return df[feature_order]

def predict_and_save(df_features, timestamps):
    model = load(MODEL_PATH)
    prediction = model.predict(df_features)
    result = pd.DataFrame({
        "Tanggal": timestamps,
        "Ramalan": prediction
    })
    result.to_csv(OUTPUT_CSV, index=False)
    return result

def save_to_gsheet(df_result):
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
    client = gspread.authorize(creds)

    spreadsheet = client.open_by_url(os.getenv("SPREADSHEET_URL"))
    sheet = spreadsheet.worksheet("Ramalan")

    sheet.clear()

    # Siapkan header + data
    data_to_write = [["Tanggal", "Ramalan"]]
    
    for _, row in df_result.iterrows():
        tanggal_str = pd.to_datetime(row['Tanggal']).strftime('%Y-%m-%d %H:%M')
        data_to_write.append([tanggal_str, row['Ramalan']])

    # Tulis data sekaligus dalam satu request
    sheet.append_rows(data_to_write)



if __name__ == "__main__":
    print("🚀 Memulai pipeline harian...")

    # 1. Ambil data terbaru
    new_df = fetch_forecast()
    new_df.to_csv(DATASET_PATH, index=False)  # Optional: simpan data mentah

    # 2. Proses dan prediksi
    processed = preprocess(new_df)
    result = predict_and_save(processed, processed.index)

    # 3. Simpan ke Google Sheet
    save_to_gsheet(result)

    print("✅ Automasi selesai.")
