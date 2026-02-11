from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd
from datetime import datetime
import os 
import sqlite3
from fastapi.responses import FileResponse




# Base de données #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

DB_DIR = os.path.join(PROJECT_ROOT, "baseDonnees")
os.makedirs(DB_DIR, exist_ok=True)

DB_PATH = os.path.join(DB_DIR, "predictions.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
                   CREATE TABLE IF NOT EXISTS predictions (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            crs_dep_hour INTEGER,
            crs_dep_min INTEGER,
            crs_arr_hour INTEGER,
            crs_arr_min INTEGER,
            flight_date TEXT,
            op_carrier_fl_num INTEGER,
            origin_city_name TEXT,
            origin_state_nm TEXT,
            dest_city_name TEXT,
            dest_state_nm TEXT,
            distance REAL,
            crs_elapsed_time REAL,
            predicted_delay INTEGER,
            predicted_probability REAL
        )
    """)
    conn.commit()
    conn.close()

init_db()

# =========================
# UTILITAIRES
# =========================

def normalize_text(value: str) -> str:
    return value.strip()

def safe_encode(encoder, value: str) -> int:
    """Encode une valeur si elle est connue, sinon retourne -1."""
    value = normalize_text(value)
    if value in encoder.classes_:
        return encoder.transform([value])[0]
    else:
        return -1  # <-- Valeur inconnue

def get_season(month: int) -> str:
    if month in [12, 1, 2]:
        return "winter"
    elif month in [3, 4, 5]:
        return "spring"
    elif month in [6, 7, 8]:
        return "summer"
    else:
        return "fall"

# =========================
# STRUCTURES DE DONNÉES
# =========================
class PredictionResponse(BaseModel):
    delay_predicted : bool
    delay_probability : float
    
class FlightData(BaseModel):
    crs_dep_hour: int = Field(..., ge=0, le=23, example=14)
    crs_dep_min: int = Field(..., ge=0, le=59, example=30)
    crs_arr_hour: int = Field(..., ge=0, le=23, example=16)
    crs_arr_min: int = Field(..., ge=0, le=59, example=10)
    flight_date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$", example="2024-07-15")
    op_carrier_fl_num: int = Field(..., ge=1, example=1234)
    origin_city_name: str = Field(..., max_length=100, pattern=r'^[\w\s,.-]+$', example="New York, NY")
    origin_state_nm: str = Field(..., max_length=50, pattern=r'^[\w\s.-]+$', example="New York")
    dest_city_name: str = Field(..., max_length=100, pattern=r'^[\w\s,.-]+$', example="Los Angeles, CA")
    dest_state_nm: str = Field(..., max_length=50, pattern=r'^[\w\s.-]+$', example="California")
    distance: float = Field(..., ge=0, example=2450.0)
    crs_elapsed_time: float = Field(..., ge=0, example=260.0)

# =========================
# INITIALISATION DE L'API
# =========================
app = FastAPI(title="Flight Delay Prediction API")

# Charger modèle et outils
model = joblib.load("models/lightgbm_model.pkl")
encoders = joblib.load("models/encoders.pkl")
selected_features = joblib.load("models/selected_features_LGB.pkl")
medians = joblib.load("models/median_values.pkl")  # médianes calculées lors du preprocessing

@app.get("/")
def home():
    return {"message": "API de prédiction de retards de vols active. Allez sur /docs pour tester."}

@app.get("/download-logs")
def download_logs():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM predictions", conn)
    conn.close()

    csv_path = os.path.join(BASE_DIR, "predictions_export.csv")
    df.to_csv(csv_path, index=False)

    return FileResponse(csv_path, media_type='text/csv', filename="predictions_log.csv")

# =========================
# ROUTE DE PREDICTION
# =========================
@app.post("/predict", response_model=PredictionResponse)
def predict_delay(flight: FlightData):
    # --- Features temporelles ---
    date = datetime.strptime(flight.flight_date, "%Y-%m-%d")
    fl_month = date.month
    fl_day = date.day
    fl_dayofweek = date.weekday()
    is_weekend = 1 if fl_dayofweek in [5, 6] else 0
    season = get_season(fl_month).lower()

    # --- Construction du dictionnaire de features ---
    data = {
        "crs_dep_hour": flight.crs_dep_hour,
        "crs_dep_min": flight.crs_dep_min,
        "crs_arr_hour": flight.crs_arr_hour,
        "crs_arr_min": flight.crs_arr_min,
        "fl_month": fl_month,
        "fl_day": fl_day,
        "fl_dayofweek": fl_dayofweek,
        "is_weekend": is_weekend,
        "op_carrier_fl_num": flight.op_carrier_fl_num,
        "season": safe_encode(encoders["season"], season),
        "origin_city_name": safe_encode(encoders["origin_city_name"], flight.origin_city_name),
        "origin_state_nm": safe_encode(encoders["origin_state_nm"], flight.origin_state_nm),
        "dest_city_name": safe_encode(encoders["dest_city_name"], flight.dest_city_name),
        "dest_state_nm": safe_encode(encoders["dest_state_nm"], flight.dest_state_nm),
        "distance": flight.distance,
        "crs_elapsed_time": flight.crs_elapsed_time
    }

    # --- Colonnes numériques manquantes ---
    for col in selected_features:
        if col not in data:
            data[col] = medians.get(col, 0)

    df = pd.DataFrame([data])
    df = df[selected_features]

    # --- DEBUG ---
    unknowns = {col: val for col, val in df.iloc[0].items() if val == -1}
    if unknowns:
        print("⚠️ Valeurs encodées comme inconnues:", unknowns)

    # --- Prédiction ---
    print(df.head())
    prediction = model.predict(df)[0]
    proba = model.predict_proba(df)[0, 1]

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO predictions (
        timestamp,
        crs_dep_hour,
        crs_dep_min,
        crs_arr_hour,
        crs_arr_min,
        flight_date,
        op_carrier_fl_num,
        origin_city_name,
        origin_state_nm,
        dest_city_name,
        dest_state_nm,
        distance,
        crs_elapsed_time,
        predicted_delay,
        predicted_probability
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
""", (
    timestamp,
    flight.crs_dep_hour,
    flight.crs_dep_min,
    flight.crs_arr_hour,
    flight.crs_arr_min,
    flight.flight_date,
    flight.op_carrier_fl_num,
    flight.origin_city_name,
    flight.origin_state_nm,
    flight.dest_city_name,
    flight.dest_state_nm,
    flight.distance,
    flight.crs_elapsed_time,
    int(prediction),
    float(proba)
))

    conn.commit()
    conn.close()

    
    # Log console
    print(f"[{timestamp}] Prédiction pour {flight.origin_city_name} -> {flight.dest_city_name} : "
          f"retard = {prediction} (proba = {round(float(proba),3)})")


    return {
        "delay_predicted": bool(prediction),
        "delay_probability": round(float(proba), 3)
    }