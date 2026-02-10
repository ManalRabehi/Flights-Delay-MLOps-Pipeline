from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd
from datetime import datetime

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
    crs_dep_hour: int = Field(..., example=14)
    crs_dep_min: int = Field(..., example=30)
    crs_arr_hour: int = Field(..., example=16)
    crs_arr_min: int = Field(..., example=10)
    flight_date: str = Field(..., example="2024-07-15")
    op_carrier_fl_num: int = Field(..., example=1234)
    origin_city_name: str = Field(..., example="New York, NY")
    origin_state_nm: str = Field(..., example="New York")
    dest_city_name: str = Field(..., example="Los Angeles, CA")
    dest_state_nm: str = Field(..., example="California")

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

    return {
        "delay_predicted": bool(prediction),
        "delay_probability": round(float(proba), 3)
    }
