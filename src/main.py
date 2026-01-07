from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from datetime import datetime

app = FastAPI(title="Flight Delay Prediction API")

# Charger modèle et outils
model = joblib.load("models/lightgbm_model.pkl")
encoders = joblib.load("models/encoders.pkl")
selected_features = joblib.load("models/selected_features_LGB.pkl")

# =========================
# INPUT UTILISATEUR (HUMAIN)
# =========================
class FlightData(BaseModel):
    crs_dep_hour: int
    crs_dep_min: int
    crs_arr_hour: int
    crs_arr_min: int
    flight_date: str  # "YYYY-MM-DD"
    op_carrier_fl_num: int
    origin_city_name: str
    origin_state_nm: str
    dest_city_name: str
    dest_state_nm: str
    
@app.get("/")
def home():
    return {"message": "API de prédiction de retards de vols active. Allez sur /docs pour tester."}
# =========================
# FONCTIONS UTILITAIRES
# =========================
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
# ENDPOINT DE PRÉDICTION
# =========================
@app.post("/predict")
def predict_delay(flight: FlightData):

    # Date
    date = datetime.strptime(flight.flight_date, "%Y-%m-%d")
    fl_month = date.month
    fl_day = date.day
    fl_dayofweek = date.weekday()
    is_weekend = 1 if fl_dayofweek in [5, 6] else 0
    season = get_season(fl_month)

    # Encodage catégoriel
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
        "season": encoders["season"].transform([season])[0],
        "origin_city_name": encoders["origin_city_name"].transform([flight.origin_city_name])[0],
        "origin_state_nm": encoders["origin_state_nm"].transform([flight.origin_state_nm])[0],
        "dest_city_name": encoders["dest_city_name"].transform([flight.dest_city_name])[0],
        "dest_state_nm": encoders["dest_state_nm"].transform([flight.dest_state_nm])[0],
    }

    df = pd.DataFrame([data])

    # S'assurer de l'ordre exact des features
    df = df[selected_features]

    prediction = model.predict(df)[0]
    proba = model.predict_proba(df)[0][1]

    return {
        "delay_predicted": bool(prediction),
        "delay_probability": round(float(proba), 3)
    }