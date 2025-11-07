import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os


def preprocess_data(
    input_path="data/raw/flight_data_2024.csv",
    output_path="data/processed/processed_data.csv"
):
    print("Chargement des données brutes...")
    df = pd.read_csv(input_path)

    # Conversion de la date
    df['fl_date'] = pd.to_datetime(df['fl_date'], errors='coerce')

    # Conversion des colonnes numériques
    numeric_cols = [
        'year', 'month', 'day_of_month', 'day_of_week',
        'op_carrier_fl_num', 'crs_dep_time', 'dep_time', 'dep_delay',
        'taxi_out', 'wheels_off', 'wheels_on', 'taxi_in', 'crs_arr_time',
        'arr_time', 'arr_delay', 'cancelled', 'diverted', 'crs_elapsed_time',
        'actual_elapsed_time', 'air_time', 'distance',
        'carrier_delay', 'weather_delay', 'nas_delay',
        'security_delay', 'late_aircraft_delay'
    ]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    #Conversion des colonnes catégorielles
    cat_cols = [
        'op_unique_carrier', 'origin', 'origin_city_name', 'origin_state_nm',
        'dest', 'dest_city_name', 'dest_state_nm', 'cancellation_code'
    ]
    for col in cat_cols:
        df[col] = df[col].astype('category')

    #Transformation des heures au format HH:MM
    def hhmm_format(hhmm):
        try:
            hhmm = int(hhmm)
            hours = hhmm // 100
            mins = hhmm % 100
            return f"{hours:02d}:{mins:02d}"
        except:
            return None

    time_cols = ['crs_dep_time', 'dep_time', 'wheels_off', 'wheels_on', 'crs_arr_time', 'arr_time']
    for col in time_cols:
        df[col + '_hhmm'] = df[col].apply(hhmm_format)

    df.drop(time_cols, axis=1, inplace=True)

    # Suppression de la colonne cancellation_code et remplissage des NaN
    df.drop('cancellation_code', axis=1, inplace=True)
    cols_fill = ['dep_delay', 'taxi_in', 'taxi_out', 'arr_delay', 'actual_elapsed_time', 'air_time']
    df[cols_fill] = df[cols_fill].fillna(df[cols_fill].median())

    # Nettoyage des valeurs manquantes et conversion en datetime
    date_cols = ['dep_time_hhmm', 'wheels_off_hhmm', 'wheels_on_hhmm', 'arr_time_hhmm']
    df.dropna(subset=date_cols, inplace=True)
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], format='%H:%M', errors='coerce')
    df.dropna(subset=date_cols, inplace=True)

    # Suppression et création de nouvelles features temporelles
    df.drop(columns=['year', 'month', 'day_of_month', 'day_of_week'], inplace=True)
    df['fl_year'] = df['fl_date'].dt.year
    df['fl_month'] = df['fl_date'].dt.month
    df['fl_day'] = df['fl_date'].dt.day
    df['fl_dayofweek'] = df['fl_date'].dt.dayofweek
    df['is_weekend'] = df['fl_dayofweek'].isin([5, 6]).astype(int)
    df.drop(columns=['fl_date'], inplace=True)

    df['season'] = pd.cut(
        df['fl_month'],
        bins=[0, 3, 6, 9, 12],
        labels=['winter', 'spring', 'summer', 'fall'],
        include_lowest=True
    )

    # Encodage des variables catégorielles
    cat_encode = [
        'op_unique_carrier', 'origin', 'origin_city_name',
        'origin_state_nm', 'dest', 'dest_city_name', 'dest_state_nm', 'season'
    ]
    df_encoded = df.copy()
    for col in cat_encode:
        enc = LabelEncoder()
        df_encoded[col] = enc.fit_transform(df_encoded[col].astype(str))

    # Standardisation des variables numériques continues
    num_col = [
        'dep_delay', 'taxi_out', 'taxi_in', 'arr_delay', 'crs_elapsed_time',
        'actual_elapsed_time', 'air_time', 'distance', 'carrier_delay',
        'weather_delay', 'nas_delay', 'security_delay', 'late_aircraft_delay'
    ]
    scaler = StandardScaler()
    df_encoded[num_col] = scaler.fit_transform(df_encoded[num_col])

    # Extraction des heures et minutes à partir des colonnes datetime
    date_cols_full = [
        'crs_dep_time_hhmm', 'dep_time_hhmm', 'wheels_off_hhmm',
        'wheels_on_hhmm', 'crs_arr_time_hhmm', 'arr_time_hhmm'
    ]
    for col in date_cols_full:
        df_encoded[f'{col}_hour'] = df_encoded[col].dt.hour
        df_encoded[f'{col}_minute'] = df_encoded[col].dt.minute

    df_encoded.drop(date_cols_full, axis=1, inplace=True)

    # Renommage des colonnes
    rename_dict = {
        'crs_dep_time_hhmm_hour': 'crs_dep_hour',
        'crs_dep_time_hhmm_minute': 'crs_dep_min',
        'dep_time_hhmm_hour': 'dep_hour',
        'dep_time_hhmm_minute': 'dep_min',
        'wheels_off_hhmm_hour': 'wheels_off_hour',
        'wheels_off_hhmm_minute': 'wheels_off_min',
        'wheels_on_hhmm_hour': 'wheels_on_hour',
        'wheels_on_hhmm_minute': 'wheels_on_min',
        'crs_arr_time_hhmm_hour': 'crs_arr_hour',
        'crs_arr_time_hhmm_minute': 'crs_arr_min',
        'arr_time_hhmm_hour': 'arr_hour',
        'arr_time_hhmm_minute': 'arr_min'
    }
    df_encoded.rename(columns=rename_dict, inplace=True)

    # Sauvegarde finale
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_encoded.to_csv(output_path, index=False)
    print(f"Données prétraitées enregistrées dans {output_path}")


if __name__== "_main_":
    preprocess_data()