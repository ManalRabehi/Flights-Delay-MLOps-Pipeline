# Script d'entraînement du modèle 

import pandas as pd
import lightgbm as lgb
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from imblearn.over_sampling import SMOTE

# Chargement et préparation

def load_data(path="../data/processed/processed_data.csv"):
    df = pd.read_csv(path)
    post_flight_features = [
        'dep_delay', 'nas_delay', 'carrier_delay', 'late_aircraft_delay',
        'taxi_out', 'taxi_in', 'weather_delay', 'air_time',
        'actual_elapsed_time', 'arr_hour', 'arr_min',
        'wheels_on_hour', 'wheels_on_min', 'wheels_off_hour', 'wheels_off_min',
        'security_delay', 'diverted', 'cancelled'
    ]
    df_preflight = df.drop(columns=post_flight_features)
    X = df_preflight.drop(columns=["arr_delay"], axis=1)
    y = (df_preflight["arr_delay"] > 0).astype(int)
    return X, y

# Séparation des données

def split_data(X, y, smote=False):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    if smote:
        sm = SMOTE(random_state=42)
        X_train, y_train = sm.fit_resample(X_train, y_train)
    return X_train, X_test, y_train, y_test

# Sélection de features importantes

def select_features(model, X_train, threshold=0.01):
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        "feature" : model.feature_name_,
        "importance" : importances / importances.sum()
    }).sort_values(by="importance", ascending=False)
    selected_features = feature_importance_df[feature_importance_df["importance"] > threshold]["feature"].tolist()
    return selected_features, feature_importance_df

# Recherche d'hyperparamètres

def tune_hyperparameters(X_train, y_train, n_iter=30):
    param_grid = {
        'n_estimators': [100, 300, 500],
        'max_depth': [8, 12, 16, -1],
        'learning_rate': [0.01, 0.05, 0.1],
        'num_leaves': [15, 31, 63],
        'min_child_samples': [10, 20, 50],
        'subsample': [0.7, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.9, 1.0]
    }
    lgb_model = lgb.LGBMClassifier(objective='binary', random_state=42, class_weight='balanced')
    grid_search = RandomizedSearchCV(
        estimator=lgb_model,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring='recall',
        cv=3,
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)
    print("Best parameters:", grid_search.best_params_)
    print("Best recall:", grid_search.best_score_)
    return grid_search.best_params_

# Entraînement final

def train_final_model(X_train, y_train, param=None):
    if params is None:
        params = {
            'subsample': 1.0,
            'num_leaves': 31,
            'n_estimators': 500,
            'min_child_samples': 20,
            'max_depth': -1,
            'learning_rate': 0.01,
            'colsample_bytree': 1.0,
            'class_weight': 'balanced',
            'random_state': 42
        }
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)
    return model

# Sauvegarde

def save_model(model, path="../models/lightgbm_model.pkl"):
    joblib.dump(model, path)
    print(f"Modèle sauvegardé dans {path}")

def save_features(features, path="../models/selected_features_LGB.pkl"):
    joblib.dump(features, path)
    print(f"Liste des features sauvegardée dans {path}")



