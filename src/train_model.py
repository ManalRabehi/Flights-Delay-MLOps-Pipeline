# Script d'entraînement du modèle 

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

#Chargement et préparation

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

def split_data(X, y):
    return train_test_split(X, y, test_size=0.25, random_state=42)


# Entraînement du modèle de base

def train_base_model(X_train, y_train, X_test, y_test):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    acc = rf.score(X_test, y_test)
    print(f"Accuracy before feature selection: {acc:.2f}")
    return rf


# Sélection de features importantes

def select_features(rf, X_train):
    feature_importances = pd.DataFrame({
        'feature': X_train.columns,
        'importance': rf.feature_importances_
    }).sort_values(by='importance', ascending=False)
    selected_features = feature_importances[feature_importances['importance'] >  0.01]['feature']
    return selected_features, feature_importances


# Optimisation via GridSearch

def tune_hyperparameters(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [12, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }
    grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1
    )
    grid.fit(X_train, y_train)
    print("Best parameters:", grid.best_params_)
    print("Best accuracy:", grid.best_score_)
    return grid.best_params_

#Entraînement final

def train_final_model(X_train, y_train, params):
    final_model = RandomForestClassifier(
        **params, random_state=42, n_jobs=-1
    )
    final_model.fit(X_train, y_train)
    return final_model

