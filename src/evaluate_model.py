import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import json
import os

def evaluate_model(model, X_test, y_test, threshold=0.5):
    """
    Évalue le modèle LightGBM et sauvegarde les métriques dans un fichier JSON.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob > threshold).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print("Confusion Matrix:\n", cm)

    metrics_dict = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall
    }

    # Crée le dossier metrics si nécessaire
    os.makedirs("metrics", exist_ok=True)
    with open("metrics/metrics.json", "w") as f:
        json.dump(metrics_dict, f, indent=4)

    return accuracy, precision, recall, cm

if __name__ == "__main__":
    model = joblib.load("models/lightgbm_model.pkl")
    selected_features = joblib.load("models/selected_features_LGB.pkl")
    df = pd.read_csv("data/processed/processed_data.csv")
    
    y_test = (df["arr_delay"] > 0).astype(int)
    X_test = df.drop(columns=["arr_delay"])
    
    # Ajouter les colonnes manquantes et réordonner
    for col in selected_features:
        if col not in X_test.columns:
            X_test[col] = 0
    X_test = X_test[selected_features]
    
    evaluate_model(model, X_test, y_test)