import joblib

# Charger le fichier .pkl
selected_features = joblib.load("models/selected_features_LGB.pkl")

# Afficher le contenu
print("Nombre de features sélectionnées :", len(selected_features))
print("Features sélectionnées :")
for f in selected_features:
    print("-", f)