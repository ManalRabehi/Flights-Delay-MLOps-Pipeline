# Flight Delay Prediction Pipeline

## Introduction
Ce projet implémente un pipeline MLOps complet pour prédire les retards de vols américains. L’objectif est de passer d’un dataset brut à une interface utilisateur fonctionnelle, en garantissant la reproductibilité via DVC et une automatisation via GitHub Actions.
---

## Description du projet
L’objectif principal est de :

- Collecter et préparer un dataset public de vols
- Entraîner un modèle de classification 
- Évaluer les performances du modèle
- Versionner les données et les modèles avec DVC
- Automatiser le pipeline avec CI/CD
- Déployer une API avec FastAPI pour faire des prédictions
- Mettre en place un monitoring simple des prédictions

## Architecture techinique: 
- Modèle : LightGBM
- Versioning : Git pour le code et DVC pour les données (.csv) et les modèles (.pkl)
- Pipeline DVC : data_preparation.py ➔ train_model.py ➔ evaluate_model.py
- API : FastAPI avec validation de données via Pydantic
- Interface : Streamlit pour une saisie utilisateur simplifiée

Architecture simplifiée :

Dataset → Préprocessing → Modèle ML → API FastAPI → Logs / Monitoring

## Organisation du projet

### Dossier `data`
Contient les datasets bruts ainsi que les données après nettoyage, transformation ou encodage.

### Dossier `notebooks`
- 01_data_preparation : préparation des données avec étapes détaillées et visualisations
- 02_rf_feature_selection : sélection des features et entraînement d’un modèle Random Forest avec analyse des résultats
- 03_lightGBM_feature_selection : sélection des features et entraînement d’un modèle LightGBM avec analyse
- 04_model_comparison:  comparaison des deux modèles et justification du choix final
- 05_model_analysis : analyse statistique des performances du modèle retenu
- 06_test_new_data : tests du modèle sur de nouvelles données avec évaluation des performances

### Dossier `models`
Contient les fichiers au format `.pkl` (modèle entraîné, encodeurs, scaler, features sélectionnées).

### Dossier `metrics`
Fichier `metrics.json` contenant les résultats et performances du modèle final.

### Dossier `src`
Contient les scripts reproductibles issus des notebooks :
- data_preparation.py
- train_model.py
- evaluate_model.py
- analyze_model.py
- inspectpkl.py
- main.py : API FastAPI, gestion des prédictions, stockage en base de données et connexion avec l’interface

---

## Structure de repo

Flights-Delay-MLOps-Pipeline/
│
├── data/
│ ├── raw/
│ └── processed/
│
├── models/
│ └── model.pkl
│
├── notebooks/
│ └──  01_data_preparation.ipynb
│ ├──  02_rf_feature_selection.ipynb
│ ├──  03_lightGBM_feature_selection.ipynb
│ ├──  04_model_comparison.ipynb
│ ├──  05_model_analysis.ipynb
│ ├──  06_test_new_data.ipynb
│
├── src/
│ ├── main.py
│ ├── data_preparation.py
│ ├── train_model.py
│ ├── evaluate_model.py
│ ├── utils.py
│ └── database.py
│
├── baseDonnees/
│ └── predictions.db
│
├── app.py
├── dvc.yaml
├── params.yaml
├── requirements.txt
├── .gitignore
├── README.md
└── .github/
└── workflows/



## Pré-requis 

### Environnement
- Python : 3.9+ ou 3.13
- DVC: Initialisé pour la CI/CD
- Conda : Environnement flights-mlops
- Git

## Libraries
scikit-learn
imbalanced-learn
pandas
numpy
fastapi
uvicorn
pydantic
joblib
streamlit
requests
seaborn

### Installation

```bash
git clone https://github.com/ManalRabehi/Flights-Delay-MLOps-Pipeline.git
cd Flights-Delay-MLOps-Pipeline
pip install -r requirements.txt
```

## Lancer et executer  le projet
Dans deux terminaux differents lancer le commands suivantes:

- streamlit run app.py
- uvicorn src.main:app --reload

À chaque demande de prédiction, les données sont automatiquement enregistrées dans la base de données.
Pour télécharger l’historique des prédictions sous forme de fichier CSV, accéder à :

http://127.0.0.1:8000/download-logs


## Resultats

A remplir avec les images

### Sécurité
Pour la protection de l’application, des mesures contre les injections SQL et les attaques XSS ont été mises en place.