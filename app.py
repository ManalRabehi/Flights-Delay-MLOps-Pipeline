import streamlit as st
import requests
from datetime import date
import re

st.set_page_config(page_title="Prédiction Retard de Vol ✈️", layout="centered")

st.title("🛫 Prédiction de retard de vol")
st.markdown(
    """
Cette application prédit si un vol sera en retard, ainsi que la probabilité du retard.
Remplissez les informations du vol ci-dessous et cliquez sur **Prédire le retard**.

⚠️ **Important :** Pour les villes, utilisez le format `City, ST`, par exemple `New York, NY`.
"""
)

# ======================
# Formulaire des informations du vol
# ======================
with st.form("flight_form"):
    st.subheader("Informations du vol")

    col1, col2 = st.columns(2)
    with col1:
        flight_date = st.date_input("Date du vol", value=date.today())
        op_carrier_fl_num = st.number_input("Numéro du vol", min_value=1, step=1)
        origin_city_name = st.text_input("Ville de départ", "New York, NY")
        origin_state_nm = st.text_input("État de départ", "New York")
    with col2:
        dest_city_name = st.text_input("Ville d'arrivée", "Los Angeles, CA")
        dest_state_nm = st.text_input("État d'arrivée", "California")
        crs_dep_hour = st.number_input("Heure départ (0-23)", min_value=0, max_value=23)
        crs_dep_min = st.number_input("Minute départ (0-59)", min_value=0, max_value=59)
        crs_arr_hour = st.number_input("Heure arrivée (0-23)", min_value=0, max_value=23)
        crs_arr_min = st.number_input("Minute arrivée (0-59)", min_value=0, max_value=59)

    submitted = st.form_submit_button("Prédire le retard")

# ======================
# Fonction de validation
# ======================
def check_city_format(city: str) -> bool:
    """Vérifie si le format est 'City, ST' (ex: New York, NY)"""
    pattern = r"^[A-Za-z\s]+,\s?[A-Z]{2}$"
    return bool(re.match(pattern, city.strip()))

# ======================
# Action lorsque le formulaire est soumis
# ======================
if submitted:
    # Vérification du format des villes
    if not check_city_format(origin_city_name):
        st.error("⚠️ La ville de départ doit être au format 'City, ST', ex: 'New York, NY'.")
    elif not check_city_format(dest_city_name):
        st.error("⚠️ La ville d'arrivée doit être au format 'City, ST', ex: 'Los Angeles, CA'.")
    else:
        url = "http://127.0.0.1:8000/predict"
        payload = {
            "flight_date": flight_date.strftime("%Y-%m-%d"),
            "op_carrier_fl_num": op_carrier_fl_num,
            "origin_city_name": origin_city_name,
            "origin_state_nm": origin_state_nm,
            "dest_city_name": dest_city_name,
            "dest_state_nm": dest_state_nm,
            "crs_dep_hour": crs_dep_hour,
            "crs_dep_min": crs_dep_min,
            "crs_arr_hour": crs_arr_hour,
            "crs_arr_min": crs_arr_min
        }

        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            result = response.json()

            st.success("✅ Prédiction obtenue !")
            st.write(f"**Retard prévu :** {'Oui' if result['delay_predicted'] else 'Non'}")
            st.write(f"**Probabilité de retard :** {result['delay_probability']*100:.1f}%")

        except requests.exceptions.RequestException as e:
            st.error(f"Impossible de contacter l'API : {e}")
