import streamlit as st
import requests
import json

st.title("Scoring Client – API MLflow Registry")

API_URL = "http://localhost:8000/predict/"  # garde pour l'envoi

# === Fonction pour obtenir dynamiquement les features depuis l'API ===
def charger_features():
    try:
        res = requests.get("http://localhost:8000/features/")
        if res.status_code == 200:
            return res.json().get("features", [])
        else:
            st.error(f"Erreur lors de la récupération des features : {res.text}")
            return []
    except Exception as e:
        st.error(f"Erreur lors de la récupération des features : {e}")
        return []

# === Récupère dynamiquement les features ===
features = charger_features()

st.header("Renseigne les caractéristiques du client :")
input_data = {}
for feat in features:
    input_data[feat] = st.text_input(feat, "0")

# === Envoi vers l'API ===
if st.button("Lancer le scoring !"):
    payload = {"data": {k: float(v) for k, v in input_data.items()}}
    try:
        res = requests.post(API_URL, json=payload)
        if res.status_code == 200:
            st.success("✅ Résultat du scoring :")
            st.json(res.json())
        else:
            st.error(f"Erreur {res.status_code} : {res.text}")
    except Exception as e:
        st.error(f"Erreur lors de l'appel API : {e}")

