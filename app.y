import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Segmentation clients RFM", layout="wide")


@st.cache_resource
def charger_modele():
    with open("modele_pipeline.pkl", "rb") as f:
        return pickle.load(f)


modele = charger_modele()
pipeline = modele["pipeline"]
segments = modele["segments"]

st.title("Segmentation clients (RFM)")
st.write(
    "Cette application attribue un segment marketing à chaque client "
    "à partir de sa Récence, sa Fréquence et son Montant d'achats."
)
st.caption("Projet pédagogique : modèle entraîné sur le jeu de données UCI Online Retail.")

onglet1, onglet2 = st.tabs(["Simuler un client", "Importer un CSV"])

with onglet1:
    st.subheader("Simuler un client")
    c1, c2, c3 = st.columns(3)
    recency = c1.number_input("Récence (jours depuis le dernier achat)", min_value=0, value=30)
    frequency = c2.number_input("Fréquence (nombre de commandes)", min_value=1, value=3)
    monetary = c3.number_input("Montant total dépensé (£)", min_value=0.0, value=500.0)

    if st.button("Prédire le segment"):
        X = pd.DataFrame({
            "Recency": [recency],
            "Frequency": [frequency],
            "Monetary": [monetary],
        })
        cluster = pipeline.predict(X)[0]
        st.success(f"Segment prédit : {segments[cluster]}")

with onglet2:
    st.subheader("Segmenter un fichier de clients")
    st.write("Le fichier CSV doit contenir les colonnes : CustomerID, Recency, Frequency, Monetary.")

    fichier = st.file_uploader("Importer un fichier CSV", type="csv")

    if fichier is not None:
        data = pd.read_csv(fichier)
        colonnes = ["Recency", "Frequency", "Monetary"]
        manquantes = [c for c in colonnes if c not in data.columns]

        if manquantes:
            st.error(f"Colonnes manquantes : {', '.join(manquantes)}")
        elif (data[colonnes] < 0).any().any():
            st.error("Les colonnes Recency, Frequency et Monetary ne doivent pas contenir de valeurs négatives.")
        else:
            clusters = pipeline.predict(data[colonnes])
            data["Segment"] = [segments[c] for c in clusters]

            st.dataframe(data, use_container_width=True)
            st.bar_chart(data["Segment"].value_counts())

            st.download_button(
                "Télécharger les résultats",
                data.to_csv(index=False).encode("utf-8"),
                file_name="clients_segments.csv",
                mime="text/csv",
            )
