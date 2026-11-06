import json
import pickle
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

import rfm_core as core

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "modele_pipeline.pkl"
REPORTS_DIR = BASE_DIR / "reports"

st.set_page_config(page_title="Segmentation clients RFM", layout="wide")


@st.cache_resource
def charger_modele():
    with open(MODEL_PATH, "rb") as f:
        modele = pickle.load(f)
    segments = {int(k): v for k, v in modele["segments"].items()}
    return modele["pipeline"], segments, modele.get("metadata", {})


try:
    pipeline, segments, metadata = charger_modele()
except Exception as exc:  # fichier absent ou version de scikit-learn incompatible
    st.error(
        "Impossible de charger le modèle. Vérifier que `modele_pipeline.pkl` est présent "
        f"et que la version de scikit-learn est celle de `requirements.txt`. Détail : {exc}"
    )
    st.stop()


def afficher_resultats(data: pd.DataFrame, nom_fichier: str, cle: str) -> None:
    """Affiche indicateurs, profil des segments, graphiques et exports."""
    n_atypiques = int(data["Profil_atypique"].sum())

    k1, k2, k3 = st.columns(3)
    k1.metric("Clients segmentés", f"{len(data):,}".replace(",", " "))
    k2.metric("Segments", data["Segment"].nunique())
    k3.metric("Profils atypiques", n_atypiques)

    if n_atypiques:
        st.warning(
            f"{n_atypiques} client(s) n'ont qu'une commande, d'un montant élevé (colonne "
            "Profil_atypique). Leur segment est à confirmer avant toute action commerciale."
        )

    st.subheader("Profil des segments")
    profil = core.segment_profile(data)
    profil["Action recommandée"] = [core.SEGMENT_ACTIONS.get(s, "") for s in profil.index]
    st.dataframe(profil, width="stretch")

    g1, g2 = st.columns(2)
    effectifs = data["Segment"].value_counts().rename_axis("Segment").reset_index(name="Clients")
    g1.plotly_chart(
        px.bar(effectifs, x="Segment", y="Clients", title="Clients par segment"),
        width="stretch",
        key=f"bar_{cle}",
    )
    ca = profil.reset_index()[["Segment", "CA_total"]]
    g2.plotly_chart(
        px.bar(ca, x="Segment", y="CA_total", title="Chiffre d'affaires par segment (£)"),
        width="stretch",
        key=f"ca_{cle}",
    )

    echantillon = data if len(data) <= 5000 else data.sample(5000, random_state=42)
    st.plotly_chart(
        px.scatter(
            echantillon,
            x="Recency",
            y="Monetary",
            color="Segment",
            log_y=True,
            hover_data=["Frequency"],
            opacity=0.6,
            title="Récence et montant par segment (échelle log sur le montant)",
        ),
        width="stretch",
        key=f"scatter_{cle}",
    )

    st.subheader("Détail et export")
    choix = st.multiselect("Filtrer par segment", sorted(data["Segment"].unique()), key=f"filtre_{cle}")
    vue = data[data["Segment"].isin(choix)] if choix else data
    st.dataframe(vue, width="stretch")
    st.download_button(
        "Télécharger les résultats affichés",
        vue.to_csv(index=False).encode("utf-8"),
        file_name=nom_fichier,
        mime="text/csv",
        key=f"dl_{cle}",
    )


st.title("Segmentation clients (RFM)")
st.write(
    "Cette application attribue un segment marketing à chaque client "
    "à partir de sa Récence, sa Fréquence et son Montant d'achats."
)
st.caption("Projet pédagogique : modèle entraîné sur le jeu de données UCI Online Retail.")

noms_onglets = ["Simuler un client", "Importer un fichier RFM", "Importer des transactions"]
metrics_path = REPORTS_DIR / "metrics.json"
if metrics_path.exists():
    noms_onglets.append("Qualité du modèle")
onglets = st.tabs(noms_onglets)

with onglets[0]:
    st.subheader("Simuler un client")
    c1, c2, c3 = st.columns(3)
    recency = c1.number_input("Récence (jours depuis le dernier achat)", min_value=0, value=30)
    frequency = c2.number_input("Fréquence (nombre de commandes)", min_value=1, value=3)
    monetary = c3.number_input("Montant total dépensé (£)", min_value=0.0, value=500.0)

    if st.button("Prédire le segment"):
        client = pd.DataFrame({"Recency": [recency], "Frequency": [frequency], "Monetary": [monetary]})
        resultat = core.assign_segments(pipeline, segments, client).iloc[0]
        st.success(f"Segment prédit : {resultat['Segment']}")
        action = core.SEGMENT_ACTIONS.get(resultat["Segment"])
        if action:
            st.write(f"Action recommandée : {action}")
        if resultat["Profil_atypique"]:
            st.warning(
                "Une seule commande, d'un montant élevé : ce profil est atypique et le segment "
                "prédit est à confirmer."
            )

with onglets[1]:
    st.subheader("Segmenter un fichier de clients")
    st.write(
        "Le fichier CSV doit contenir les colonnes Recency, Frequency et Monetary. "
        "La colonne CustomerID est facultative mais conservée dans l'export."
    )
    fichier = st.file_uploader("Importer un fichier CSV", type="csv", key="csv_rfm")

    if fichier is not None:
        try:
            data = pd.read_csv(fichier)
        except Exception as exc:
            st.error(f"Lecture du fichier impossible : {exc}")
        else:
            validation = core.validate_rfm_input(data)
            for erreur in validation.errors:
                st.error(erreur)
            for avertissement in validation.warnings:
                st.warning(avertissement)
            if validation.ok:
                data[core.RFM_COLUMNS] = data[core.RFM_COLUMNS].apply(pd.to_numeric)
                afficher_resultats(
                    core.assign_segments(pipeline, segments, data), "clients_segments.csv", "rfm"
                )

with onglets[2]:
    st.subheader("Segmenter à partir des transactions")
    st.write(
        "Importer un fichier de transactions (CSV ou Excel) avec les colonnes : "
        f"{', '.join(core.TRANSACTION_COLUMNS)}. Le RFM est calculé automatiquement, "
        "avec le lendemain de la dernière transaction comme date de référence."
    )
    fichier_tr = st.file_uploader("Importer les transactions", type=["csv", "xlsx"], key="transactions")

    if fichier_tr is not None:
        try:
            if fichier_tr.name.lower().endswith(".xlsx"):
                brut = pd.read_excel(fichier_tr)
            else:
                brut = pd.read_csv(fichier_tr)
        except Exception as exc:
            st.error(f"Lecture du fichier impossible : {exc}")
        else:
            manquantes = core.check_transaction_columns(brut)
            if manquantes:
                st.error(f"Colonnes manquantes : {', '.join(manquantes)}.")
            else:
                try:
                    propres, rapport = core.clean_transactions(brut)
                    if propres.empty:
                        st.error("Aucune transaction valide après nettoyage.")
                    else:
                        rfm = core.compute_rfm(propres)
                        with st.expander("Rapport de nettoyage"):
                            st.json(rapport)
                        afficher_resultats(
                            core.assign_segments(pipeline, segments, rfm),
                            "clients_segments.csv",
                            "transactions",
                        )
                except Exception as exc:
                    st.error(f"Traitement impossible : {exc}")

if metrics_path.exists():
    with onglets[3]:
        st.subheader("Qualité du modèle")
        metriques = json.loads(metrics_path.read_text(encoding="utf-8"))
        stabilite = metriques.get("stabilite", {})
        scores = metriques.get("scores_k_choisi", {})
        m1, m2, m3 = st.columns(3)
        m1.metric("Silhouette", f"{scores.get('Silhouette', float('nan')):.3f}")
        m2.metric("Davies-Bouldin", f"{scores.get('Davies_Bouldin', float('nan')):.3f}")
        m3.metric("Stabilité (ARI moyen)", f"{stabilite.get('ari_moyen', float('nan')):.3f}")
        selection_path = REPORTS_DIR / "selection_k.csv"
        if selection_path.exists():
            selection = pd.read_csv(selection_path)
            st.plotly_chart(
                px.line(selection, x="k", y="Silhouette", markers=True, title="Silhouette selon k"),
                width="stretch",
                key="qualite_silhouette",
            )
            st.dataframe(selection, width="stretch")
