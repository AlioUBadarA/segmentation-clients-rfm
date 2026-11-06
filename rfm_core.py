"""Fonctions métier du projet de segmentation clients RFM.

Ce module regroupe la logique partagée par le script d'entraînement (train.py),
l'application Streamlit (app.py) et les tests : nettoyage des transactions,
calcul du RFM, construction du pipeline, choix du nombre de clusters,
nommage des segments et validation des fichiers importés.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

RFM_COLUMNS = ["Recency", "Frequency", "Monetary"]
TRANSACTION_COLUMNS = ["InvoiceNo", "CustomerID", "InvoiceDate", "Quantity", "UnitPrice"]
RANDOM_STATE = 42

# Seuil (en ecarts-types, echelle log) au-dela duquel le montant d'un client
# a une seule commande est juge atypique. Heuristique a ajuster selon les donnees.
ATYPICAL_Z_MONETARY = 1.0

SEG_VIP = "Clients VIP"
SEG_FIDELES = "Clients fidèles"
SEG_NOUVEAUX = "Nouveaux clients"
SEG_CHURN = "Clients à risque (churn)"

# Actions marketing proposees par segment (cles = noms utilises dans le modele).
SEGMENT_ACTIONS = {
    SEG_VIP: "Programme de fidélité premium, accès anticipé aux nouveautés, relation dédiée.",
    SEG_FIDELES: "Ventes croisées, offres de montée en gamme, programme de parrainage.",
    SEG_NOUVEAUX: "Séquence d'accueil, incitation à la deuxième commande, recommandations de produits.",
    SEG_CHURN: "Campagne de réactivation, offre personnalisée limitée dans le temps, enquête de satisfaction.",
}


# ---------------------------------------------------------------------------
# Donnees : lecture, nettoyage, RFM
# ---------------------------------------------------------------------------

def load_transactions(path: str | Path) -> pd.DataFrame:
    """Lit un fichier de transactions (.xlsx, .xls ou .csv)."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    elif suffix == ".csv":
        df = pd.read_csv(path, parse_dates=["InvoiceDate"])
    else:
        raise ValueError(f"Format non pris en charge : {suffix} (utiliser .xlsx ou .csv)")
    return df


def check_transaction_columns(df: pd.DataFrame) -> list[str]:
    """Retourne la liste des colonnes de transactions manquantes."""
    return [c for c in TRANSACTION_COLUMNS if c not in df.columns]


def clean_transactions(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Nettoie les transactions et retourne (donnees propres, rapport).

    Etapes identiques a celles du notebook : suppression des lignes sans client,
    des annulations (factures commencant par C), des quantites ou prix non
    positifs et des doublons.
    """
    manquantes = check_transaction_columns(df)
    if manquantes:
        raise ValueError(f"Colonnes manquantes : {', '.join(manquantes)}")

    report = {"lignes_brutes": int(len(df))}
    out = df.dropna(subset=["CustomerID"])
    report["apres_clients_manquants"] = int(len(out))

    out = out[~out["InvoiceNo"].astype(str).str.startswith("C")]
    report["apres_annulations"] = int(len(out))

    out = out[(out["Quantity"] > 0) & (out["UnitPrice"] > 0)]
    report["apres_valeurs_invalides"] = int(len(out))

    out = out.drop_duplicates().copy()
    report["apres_doublons"] = int(len(out))

    out["CustomerID"] = out["CustomerID"].astype(int)
    out["InvoiceDate"] = pd.to_datetime(out["InvoiceDate"])
    out["TotalPrice"] = out["Quantity"] * out["UnitPrice"]
    report["clients"] = int(out["CustomerID"].nunique())
    return out, report


def compute_rfm(
    df: pd.DataFrame, reference_date: pd.Timestamp | dt.datetime | None = None
) -> pd.DataFrame:
    """Calcule le tableau RFM (une ligne par client) a partir de transactions propres.

    La date de reference par defaut est le lendemain de la derniere transaction.
    """
    if "TotalPrice" not in df.columns:
        df = df.assign(TotalPrice=df["Quantity"] * df["UnitPrice"])
    if reference_date is None:
        reference_date = df["InvoiceDate"].max() + dt.timedelta(days=1)
    reference_date = pd.Timestamp(reference_date)

    grouped = df.groupby("CustomerID")
    rfm = pd.DataFrame(
        {
            "Recency": (reference_date - grouped["InvoiceDate"].max()).dt.days,
            "Frequency": grouped["InvoiceNo"].nunique(),
            "Monetary": grouped["TotalPrice"].sum(),
        }
    ).reset_index()
    return rfm


# ---------------------------------------------------------------------------
# Modele
# ---------------------------------------------------------------------------

def build_pipeline(k: int = 4, random_state: int = RANDOM_STATE) -> Pipeline:
    """Pipeline identique a celui du notebook : log1p, normalisation, K-Means."""
    return Pipeline(
        [
            ("log", FunctionTransformer(np.log1p)),
            ("scaler", StandardScaler()),
            ("kmeans", KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=random_state)),
        ]
    )


def scale_features(rfm: pd.DataFrame) -> np.ndarray:
    """Applique log1p puis la standardisation (sans K-Means)."""
    prep = build_pipeline()[:-1]
    return prep.fit_transform(rfm[RFM_COLUMNS])


def evaluate_k(
    rfm: pd.DataFrame,
    k_range: range = range(2, 11),
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Calcule inertie, silhouette, Davies-Bouldin et Calinski-Harabasz pour chaque k."""
    X = scale_features(rfm)
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=random_state)
        labels = km.fit_predict(X)
        rows.append(
            {
                "k": k,
                "Inertie": km.inertia_,
                "Silhouette": silhouette_score(X, labels),
                "Davies_Bouldin": davies_bouldin_score(X, labels),
                "Calinski_Harabasz": calinski_harabasz_score(X, labels),
            }
        )
    return pd.DataFrame(rows)


def stability_score(
    rfm: pd.DataFrame,
    k: int = 4,
    n_runs: int = 20,
    sample_frac: float = 0.8,
    random_state: int = RANDOM_STATE,
) -> dict:
    """Mesure la stabilite des clusters par sous-echantillonnage.

    Pour chaque tirage, K-Means est ajuste sur une fraction des clients puis
    applique a tous les clients. Le resultat est compare a la partition de
    reference avec l'indice de Rand ajuste (1 = partitions identiques).
    """
    X = scale_features(rfm)
    reference = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=random_state).fit_predict(X)
    rng = np.random.default_rng(random_state)
    scores = []
    for i in range(n_runs):
        idx = rng.choice(len(X), size=int(len(X) * sample_frac), replace=False)
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=random_state + i + 1)
        km.fit(X[idx])
        scores.append(adjusted_rand_score(reference, km.predict(X)))
    return {
        "ari_moyen": float(np.mean(scores)),
        "ari_ecart_type": float(np.std(scores)),
        "ari_min": float(np.min(scores)),
        "n_runs": n_runs,
        "sample_frac": sample_frac,
    }


def name_segments(rfm_with_cluster: pd.DataFrame) -> dict[int, str]:
    """Attribue un nom metier a chaque cluster d'apres les moyennes RFM.

    La regle (k = 4) est celle du notebook : VIP = montant moyen le plus eleve,
    a risque = recence moyenne la plus elevee, nouveaux = recence la plus faible
    parmi les deux restants, fideles = le dernier. Si la regle est ambigue ou si
    k est different de 4, des noms generiques sont utilises.
    """
    profil = rfm_with_cluster.groupby("Cluster")[RFM_COLUMNS].mean()
    clusters = [int(c) for c in profil.index]
    if len(clusters) != 4:
        return {c: f"Segment {c}" for c in clusters}

    vip = int(profil["Monetary"].idxmax())
    churn = int(profil["Recency"].idxmax())
    if vip == churn:
        return {c: f"Segment {c}" for c in clusters}

    reste = profil.drop([vip, churn])
    nouveaux = int(reste["Recency"].idxmin())
    fideles = int(next(c for c in reste.index if c != nouveaux))
    return {
        vip: SEG_VIP,
        churn: SEG_CHURN,
        nouveaux: SEG_NOUVEAUX,
        fideles: SEG_FIDELES,
    }


def segment_profile(rfm: pd.DataFrame, segment_col: str = "Segment") -> pd.DataFrame:
    """Profil par segment : effectif, part des clients, moyennes RFM, CA et part du CA."""
    grouped = rfm.groupby(segment_col)
    profil = pd.DataFrame(
        {
            "Nb_clients": grouped.size(),
            "Recency_moy": grouped["Recency"].mean(),
            "Frequency_moy": grouped["Frequency"].mean(),
            "Monetary_moy": grouped["Monetary"].mean(),
            "CA_total": grouped["Monetary"].sum(),
        }
    )
    profil["Part_clients_%"] = profil["Nb_clients"] / profil["Nb_clients"].sum() * 100
    profil["Part_CA_%"] = profil["CA_total"] / profil["CA_total"].sum() * 100
    return profil.sort_values("CA_total", ascending=False).round(1)


# ---------------------------------------------------------------------------
# Utilisation du modele (application et scripts)
# ---------------------------------------------------------------------------

def flag_atypical(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Signale les clients a commande unique et montant eleve.

    Ces profils (un seul gros achat) peuvent etre classes dans un segment qui ne
    reflete pas leur comportement reel. Le signalement est une aide a la decision,
    pas une correction du modele.
    """
    scaled = np.asarray(pipeline[:-1].transform(X[RFM_COLUMNS]))
    z_monetary = scaled[:, RFM_COLUMNS.index("Monetary")]
    return ((X["Frequency"].to_numpy() <= 1) & (z_monetary >= ATYPICAL_Z_MONETARY)).astype(bool)


def assign_segments(pipeline: Pipeline, segments: dict, rfm: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les colonnes Segment et Profil_atypique a un tableau RFM."""
    X = rfm[RFM_COLUMNS].astype(float)
    clusters = pipeline.predict(X)
    segments = {int(k): v for k, v in segments.items()}
    out = rfm.copy()
    out["Segment"] = [segments[int(c)] for c in clusters]
    out["Profil_atypique"] = flag_atypical(pipeline, X)
    return out


# ---------------------------------------------------------------------------
# Validation des fichiers importes
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_rfm_input(df: pd.DataFrame) -> ValidationResult:
    """Verifie un tableau RFM importe : colonnes, types, valeurs manquantes, negatifs."""
    result = ValidationResult()
    manquantes = [c for c in RFM_COLUMNS if c not in df.columns]
    if manquantes:
        result.errors.append(f"Colonnes manquantes : {', '.join(manquantes)}.")
        return result

    if df.empty:
        result.errors.append("Le fichier ne contient aucune ligne.")
        return result

    numeriques = df[RFM_COLUMNS].apply(pd.to_numeric, errors="coerce")
    for col in RFM_COLUMNS:
        n_invalides = int(numeriques[col].isna().sum())
        if n_invalides:
            result.errors.append(
                f"{n_invalides} valeur(s) vide(s) ou non numérique(s) dans la colonne {col}."
            )
    if result.errors:
        return result

    if (numeriques < 0).any().any():
        result.errors.append("Les colonnes Recency, Frequency et Monetary ne doivent pas contenir de valeurs négatives.")
        return result

    if "CustomerID" in df.columns and df["CustomerID"].duplicated().any():
        n = int(df["CustomerID"].duplicated().sum())
        result.warnings.append(f"{n} identifiant(s) client en double dans le fichier.")
    if (numeriques["Frequency"] == 0).any():
        result.warnings.append("Certains clients ont une fréquence de 0 : vérifier la source des données.")
    return result
