import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import sklearn

import rfm_core as core

MODEL_PATH = Path(__file__).resolve().parent.parent / "modele_pipeline.pkl"


# --- Nettoyage et calcul du RFM -------------------------------------------

def test_clean_transactions_removes_invalid_rows(transactions, dirty_transactions):
    clean, report = core.clean_transactions(dirty_transactions)
    assert len(clean) == len(transactions)
    assert report["lignes_brutes"] == len(dirty_transactions)
    assert (clean["Quantity"] > 0).all() and (clean["UnitPrice"] > 0).all()
    assert not clean["InvoiceNo"].astype(str).str.startswith("C").any()
    assert clean["CustomerID"].dtype.kind == "i"


def test_clean_transactions_rejects_missing_columns():
    with pytest.raises(ValueError, match="Colonnes manquantes"):
        core.clean_transactions(pd.DataFrame({"InvoiceNo": ["1"]}))


def test_compute_rfm_exact_values():
    df = pd.DataFrame(
        {
            "InvoiceNo": ["1", "2", "2", "3"],
            "CustomerID": [1, 1, 1, 2],
            "InvoiceDate": pd.to_datetime(["2011-12-01", "2011-12-05", "2011-12-05", "2011-11-30"]),
            "Quantity": [2, 1, 3, 1],
            "UnitPrice": [10.0, 5.0, 2.0, 100.0],
        }
    )
    rfm = core.compute_rfm(df, reference_date="2011-12-10").set_index("CustomerID")
    assert rfm.loc[1, "Recency"] == 5
    assert rfm.loc[1, "Frequency"] == 2          # factures distinctes
    assert rfm.loc[1, "Monetary"] == pytest.approx(20 + 5 + 6)
    assert rfm.loc[2, "Recency"] == 10
    assert rfm.loc[2, "Monetary"] == pytest.approx(100)


def test_compute_rfm_default_reference_is_day_after_last_transaction():
    df = pd.DataFrame(
        {
            "InvoiceNo": ["1"],
            "CustomerID": [1],
            "InvoiceDate": pd.to_datetime(["2011-12-09"]),
            "Quantity": [1],
            "UnitPrice": [1.0],
        }
    )
    assert core.compute_rfm(df)["Recency"].iloc[0] == 1


# --- Pipeline et sélection de k ---------------------------------------------

def test_pipeline_is_deterministic(rfm):
    a = core.build_pipeline(4).fit(rfm[core.RFM_COLUMNS]).predict(rfm[core.RFM_COLUMNS])
    b = core.build_pipeline(4).fit(rfm[core.RFM_COLUMNS]).predict(rfm[core.RFM_COLUMNS])
    assert (a == b).all()


def test_evaluate_k_returns_all_metrics(rfm):
    res = core.evaluate_k(rfm, range(2, 6))
    assert list(res["k"]) == [2, 3, 4, 5]
    for col in ["Inertie", "Silhouette", "Davies_Bouldin", "Calinski_Harabasz"]:
        assert res[col].notna().all()
    assert res["Inertie"].is_monotonic_decreasing


def test_stability_score_in_valid_range(rfm):
    s = core.stability_score(rfm, k=4, n_runs=5)
    assert -1 <= s["ari_min"] <= s["ari_moyen"] <= 1
    assert s["ari_moyen"] > 0.5  # les profils synthétiques sont bien séparés


# --- Nommage et profil des segments ------------------------------------------

def _fitted_with_names(rfm):
    pipeline = core.build_pipeline(4).fit(rfm[core.RFM_COLUMNS])
    data = rfm.copy()
    data["Cluster"] = pipeline.predict(rfm[core.RFM_COLUMNS])
    return pipeline, data, core.name_segments(data)


def test_name_segments_assigns_the_four_roles(rfm):
    _, data, names = _fitted_with_names(rfm)
    assert sorted(names.values()) == sorted(
        [core.SEG_VIP, core.SEG_FIDELES, core.SEG_NOUVEAUX, core.SEG_CHURN]
    )
    means = data.groupby("Cluster")[core.RFM_COLUMNS].mean()
    assert names[means["Monetary"].idxmax()] == core.SEG_VIP
    assert names[means["Recency"].idxmax()] == core.SEG_CHURN


def test_name_segments_falls_back_when_k_is_not_four(rfm):
    pipeline = core.build_pipeline(3).fit(rfm[core.RFM_COLUMNS])
    data = rfm.assign(Cluster=pipeline.predict(rfm[core.RFM_COLUMNS]))
    names = core.name_segments(data)
    assert set(names.values()) == {"Segment 0", "Segment 1", "Segment 2"}


def test_segment_profile_shares_sum_to_100(rfm):
    _, data, names = _fitted_with_names(rfm)
    data["Segment"] = data["Cluster"].map(names)
    profil = core.segment_profile(data)
    assert profil["Part_clients_%"].sum() == pytest.approx(100, abs=0.5)
    assert profil["Part_CA_%"].sum() == pytest.approx(100, abs=0.5)
    assert profil["Nb_clients"].sum() == len(data)


def test_every_segment_has_a_marketing_action():
    for seg in [core.SEG_VIP, core.SEG_FIDELES, core.SEG_NOUVEAUX, core.SEG_CHURN]:
        assert core.SEGMENT_ACTIONS[seg]


# --- Utilisation du modèle ------------------------------------------------------

def test_assign_segments_and_atypical_flag(rfm):
    pipeline, data, names = _fitted_with_names(rfm)
    clients = pd.DataFrame(
        {
            "Recency": [5, 300, 10],
            "Frequency": [20, 1, 1],
            "Monetary": [9000.0, 50.0, 8000.0],
        }
    )
    out = core.assign_segments(pipeline, names, clients)
    assert out.loc[0, "Segment"] == core.SEG_VIP
    assert out.loc[1, "Segment"] == core.SEG_CHURN
    assert list(out["Profil_atypique"]) == [False, False, True]   # un seul gros achat


# --- Validation des imports ------------------------------------------------------

def test_validate_rfm_input_ok_and_warning():
    df = pd.DataFrame(
        {"CustomerID": [1, 1], "Recency": [3, 4], "Frequency": [2, 0], "Monetary": [10.0, 20.0]}
    )
    res = core.validate_rfm_input(df)
    assert res.ok
    assert len(res.warnings) == 2   # identifiant en double + fréquence de 0


@pytest.mark.parametrize(
    "df, fragment",
    [
        (pd.DataFrame({"Recency": [1]}), "Colonnes manquantes"),
        (pd.DataFrame({"Recency": [], "Frequency": [], "Monetary": []}), "aucune ligne"),
        (pd.DataFrame({"Recency": [1, None], "Frequency": [1, 1], "Monetary": [1, 1]}), "vide"),
        (pd.DataFrame({"Recency": ["a"], "Frequency": [1], "Monetary": [1]}), "non numérique"),
        (pd.DataFrame({"Recency": [-1], "Frequency": [1], "Monetary": [1]}), "négatives"),
    ],
)
def test_validate_rfm_input_errors(df, fragment):
    res = core.validate_rfm_input(df)
    assert not res.ok
    assert any(fragment in e for e in res.errors)


# --- Modèle livré dans le dépôt -----------------------------------------------------

@pytest.mark.skipif(sklearn.__version__ != "1.6.1", reason="le .pkl exige scikit-learn 1.6.1")
def test_shipped_model_matches_readme_profiles():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    clients = pd.DataFrame(
        {"Recency": [10, 300], "Frequency": [15, 1], "Monetary": [9000.0, 100.0]}
    )
    out = core.assign_segments(model["pipeline"], model["segments"], clients)
    assert out.loc[0, "Segment"] == core.SEG_VIP
    assert out.loc[1, "Segment"] == core.SEG_CHURN
    assert set(model["segments"].values()) == set(core.SEGMENT_ACTIONS)
