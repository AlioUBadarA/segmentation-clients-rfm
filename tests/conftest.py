import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _make_transactions(n_customers: int = 400, seed: int = 0) -> pd.DataFrame:
    """Transactions synthétiques avec quatre profils de clients distincts."""
    rng = np.random.default_rng(seed)
    profiles = [
        # (poids, jours depuis dernier achat, nb commandes, montant par commande)
        (0.15, (1, 25), (8, 20), (300, 900)),    # gros acheteurs récents
        (0.25, (30, 110), (3, 7), (120, 300)),   # réguliers
        (0.20, (1, 40), (1, 3), (40, 120)),      # récents, peu de commandes
        (0.40, (150, 360), (1, 2), (20, 90)),    # anciens, inactifs
    ]
    end = pd.Timestamp("2011-12-09")
    rows = []
    cid = 12000
    for weight, rec, freq, amount in profiles:
        for _ in range(int(n_customers * weight)):
            cid += 1
            last = end - pd.Timedelta(days=int(rng.integers(*rec)))
            n_orders = int(rng.integers(freq[0], freq[1] + 1))
            for j in range(n_orders):
                date = last - pd.Timedelta(days=int(rng.integers(0, 40)) * j)
                qty = int(rng.integers(1, 10))
                price = float(rng.uniform(*amount)) / qty
                rows.append((f"5{cid}{j}", cid, date, qty, round(price, 2)))
    return pd.DataFrame(rows, columns=["InvoiceNo", "CustomerID", "InvoiceDate", "Quantity", "UnitPrice"])


@pytest.fixture(scope="session")
def transactions() -> pd.DataFrame:
    return _make_transactions()


@pytest.fixture(scope="session")
def dirty_transactions(transactions) -> pd.DataFrame:
    """Transactions auxquelles on ajoute des lignes à nettoyer."""
    extra = pd.DataFrame(
        [
            ("C9001", 12001, pd.Timestamp("2011-11-01"), -2, 5.0),   # annulation
            ("9002", np.nan, pd.Timestamp("2011-11-01"), 1, 5.0),    # client manquant
            ("9003", 12002, pd.Timestamp("2011-11-01"), -1, 5.0),    # quantité négative
            ("9004", 12002, pd.Timestamp("2011-11-01"), 1, 0.0),     # prix nul
        ],
        columns=transactions.columns,
    )
    duplicate = transactions.iloc[[0]]
    return pd.concat([transactions, extra, duplicate], ignore_index=True)


@pytest.fixture(scope="session")
def rfm(transactions):
    import rfm_core as core

    clean, _ = core.clean_transactions(transactions)
    return core.compute_rfm(clean)
