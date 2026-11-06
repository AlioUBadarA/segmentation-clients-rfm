from pathlib import Path

import pandas as pd
import pytest
import sklearn

pytest.importorskip("streamlit")
pytest.importorskip("plotly")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")

pytestmark = pytest.mark.skipif(
    sklearn.__version__ != "1.6.1", reason="le modèle livré exige scikit-learn 1.6.1"
)


def _run_app() -> AppTest:
    return AppTest.from_file(APP_PATH, default_timeout=120).run()


def _upload(at: AppTest, index: int, name: str, content: bytes) -> AppTest:
    at.get("file_uploader")[index].set_value((name, content, "text/csv"))
    return at.run()


def test_app_starts_without_error():
    at = _run_app()
    assert not at.exception
    assert len(at.tabs) >= 3


def test_simulation_returns_segment_and_action():
    at = _run_app()
    at.button[0].click().run()
    assert not at.exception
    assert at.success[0].value.startswith("Segment prédit : ")
    assert any(m.value.startswith("Action recommandée") for m in at.markdown)


def test_simulation_flags_single_large_order():
    at = _run_app()
    at.number_input[0].set_value(10).run()
    at.number_input[1].set_value(1).run()
    at.number_input[2].set_value(8000.0).run()
    at.button[0].click().run()
    assert any("atypique" in w.value for w in at.warning)


def test_rfm_csv_import_valid():
    at = _run_app()
    csv = pd.DataFrame(
        {
            "CustomerID": [1, 2, 3],
            "Recency": [5, 300, 40],
            "Frequency": [20, 1, 4],
            "Monetary": [9000, 80, 1500],
        }
    ).to_csv(index=False).encode()
    at = _upload(at, 0, "rfm.csv", csv)
    assert not at.exception and not at.error
    assert dict((m.label, m.value) for m in at.metric)["Clients segmentés"] == "3"


def test_rfm_csv_import_rejects_negative_values():
    at = _run_app()
    csv = b"Recency,Frequency,Monetary\n-1,2,100\n"
    at = _upload(at, 0, "bad.csv", csv)
    assert any("négatives" in e.value for e in at.error)


def test_rfm_csv_import_rejects_missing_columns():
    at = _run_app()
    at = _upload(at, 0, "bad.csv", b"Recency\n1\n")
    assert any("Colonnes manquantes" in e.value for e in at.error)


def test_transactions_import_computes_rfm(transactions):
    at = _run_app()
    at = _upload(at, 1, "transactions.csv", transactions.to_csv(index=False).encode())
    assert not at.exception and not at.error
    clients = int(transactions["CustomerID"].nunique())
    assert dict((m.label, m.value) for m in at.metric)["Clients segmentés"] == str(clients)


def test_transactions_import_rejects_missing_columns():
    at = _run_app()
    at = _upload(at, 1, "tr.csv", b"InvoiceNo,CustomerID\n1,2\n")
    assert any("Colonnes manquantes" in e.value for e in at.error)
