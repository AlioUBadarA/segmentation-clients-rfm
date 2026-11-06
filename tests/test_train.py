import json
import pickle

import pytest

import rfm_core as core
import train


def test_train_end_to_end_on_synthetic_csv(tmp_path, transactions):
    data_path = tmp_path / "transactions.csv"
    transactions.to_csv(data_path, index=False)
    out_dir = tmp_path / "out"

    metrics = train.main(
        [
            "--data", str(data_path),
            "--k", "4",
            "--k-max", "6",
            "--output-dir", str(out_dir),
            "--n-stability-runs", "3",
        ]
    )

    for name in ["modele_pipeline.pkl", "rfm_segments.csv"]:
        assert (out_dir / name).exists()
    for name in ["metrics.json", "selection_k.csv", "selection_k.png"]:
        assert (out_dir / "reports" / name).exists()

    saved = json.loads((out_dir / "reports" / "metrics.json").read_text(encoding="utf-8"))
    assert saved["k"] == 4
    assert saved["n_clients"] == metrics["n_clients"]
    assert len(saved["profil_segments"]) == 4

    with open(out_dir / "modele_pipeline.pkl", "rb") as f:
        model = pickle.load(f)
    # Même format que le modèle historique : clés "pipeline" et "segments".
    assert {"pipeline", "segments", "metadata"} <= set(model)
    assert sorted(model["segments"].values()) == sorted(core.SEGMENT_ACTIONS)


def test_train_rejects_unknown_format(tmp_path):
    bad = tmp_path / "data.txt"
    bad.write_text("x")
    with pytest.raises(ValueError, match="Format non pris en charge"):
        train.main(["--data", str(bad), "--output-dir", str(tmp_path)])
