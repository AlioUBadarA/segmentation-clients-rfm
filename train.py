"""Entraînement reproductible du modèle de segmentation RFM.

Exemple :
    python train.py --data "Online Retail.xlsx" --k 4 --output-dir .

Le script nettoie les transactions, calcule le RFM, évalue plusieurs valeurs de k
(inertie, silhouette, Davies-Bouldin, Calinski-Harabasz), mesure la stabilité des
clusters, puis écrit :
    - modele_pipeline.pkl   : pipeline + noms des segments + métadonnées
    - rfm_segments.csv      : tableau RFM avec le segment de chaque client
    - reports/metrics.json  : métriques et profil des segments
    - reports/selection_k.csv et reports/selection_k.png : choix du nombre de clusters
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pickle
from pathlib import Path

import sklearn

import rfm_core as core


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Entraîne le modèle de segmentation RFM.")
    p.add_argument("--data", required=True, help="Fichier de transactions (.xlsx ou .csv).")
    p.add_argument("--k", type=int, default=4, help="Nombre de clusters (défaut : 4).")
    p.add_argument("--k-max", type=int, default=10, help="Plus grand k évalué (défaut : 10).")
    p.add_argument("--output-dir", default=".", help="Dossier de sortie (défaut : dossier courant).")
    p.add_argument("--n-stability-runs", type=int, default=20, help="Tirages pour la stabilité (défaut : 20).")
    p.add_argument("--no-figures", action="store_true", help="Ne pas produire le graphique de sélection de k.")
    return p.parse_args(argv)


def save_selection_figure(selection, k_chosen: int, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    series = [
        ("Inertie", "Méthode du coude"),
        ("Silhouette", "Score de silhouette (plus haut = mieux)"),
        ("Davies_Bouldin", "Davies-Bouldin (plus bas = mieux)"),
    ]
    for ax, (col, title) in zip(axes, series):
        ax.plot(selection["k"], selection[col], marker="o", color="black")
        ax.axvline(k_chosen, linestyle="--", color="grey", label=f"k = {k_chosen}")
        ax.set_title(title)
        ax.set_xlabel("Nombre de clusters (k)")
        ax.set_xticks(list(selection["k"]))
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    out_dir = Path(args.output_dir)
    reports_dir = out_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    print(f"Lecture de {args.data}")
    raw = core.load_transactions(args.data)
    clean, report = core.clean_transactions(raw)
    print("Nettoyage :", report)

    reference_date = clean["InvoiceDate"].max() + dt.timedelta(days=1)
    rfm = core.compute_rfm(clean, reference_date)
    print(f"RFM calculé pour {len(rfm)} clients (date de référence : {reference_date.date()})")

    print("Évaluation de k")
    selection = core.evaluate_k(rfm, range(2, args.k_max + 1))
    selection.round(4).to_csv(reports_dir / "selection_k.csv", index=False)
    print(selection.round(3).to_string(index=False))
    if not args.no_figures:
        save_selection_figure(selection, args.k, reports_dir / "selection_k.png")

    print(f"Entraînement du pipeline (k = {args.k})")
    pipeline = core.build_pipeline(args.k)
    pipeline.fit(rfm[core.RFM_COLUMNS])
    rfm["Cluster"] = pipeline.predict(rfm[core.RFM_COLUMNS])

    segments = core.name_segments(rfm)
    rfm["Segment"] = rfm["Cluster"].map(segments)
    profile = core.segment_profile(rfm)
    print(profile.to_string())

    stability = core.stability_score(rfm, args.k, n_runs=args.n_stability_runs)
    print("Stabilité (ARI par sous-échantillonnage) :", stability)

    row_k = selection.loc[selection["k"] == args.k]
    metrics = {
        "k": args.k,
        "n_clients": int(len(rfm)),
        "date_reference": str(reference_date.date()),
        "nettoyage": report,
        "scores_k_choisi": (
            {c: float(row_k.iloc[0][c]) for c in selection.columns if c != "k"} if not row_k.empty else {}
        ),
        "stabilite": stability,
        "profil_segments": profile.reset_index().to_dict(orient="records"),
    }
    (reports_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    artefact = {
        "pipeline": pipeline,
        "segments": segments,
        "metadata": {
            "sklearn_version": sklearn.__version__,
            "k": args.k,
            "n_clients": int(len(rfm)),
            "date_reference": str(reference_date.date()),
        },
    }
    with open(out_dir / "modele_pipeline.pkl", "wb") as f:
        pickle.dump(artefact, f)
    rfm.to_csv(out_dir / "rfm_segments.csv", index=False)

    print(f"Fichiers écrits dans {out_dir.resolve()} (version scikit-learn : {sklearn.__version__})")
    return metrics


if __name__ == "__main__":
    main()
