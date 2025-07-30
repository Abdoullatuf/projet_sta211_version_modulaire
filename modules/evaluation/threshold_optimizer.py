
#modules/evaluation/threshold_optimizer.py

"""Optimisation des seuils de classification pour les modèles de machine learning (personnalisé)
"""


import joblib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    precision_recall_curve, f1_score, precision_score, recall_score
)
import logging

log = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────
def optimize_threshold(pipeline, X_val, y_val, *, plot=True, label="model") -> dict:
    if hasattr(pipeline, "predict_proba"):
        y_scores = pipeline.predict_proba(X_val)[:, 1]
    elif hasattr(pipeline, "decision_function"):
        y_scores = pipeline.decision_function(X_val)
    else:
        raise AttributeError(f"Pipeline {label} n'a ni predict_proba ni decision_function")

    prec, rec, thr = precision_recall_curve(y_val, y_scores)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    best_idx = np.argmax(f1)
    best_thr = float(thr[best_idx])
    y_pred_opt = (y_scores >= best_thr).astype(int)

    metrics = {
        "threshold": round(best_thr, 4),
        "f1": round(f1_score(y_val, y_pred_opt), 4),
        "precision": round(precision_score(y_val, y_pred_opt, zero_division=0), 4),
        "recall": round(recall_score(y_val, y_pred_opt, zero_division=0), 4),
    }

    if plot:
        plt.figure(figsize=(4, 3))
        plt.plot(rec, prec, lw=1.2)
        plt.scatter(rec[best_idx], prec[best_idx], c="red", s=50,
                    label=f"F1={metrics['f1']:.3f}")
        plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.title(f"PR Curve – {label}\nThreshold = {metrics['threshold']}")
        plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()

    return metrics

# ────────────────────────────────────────────────────────────────────────
def optimize_multiple(pipes: dict, X_val, y_val, *, plot_each=False, show_df=True) -> pd.DataFrame:
    rows = []
    print("Optimisation des seuils sur jeu de VALIDATION")
    print(f"📊 {len(pipes)} modèles à optimiser...\n" + "-" * 50)

    for name, obj in pipes.items():
        try:
            pipe = joblib.load(obj) if isinstance(obj, (str, Path)) else obj
            metr = optimize_threshold(pipe, X_val, y_val, plot=plot_each, label=name)
            metr["model"] = name
            rows.append(metr)
            print(f"✅ {name:<12}: F1={metr['f1']:.4f}, seuil={metr['threshold']:.3f}")
        except Exception as e:
            print(f"❌ {name:<12}: Erreur → {e}")

    df = pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)

    if show_df and not df.empty:
        print("\n📋 Résultats triés par F1-score (validation)")
        display(df.style.format({
            "f1": "{:.4f}", "precision": "{:.4f}",
            "recall": "{:.4f}", "threshold": "{:.3f}"
        }).background_gradient(subset=["f1"], cmap="Greens"))

        print(f"\n🏆 Meilleur seuil : {df.iloc[0]['model']} (F1={df.iloc[0]['f1']:.4f})")

    return df

# ────────────────────────────────────────────────────────────────────────
def save_optimized_thresholds(df: pd.DataFrame, imputation: str, version: str, output_dir: Path):
    assert imputation in ["knn", "mice"]
    assert version in ["full", "reduced"]

    path = output_dir / f"optimized_thresholds_{imputation}_{version}.json"
    thresholds_dict = {
        row["model"]: {
            "threshold": row["threshold"],
            "f1": row["f1"],
            "precision": row["precision"],
            "recall": row["recall"]
        }
        for _, row in df.iterrows()
    }

    with open(path, "w") as f:
        json.dump(thresholds_dict, f, indent=2)

    print(f"💾 Seuils optimisés sauvegardés → {path.name}")

# ────────────────────────────────────────────────────────────────────────
def load_best_pipelines(imputation: str = "knn", version: str = "full", model_dir: Path = None) -> dict:
    assert imputation in ["knn", "mice"]
    assert version in ["full", "reduced"]

    suffix = f"{imputation}_{version}"
    index_path = model_dir / f"best_{suffix}_pipelines.json"

    if not index_path.exists():
        raise FileNotFoundError(f"❌ Fichier d'index introuvable : {index_path}")

    with open(index_path, "r") as f:
        pipeline_paths = json.load(f)

    best_pipes = {name: joblib.load(Path(path)) for name, path in pipeline_paths.items()}

    print(f"✅ {len(best_pipes)} pipelines rechargés ({imputation.upper()} – {version.upper()})")
    return best_pipes





def optimize_all_thresholds(pipelines_dict, splits_dict, output_dir):
    """
    Optimise et sauvegarde les seuils pour toutes les combinaisons KNN/MICE × Full/Reduced.

    Paramètres :
    ------------
    pipelines_dict : dict[str, dict]
        Dictionnaire des pipelines, ex: {"knn_full": ..., "mice_reduced": ...}
    splits_dict : dict[str, dict]
        Dictionnaire des splits de validation, ex: {"knn_full": ..., "mice_reduced": ...}
    output_dir : Path
        Dossier de sauvegarde des fichiers JSON des seuils optimisés.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    combinations = [
        ("knn", "full"),
        ("knn", "reduced"),
        ("mice", "full"),
        ("mice", "reduced"),
    ]

    for imput, version in combinations:
        key = f"{imput}_{version}"
        print(f"--- {imput.upper()} {version.upper()} ---")

        pipes = pipelines_dict.get(key)
        val_X = splits_dict[key]["X_val"] if version == "full" else splits_dict[key]["val"]["X"]
        val_y = splits_dict[key]["y_val"] if version == "full" else splits_dict[key]["val"]["y"]

        thresholds = optimize_multiple(
            pipes, val_X, val_y, plot_each=False
        )
        save_optimized_thresholds(thresholds, imput, version, output_dir)



# ────────────────────────────────────────────────────────────────────────
def load_optimized_thresholds(imputation: str, version: str, input_dir: Path) -> pd.DataFrame:
    """
    Charge les seuils optimisés à partir d'un fichier JSON sauvegardé.

    Paramètres :
    ------------
    imputation : str ("knn" ou "mice")
    version : str ("full" ou "reduced")
    input_dir : Path vers le dossier contenant les seuils optimisés

    Retour :
    --------
    pd.DataFrame avec colonnes : model, threshold, f1, precision, recall, Imputation, Version
    """
    assert imputation in ["knn", "mice"]
    assert version in ["full", "reduced"]

    path = input_dir / f"optimized_thresholds_{imputation}_{version}.json"
    if not path.exists():
        raise FileNotFoundError(f"❌ Fichier seuils introuvable : {path}")

    with open(path, "r") as f:
        data = json.load(f)

    df = pd.DataFrame(data).T
    df["model"] = df.index
    df["Imputation"] = imputation.upper()
    df["Version"] = version.upper()
    return df.reset_index(drop=True)
