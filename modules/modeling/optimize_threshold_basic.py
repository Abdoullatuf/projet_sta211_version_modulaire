"""
optimize_threshold_basic.py
───────────────────────────
Fonctions minimalistes pour :

• rechercher le seuil qui maximise le F1-score (ou une autre métrique)
  sur un *jeu de validation* ;
• optimiser ce seuil pour plusieurs modèles déjà entraînés.

Aucune dépendance vers d’autres modules maison → pas de circular import 🎉
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_curve,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score)

import matplotlib.pyplot as plt
import logging
import joblib




log = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# 1) Fonction unique : optimise un SEUL pipeline
# ----------------------------------------------------------------------
def optimize_threshold(
    pipeline,
    X_val,
    y_val,
    *,
    plot: bool = True,
    label: str = "model",
) -> Dict[str, float]:
    """
    Calcule le seuil maximisant le F1 sur `X_val / y_val`.

    Retourne : dict(threshold, f1, precision, recall)
    """
    # 1. Obtenir un *score continu* pour chaque observation
    if hasattr(pipeline, "predict_proba"):
        y_scores = pipeline.predict_proba(X_val)[:, 1]
    else:  # ex. SVM linéaire
        y_scores = pipeline.decision_function(X_val)

    # 2. Courbe Pr-Re
    prec, rec, thr = precision_recall_curve(y_val, y_scores)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    best_idx = int(np.argmax(f1))
    best_thr = float(thr[best_idx])

    # 3. Métriques au seuil optimal
    y_pred_opt = (y_scores >= best_thr).astype(int)
    metrics = {
        "threshold":  round(best_thr, 4),
        "f1":         round(f1_score(y_val, y_pred_opt), 4),
        "precision":  round(precision_score(y_val, y_pred_opt), 4),
        "recall":     round(recall_score(y_val, y_pred_opt), 4),
    }

    # 4. Plot facultatif
    if plot:
        plt.figure(figsize=(4, 3))
        plt.plot(rec, prec, lw=1.2)
        plt.scatter(
            rec[best_idx], prec[best_idx],
            c="red", label=f"F1={metrics['f1']:.3f}",
        )
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Pr–Re curve – {label}\nBest thr = {metrics['threshold']}")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return metrics


# ----------------------------------------------------------------------
# 2) Optimiser une COLLECTION de pipelines
# ----------------------------------------------------------------------
def optimize_multiple(
    pipes: Dict[str, Union[str, Path, object]],
    X_val,
    y_val,
    *,
    plot_each: bool = False,
    show_df: bool = True,
) -> pd.DataFrame:
    """
    Optimise le seuil pour plusieurs pipelines.

    • `pipes` : {name: Pipeline  *ou*  chemin .joblib}
    • `plot_each` : True → trace la courbe Pr-Re pour chacun
    • `show_df`   : False → retourne seulement le DataFrame (pas d’affichage)

    Retour : DataFrame (model, threshold, f1, precision, recall) trié par f1
    """
    rows = []
    for name, obj in pipes.items():
        pipe = joblib.load(obj) if isinstance(obj, (str, Path)) else obj
        metr = optimize_threshold(
            pipe, X_val, y_val,
            plot=plot_each, label=name,
        )
        metr["model"] = name
        rows.append(metr)

    df = (
        pd.DataFrame(rows)
          .sort_values("f1", ascending=False)
          .reset_index(drop=True)
    )

    if show_df:
        display_cols = {"f1": "{:.4f}", "precision": "{:.4f}",
                        "recall": "{:.4f}", "threshold": "{:.3f}"}
        display(df.style.format(display_cols))

    return df




def evaluate_thresholds(model, X_test, y_test, thresholds=np.linspace(0.2, 0.8, 61)):
    """
    Évalue les performances du modèle pour différents seuils.
    Retourne un DataFrame trié par F1-score décroissant, et le seuil optimal.
    """
    y_probs = model.predict_proba(X_test)[:, 1]
    results = []

    for threshold in thresholds:
        y_pred = (y_probs >= threshold).astype(int)

        f1 = f1_score(y_test, y_pred, zero_division=0)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_probs)

        results.append({
            "Threshold": threshold,
            "F1-score": f1,
            "Precision": precision,
            "Recall": recall,
            "AUC": auc
        })

    df = pd.DataFrame(results).sort_values(by="F1-score", ascending=False)

    if df.iloc[0]["F1-score"] == 0:
        print("⚠️ Aucun seuil ne donne un F1-score > 0, utilisation du seuil 0.5")
        return df, 0.5

    return df, df.iloc[0]["Threshold"]

