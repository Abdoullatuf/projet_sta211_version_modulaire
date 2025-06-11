#run_ablation_analysis.py


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.metrics import f1_score
from typing import List, Tuple

def run_ablation_analysis(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_importance: pd.DataFrame,
    threshold: float,
    max_vars: int = 50,
    step: int = 5,
    model_name: str = "Stacking",
    dataset_name: str = "MICE",
    save_path: str = None
) -> Tuple[pd.DataFrame, int, float]:
    """
    Effectue une étude d'ablation en F1-score selon le nombre de variables retenues.

    Returns:
        - df_resultats: DataFrame contenant les F1-scores selon N
        - best_n: nombre de variables optimal
        - best_f1: F1-score correspondant
    """
    results = []

    variables_sorted = feature_importance['Variable'].tolist()
    n_features_range = list(range(step, min(max_vars, len(variables_sorted)) + 1, step))

    print(f"🧪 Étude d’ablation en cours ({len(n_features_range)} valeurs de N)...")

    for n in n_features_range:
        selected_features = variables_sorted[:n]
        X_train_sel = X_train[selected_features]
        X_test_sel = X_test[selected_features]

        model_copy = clone(model)
        model_copy.fit(X_train_sel, y_train)
        y_pred_proba = model_copy.predict_proba(X_test_sel)[:, 1]
        y_pred_opt = (y_pred_proba >= threshold).astype(int)
        f1 = f1_score(y_test, y_pred_opt)
        results.append((n, f1))

        print(f"   ➤ N={n} → F1={f1:.4f}")

    df_resultats = pd.DataFrame(results, columns=["N_variables", "F1_score"])
    best_row = df_resultats.loc[df_resultats["F1_score"].idxmax()]
    best_n = int(best_row["N_variables"])
    best_f1 = float(best_row["F1_score"])

    # Courbe
    plt.figure(figsize=(8, 4))
    plt.plot(df_resultats["N_variables"], df_resultats["F1_score"], marker="o", linestyle="-")
    plt.axvline(best_n, color="red", linestyle="--", label=f"Optimal = {best_n} variables")
    plt.title(f"Ablation Study - {model_name} ({dataset_name})")
    plt.xlabel("Nombre de variables utilisées")
    plt.ylabel("F1-score (à seuil optimisé)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"📈 Figure sauvegardée : {save_path}")
    plt.show()

    return df_resultats, best_n, best_f1
