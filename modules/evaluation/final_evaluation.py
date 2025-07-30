# Fichier : modules/evaluation/final_evaluation.py
import joblib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay
)

def load_thresholds(imputation: str, version: str, thresholds_dir: Path) -> dict:
    path = thresholds_dir / f"optimized_thresholds_{imputation}_{version}.json"
    if not path.exists():
        raise FileNotFoundError(f"Fichier manquant : {path}")
    with open(path, "r") as f:
        return json.load(f)

def evaluate_all_models_on_test(pipelines_dict, thresholds_dict, splits_dict, output_dir: Path):
    """
    Évalue tous les modèles avec leurs seuils sur les jeux de test (KNN/MICE × Full/Reduced).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = []

    for key in pipelines_dict:
        print(f"\n🔍 Évaluation : {key.upper()}")
        pipes = pipelines_dict[key]
        thresholds = thresholds_dict[key]

        # Support des splits réduits avec structure différente
        split = splits_dict[key]
        # Corrected logic to access test data
        if "test" in split and isinstance(split["test"], dict):
            test_X = split["test"]["X"]
            test_y = split["test"]["y"]
        elif "X_test" in split and "y_test" in split:
            test_X = split["X_test"]
            test_y = split["y_test"]
        else:
            raise ValueError(f"Could not find test data in split dictionary for key: {key}")


        for model_name, pipe in pipes.items():
            if model_name not in thresholds:
                print(f"⚠️ Seuil manquant pour {model_name} ({key}) – ignoré.")
                continue

            # Check if the model has predict_proba or decision_function
            if hasattr(pipe, "predict_proba"):
                y_scores = pipe.predict_proba(test_X)[:, 1]
            elif hasattr(pipe, "decision_function"):
                y_scores = pipe.decision_function(test_X)
            else:
                print(f"⚠️ Modèle {model_name} ({key}) ne supporte ni predict_proba ni decision_function – ignoré.")
                continue

            thr = thresholds[model_name]["threshold"]
            y_pred = (y_scores >= thr).astype(int)

            f1 = f1_score(test_y, y_pred)
            precision = precision_score(test_y, y_pred)
            recall = recall_score(test_y, y_pred)
            auc = roc_auc_score(test_y, y_scores)

            result = {
                "model": model_name,
                "imputation": key.split("_")[0].upper(),
                "version": key.split("_")[1].upper(),
                "threshold": thr,
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "auc": auc
            }
            all_results.append(result)

            # Affichage de la matrice de confusion
            print(f"\n📌 {model_name} ({key}) - F1={f1:.4f}, Précision={precision:.4f}, Rappel={recall:.4f}, AUC={auc:.4f}")
            cm = confusion_matrix(test_y, y_pred)
            disp = ConfusionMatrixDisplay(cm)
            disp.plot(cmap="Blues")
            plt.title(f"{model_name} – {key.upper()} – TEST")
            plt.tight_layout()
            plt.gcf().set_size_inches(4, 4)  # ✅ Taille réduite
            plt.show()

    df_results = pd.DataFrame(all_results)
    path = output_dir / "test_results_all_models.csv"
    df_results.to_csv(path, index=False)
    print(f"\n💾 Résultats sauvegardés → {path.name}")
    return df_results

def plot_test_performance(df_results):
    """
    Affiche un barplot des performances F1 sur TEST.
    """
    if df_results.empty:
        print("❌ Données vides. Rien à afficher.")
        return

    df_sorted = df_results.sort_values("f1", ascending=True)
    plt.figure(figsize=(8, 4))
    ax = sns.barplot(data=df_sorted, x="f1", y="model", hue="imputation", dodge=True)
    plt.title("F1-score sur TEST (toutes configurations)", fontsize=14)
    plt.xlabel("F1-score"); plt.ylabel("Modèle")
    plt.grid(axis="x", alpha=0.3, linestyle="--")
    plt.legend(title="Imputation")
    plt.tight_layout()
    plt.show()