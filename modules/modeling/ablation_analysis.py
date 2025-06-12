# run_ablation_analysis.py
from sklearn.metrics import f1_score
from sklearn.base import clone
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib

def run_ablation_analysis(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_importance: pd.DataFrame,
    threshold: float = 0.5,
    n_features_range: list = None,
    model_name: str = "Stacking",
    dataset_name: str = "MICE",
    save_path=None,
    apply_smote: bool = True  # 🔧 Ajout du paramètre SMOTE
):
    """
    Étudie l'impact du nombre de variables sur la performance (F1-score).
    """

    print(f"🧪 Étude d’ablation sur {dataset_name} ({model_name})")

    # Valeurs par défaut
    if n_features_range is None:
        n_features_range = list(range(5, min(X_train.shape[1], 105), 5))

    f1_scores = []
    used_features = []

    for n in n_features_range:
        top_features = feature_importance["Variable"].head(n).tolist()

        # Sélection
        X_train_sel = X_train[top_features]
        X_test_sel = X_test[top_features]

        # Préparation du pipeline
        steps = []
        if apply_smote:
            steps.append(('smote', SMOTE(random_state=42)))
        steps.append(('classifier', clone(model)))
        pipeline = ImbPipeline(steps)

        pipeline.fit(X_train_sel, y_train)
        y_pred = pipeline.predict(X_test_sel)
        score = f1_score(y_test, y_pred)
        f1_scores.append(score)
        used_features.append(top_features)

        print(f" → {n} variables : F1 = {score:.4f}")

    # Résultats
    ablation_df = pd.DataFrame({
        "N_Variables": n_features_range,
        "F1_score": f1_scores
    })

    best_idx = np.argmax(f1_scores)
    best_n = n_features_range[best_idx]
    best_f1 = f1_scores[best_idx]
    best_features = used_features[best_idx]

    print(f"\n✅ Meilleur score F1 : {best_f1:.4f} avec {best_n} variables.")

    # Visualisation
    plt.figure(figsize=(8, 5))
    plt.plot(n_features_range, f1_scores, 'o-', color='teal')
    plt.axvline(best_n, linestyle="--", color="red", label=f"N optimal = {best_n}")
    plt.title(f"Ablation - {model_name} ({dataset_name})")
    plt.xlabel("Nombre de variables utilisées")
    plt.ylabel("F1-score")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save_path:
        fig_name = f"ablation_{model_name.lower()}_{dataset_name.lower()}.png"
        plt.savefig(save_path / fig_name)
        print(f"📁 Figure sauvegardée : {fig_name}")
    plt.show()

    return ablation_df, best_n, best_f1, best_features
