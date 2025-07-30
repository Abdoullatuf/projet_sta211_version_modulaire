# File: modules/reporting/final_synthesis.py

import pandas as pd
import json
from pathlib import Path
from IPython.display import display, Markdown
from sklearn.metrics import f1_score, precision_score, recall_score

def generate_final_synthesis(
    all_thresholds: dict,
    all_optimized_pipelines: dict,
    splits: dict,
    models_dir: Path,
    display_results: bool = True
) -> pd.DataFrame:
    """
    Génère une synthèse comparative complète des modèles optimisés et du stacking.

    Args:
        all_thresholds (dict): Dictionnaire des seuils optimaux chargés.
        all_optimized_pipelines (dict): Dictionnaire des pipelines de modèles chargés.
        splits (dict): Dictionnaire contenant les données de test (X_test_knn, y_test_knn, etc.).
        models_dir (Path): Chemin vers le dossier racine des modèles.
        display_results (bool): Si True, affiche les tableaux et recommandations dans le notebook.

    Returns:
        pd.DataFrame: Le DataFrame de synthèse final, trié par F1-Score.
    """
    print("🚀 Génération de la synthèse comparative finale...")
    print("=" * 60)

    # --- 1. Calculer les performances réelles des meilleurs modèles uniques ---
    # SVM (KNN)
    pipe_svm_knn = all_optimized_pipelines["svm_knn"]
    thr_svm_knn = all_thresholds["svm_knn"]["threshold"]
    X_test_knn, y_test_knn = splits['knn']['X_test'], splits['knn']['y_test']
    proba_svm_knn = pipe_svm_knn.predict_proba(X_test_knn)[:, 1]
    pred_svm_knn = (proba_svm_knn >= thr_svm_knn).astype(int)
    perf_svm_knn = {
        "f1": f1_score(y_test_knn, pred_svm_knn),
        "precision": precision_score(y_test_knn, pred_svm_knn),
        "recall": recall_score(y_test_knn, pred_svm_knn)
    }

    # XGBoost (MICE)
    pipe_xgb_mice = all_optimized_pipelines["xgboost_mice"]
    thr_xgb_mice = all_thresholds["xgboost_mice"]["threshold"]
    X_test_mice, y_test_mice = splits['mice']['X_test'], splits['mice']['y_test']
    proba_xgb_mice = pipe_xgb_mice.predict_proba(X_test_mice)[:, 1]
    pred_xgb_mice = (proba_xgb_mice >= thr_xgb_mice).astype(int)
    perf_xgb_mice = {
        "f1": f1_score(y_test_mice, pred_xgb_mice),
        "precision": precision_score(y_test_mice, pred_xgb_mice),
        "recall": recall_score(y_test_mice, pred_xgb_mice)
    }
    
    # --- 2. Charger les performances des modèles de stacking ---
    stacking_dir = models_dir / "notebook3" / "stacking"
    
    def load_json_performance(path, default_perf):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"⚠️ Fichier non trouvé : {path.name}. Utilisation des valeurs par défaut.")
            return default_perf

    perf_stack_refit_knn = load_json_performance(stacking_dir / "stack_refit_knn_performance.json", {"f1_score": 0, "precision": 0, "recall": 0})
    perf_stack_refit_mice = load_json_performance(stacking_dir / "stack_refit_mice_performance.json", {"f1_score": 0, "precision": 0, "recall": 0})
    perf_stack_no_refit_knn = load_json_performance(stacking_dir / "stack_no_refit_knn_performance.json", {"f1_score": 0, "precision": 0, "recall": 0})
    perf_stack_no_refit_mice = load_json_performance(stacking_dir / "stack_no_refit_mice_performance.json", {"f1_score": 0, "precision": 0, "recall": 0})

    # --- 3. Création du tableau de synthèse ---
    synthese_data = {
        "Modèle": [
            "SVM (KNN) - Optimisé", "XGBoost (MICE) - Optimisé",
            "Stacking avec refit (KNN)", "Stacking avec refit (MICE)",
            "Stacking sans refit (KNN)", "Stacking sans refit (MICE)"
        ],
        "F1-Score": [
            perf_svm_knn["f1"], perf_xgb_mice["f1"],
            perf_stack_refit_knn.get("f1_score", 0), perf_stack_refit_mice.get("f1_score", 0),
            perf_stack_no_refit_knn.get("f1_score", 0), perf_stack_no_refit_mice.get("f1_score", 0)
        ],
        "Précision": [
            perf_svm_knn["precision"], perf_xgb_mice["precision"],
            perf_stack_refit_knn.get("precision", 0), perf_stack_refit_mice.get("precision", 0),
            perf_stack_no_refit_knn.get("precision", 0), perf_stack_no_refit_mice.get("precision", 0)
        ],
        "Rappel": [
            perf_svm_knn["recall"], perf_xgb_mice["recall"],
            perf_stack_refit_knn.get("recall", 0), perf_stack_refit_mice.get("recall", 0),
            perf_stack_no_refit_knn.get("recall", 0), perf_stack_no_refit_mice.get("recall", 0)
        ]
    }
    df_synthese = pd.DataFrame(synthese_data).sort_values("F1-Score", ascending=False).reset_index(drop=True)

    # --- 4. Affichage et analyse (optionnel) ---
    if display_results:
        display(Markdown("### Tableau de synthèse comparative"))
        display(df_synthese.style.format({
            "F1-Score": "{:.4f}", "Précision": "{:.4f}", "Rappel": "{:.4f}"
        }).background_gradient(subset=["F1-Score"], cmap="Greens"))

        display(Markdown("### Recommandations"))
        champion = df_synthese.iloc[0]
        display(Markdown(f"""
        **Pour la performance maximale :**
        - **{champion['Modèle']}** : Atteint le meilleur F1-Score de **{champion['F1-Score']:.4f}**.
        
        **Conclusion pour la soumission finale :**
        - Le modèle **{champion['Modèle']}** est le meilleur candidat pour la soumission au challenge.
        """))

    return df_synthese