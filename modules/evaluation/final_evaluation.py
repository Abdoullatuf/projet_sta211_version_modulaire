# Fichier : modules/evaluation/final_evaluation.py

import json
import joblib
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import logging

# Configuration du logging
log = logging.getLogger(__name__)

def _plot_confusion_matrix(cm, title, figures_dir):
    """Affiche et sauvegarde une matrice de confusion."""
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Non-Pub", "Pub"], yticklabels=["Non-Pub", "Pub"])
    plt.title(title, pad=20)
    plt.xlabel("Prédiction")
    plt.ylabel("Réel")
    plt.tight_layout()
    
    # Sauvegarde de la figure
    if figures_dir:
        figures_dir.mkdir(parents=True, exist_ok=True)
        file_path = figures_dir / "final_test_confusion_matrix.png"
        plt.savefig(file_path, dpi=150)
        log.info(f"📊 Matrice de confusion sauvegardée : {file_path.name}")
        
    plt.show()

def run_final_evaluation(models_dir: Path, figures_dir: Path):
    """
    Exécute le pipeline complet d'évaluation finale sur le jeu de test.
    
    Args:
        models_dir (Path): Chemin vers le dossier racine des modèles.
        figures_dir (Path): Chemin vers le dossier où sauvegarder les figures.
        
    Returns:
        dict: Un dictionnaire contenant les métriques de performance sur le jeu de test.
    """
    print("🚀 Évaluation finale du champion sur le jeu de TEST...")
    print("=" * 60)

    # --- 1. Chargement des informations du champion ---
    try:
        info_path = models_dir / "notebook2" / "meilleur_modele" / "champion_info.json"
        with open(info_path, "r") as f:
            champion_info = json.load(f)
        
        champ_pipe_path = Path(champion_info["pipeline_path"])
        performance_data = champion_info["performance"]
        champion_threshold = performance_data["threshold"]
        champ_name = champion_info["model_name"]
        champ_imp = champion_info["imputation"]
        f1_val = performance_data["f1"]

        print(f"🏆 Champion identifié : {champ_name} ({champ_imp.upper()})")
        print(f"   Seuil optimisé : {champion_threshold:.3f}")
        
        champion_pipeline = joblib.load(champ_pipe_path)
        print("✅ Pipeline champion chargé.")

    except (FileNotFoundError, KeyError) as e:
        log.error(f"❌ Erreur critique lors du chargement des informations du champion : {e}")
        raise

    # --- 2. Chargement des données de test ---
    try:
        test_data_path = models_dir / "notebook2" / champ_imp.lower() / f"{champ_imp.lower()}_test.pkl"
        test_data = joblib.load(test_data_path)
        X_test_final = test_data["X"]
        y_test_final = test_data["y"]
        print(f"✅ Jeu de test ({champ_imp}) chargé : {X_test_final.shape}")
    except (FileNotFoundError, KeyError) as e:
        log.error(f"❌ Erreur critique lors du chargement des données de test : {e}")
        raise

    # --- 3. Évaluation sur le jeu de test ---
    print("\n📊 Évaluation des performances sur TEST final...")
    
    y_scores_test = champion_pipeline.predict_proba(X_test_final)[:, 1]
    y_pred_test = (y_scores_test >= champion_threshold).astype(int)

    # --- 4. Calcul et affichage des métriques ---
    f1_test = f1_score(y_test_final, y_pred_test)
    precision_test = precision_score(y_test_final, y_pred_test)
    recall_test = recall_score(y_test_final, y_pred_test)
    auc_test = roc_auc_score(y_test_final, y_scores_test)
    gap_f1 = f1_val - f1_test

    test_metrics = {
        "model": champ_name,
        "imputation": champ_imp,
        "f1_test": f1_test,
        "precision_test": precision_test,
        "recall_test": recall_test,
        "auc_test": auc_test,
        "f1_validation": f1_val,
        "gap_val_test": gap_f1,
        "threshold": champion_threshold
    }

    print("\n📈 Analyse de la généralisation (Gap VAL vs TEST)...")
    print(f"   F1-score (Validation) : {f1_val:.4f}")
    print(f"   F1-score (Test)       : {f1_test:.4f}")
    print(f"   Gap F1 (Val - Test)   : {gap_f1:+.4f}")
    
    if abs(gap_f1) <= 0.05:
        print("   ✅ Bonne généralisation !")
    else:
        print("   ⚠️ Surapprentissage potentiel (gap > 0.05)")

    # Affichage et sauvegarde de la matrice de confusion
    cm_title = f"Matrice de Confusion (TEST)\n{champ_name} ({champ_imp}) @ Seuil={champion_threshold:.3f}"
    _plot_confusion_matrix(confusion_matrix(y_test_final, y_pred_test), cm_title, figures_dir)

    # --- 5. Sauvegarde des résultats finaux ---
    results_path = figures_dir / "final_test_results.json"
    with open(results_path, "w") as f:
        json.dump(test_metrics, f, indent=2)
    log.info(f" Métriques de test finales sauvegardées : {results_path.name}")
    print("\n✅ Évaluation finale terminée. Résultats sauvegardés.")
    
    return test_metrics