# modules/modeling/run_stacking_no_refit.py (exemple de nom de fichier)

import numpy as np
from pathlib import Path
import json
from sklearn.metrics import f1_score, precision_score, recall_score
# Assurez-vous que ces modules existent et sont accessibles
from modules.modeling.load_pipeline import load_pipeline
from modules.modeling.generate_mean_proba import generate_mean_proba

def run_stacking_no_refit(X_val, y_val, X_test, y_test, imputation_method, models_dir, output_dir=None):
    """
    Exécute le processus complet de stacking sans refit pour une méthode d'imputation donnée.

    Args:
        X_val, y_val, X_test, y_test: Données de validation et de test.
        imputation_method (str): 'knn' ou 'mice'.
        models_dir (Path): Chemin du répertoire des modèles entraînés.
        output_dir (Path, optional): Chemin du répertoire de sauvegarde des résultats.
                                     Si None, utilise models_dir / "notebook3" / "stacking".

    Returns:
        dict: Dictionnaire contenant les métriques, le seuil et le chemin de sauvegarde.
              Retourne None en cas d'erreur.
    """
    print(f"🎯 STACKING SANS REFIT - {imputation_method.upper()}")
    print("=" * 80)

    # --- 1. Déterminer les répertoires ---
    if output_dir is None:
        stacking_dir = Path(models_dir) / "notebook3" / "stacking"
    else:
        stacking_dir = Path(output_dir)
    stacking_dir.mkdir(parents=True, exist_ok=True)

    results_summary = {
        'metrics': {},
        'threshold': None,
        'paths': {}
    }

    try:
        # --- 2. Chargement des pipelines ---
        print(f"🔄 Chargement des pipelines pour {imputation_method.upper()}...")
        pipelines = [
            load_pipeline('randforest', imputation_method, 'full'),
            load_pipeline('xgboost', imputation_method, 'full'),
            load_pipeline('svm', imputation_method, 'full'),
            load_pipeline('mlp', imputation_method, 'full'),
            load_pipeline('gradboost', imputation_method, 'full')
        ]
        if not all(pipelines): # Vérifie si tous les pipelines sont chargés
             raise RuntimeError("Échec du chargement d'un ou plusieurs pipelines.")
        print(f"✅ {len(pipelines)} pipelines chargés avec succès.")

        # --- 3. Calcul des probabilités moyennes ---
        print(f"📊 Calcul des probabilités moyennes sur validation {imputation_method.upper()}...")
        proba_mean_val = generate_mean_proba(pipelines, X_val)
        print(f"📊 Calcul des probabilités moyennes sur test {imputation_method.upper()}...")
        proba_mean_test = generate_mean_proba(pipelines, X_test)
        print("✅ Probabilités moyennes calculées.")

        # --- 4. Optimisation du seuil ---
        print(f"🔍 Optimisation du seuil sur validation {imputation_method.upper()}...")
        thresholds = np.linspace(0.2, 0.8, 61)
        best_f1 = -1
        best_threshold = 0.5

        for thr in thresholds:
            y_pred_val = (proba_mean_val >= thr).astype(int)
            f1 = f1_score(y_val, y_pred_val)
            if np.isfinite(f1) and f1 > best_f1:
                best_f1 = f1
                best_threshold = thr

        print(f"✅ Seuil optimal: {best_threshold:.3f} (F1-val: {best_f1:.4f})")
        results_summary['threshold'] = best_threshold

        # --- 5. Évaluation sur Test ---
        y_pred_test = (proba_mean_test >= best_threshold).astype(int)
        f1_test = f1_score(y_test, y_pred_test)
        precision_test = precision_score(y_test, y_pred_test)
        recall_test = recall_score(y_test, y_pred_test)

        print(f"\n🏆 RÉSULTATS STACKING SANS REFIT {imputation_method.upper()}:")
        print(f"   F1-score (test) : {f1_test:.4f}")
        print(f"   Précision (test): {precision_test:.4f}")
        print(f"   Rappel (test)   : {recall_test:.4f}")
        print(f"   Seuil           : {best_threshold:.3f}")

        results_summary['metrics'] = {
            'f1_score_val': float(best_f1),
            'f1_score_test': float(f1_test),
            'precision_test': float(precision_test),
            'recall_test': float(recall_test)
        }

        # --- 6. Sauvegarde des résultats ---
        print(f"\n💾 Sauvegarde des résultats...")
        stacking_no_refit_results = {
            "method": f"stacking_no_refit_{imputation_method}_full",
            "threshold": float(best_threshold),
            "performance": results_summary['metrics'],
            "predictions": {
                "validation": proba_mean_val.tolist(),
                "test": proba_mean_test.tolist()
            }
        }

        results_filename = f"stacking_no_refit_{imputation_method}_full.json"
        results_path = stacking_dir / results_filename

        with open(results_path, 'w') as f:
            json.dump(stacking_no_refit_results, f, indent=2)

        print(f"✅ Résultats sauvegardés dans: {results_path}")
        results_summary['paths']['results'] = results_path

    except Exception as e:
        print(f"❌ Erreur lors du stacking sans refit - {imputation_method.upper()}: {e}")
        return None # Retourne None en cas d'erreur

    print(f"\n🎉 STACKING SANS REFIT {imputation_method.upper()} TERMINÉ AVEC SUCCÈS !")
    return results_summary

# --- Exemple d'utilisation ---
# results_knn_noref = run_stacking_no_refit(
#     X_val_knn, y_val_knn, X_test_knn, y_test_knn,
#     imputation_method='knn',
#     models_dir=MODELS_DIR,
#     output_dir=OUTPUTS_DIR / "modeling" / "results"
# )
#
# results_mice_noref = run_stacking_no_refit(
#     X_val_mice, y_val_mice, X_test_mice, y_test_mice,
#     imputation_method='mice',
#     models_dir=MODELS_DIR,
#     output_dir=OUTPUTS_DIR / "modeling" / "results"
# )
