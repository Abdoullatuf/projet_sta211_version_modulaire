# Fichier : modules/visualization/roc_comparison.py

import pandas as pd
import joblib
import json
import logging
from pathlib import Path
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)

def plot_best_roc_curves_comparison(
    models_dir: Path,
    figures_dir: Path,
    splits: dict
):
    """
    Identifie les meilleurs modèles pour KNN et MICE, puis trace et sauvegarde
    la comparaison de leurs courbes ROC sur le jeu de validation.
    
    Args:
        models_dir (Path): Chemin vers le dossier racine des modèles (ex: .../notebook2).
        figures_dir (Path): Chemin vers le dossier où sauvegarder la figure.
        splits (dict): Dictionnaire contenant les données 'X_val' et 'y_val' pour 'knn' et 'mice'.
    """
    print("📊 Comparaison des courbes ROC des meilleurs modèles (KNN vs MICE)")
    print("=" * 70)

    try:
        # --- 1. Identifier les meilleurs modèles ---
        df_all_thr = pd.read_csv(models_dir / "df_all_thresholds.csv")
        
        best_knn_row = df_all_thr[df_all_thr['Imputation'] == 'KNN'].sort_values('f1', ascending=False).iloc[0]
        best_mice_row = df_all_thr[df_all_thr['Imputation'] == 'MICE'].sort_values('f1', ascending=False).iloc[0]

        best_knn_name = best_knn_row['model']
        best_mice_name = best_mice_row['model']

        print(f"🏆 Meilleur modèle KNN (selon F1): {best_knn_name}")
        print(f"🏆 Meilleur modèle MICE (selon F1): {best_mice_name}")

        # --- 2. Charger les pipelines ---
        with open(models_dir / "best_knn_pipelines.json", "r") as f:
            knn_paths = json.load(f)
        with open(models_dir / "best_mice_pipelines.json", "r") as f:
            mice_paths = json.load(f)
            
        pipe_knn = joblib.load(Path(knn_paths[best_knn_name]))
        pipe_mice = joblib.load(Path(mice_paths[best_mice_name]))
        print("✅ Pipelines chargés.")

        # --- 3. Calculer les probabilités ---
        X_val_knn, y_val_knn = splits["knn"]["X_val"], splits["knn"]["y_val"]
        X_val_mice, y_val_mice = splits["mice"]["X_val"], splits["mice"]["y_val"]
        
        y_scores_knn = pipe_knn.predict_proba(X_val_knn)[:, 1]
        y_scores_mice = pipe_mice.predict_proba(X_val_mice)[:, 1]

        # --- 4. Calculer les courbes ROC ---
        fpr_knn, tpr_knn, _ = roc_curve(y_val_knn, y_scores_knn)
        roc_auc_knn = auc(fpr_knn, tpr_knn)
        
        fpr_mice, tpr_mice, _ = roc_curve(y_val_mice, y_scores_mice)
        roc_auc_mice = auc(fpr_mice, tpr_mice)

        # --- 5. Tracer le graphique ---
        plt.figure(figsize=(6, 4))
        plt.plot(fpr_knn, tpr_knn, color='darkorange', lw=2, label=f'{best_knn_name} (KNN) (AUC = {roc_auc_knn:.3f})')
        plt.plot(fpr_mice, tpr_mice, color='cornflowerblue', lw=2, label=f'{best_mice_name} (MICE) (AUC = {roc_auc_mice:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('Taux de Faux Positifs (FPR)')
        plt.ylabel('Taux de Vrais Positifs (TPR)')
        plt.title('Comparaison ROC (Meilleurs modèles KNN vs MICE)', fontsize=12)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # --- 6. Sauvegarder ---
        figures_dir.mkdir(parents=True, exist_ok=True)
        fig_path = figures_dir / "roc_curve_comparison_best_models.png"
        plt.savefig(fig_path, dpi=150)
        log.info(f"📊 Graphique ROC sauvegardé → {fig_path.name}")
        print(f"\n✅ Graphique ROC sauvegardé : {fig_path}")
        plt.show()

    except FileNotFoundError as e:
        log.error(f"❌ Fichier manquant : {e}")
        print("❌ Erreur : un fichier nécessaire n'a pas été trouvé. Vérifiez les sauvegardes de l'étape précédente.")
    except Exception as e:
        log.error(f"❌ Erreur inattendue lors de la génération du graphique : {e}")
        print(f"❌ Une erreur s'est produite : {e}")