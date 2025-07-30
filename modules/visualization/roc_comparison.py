import joblib, json
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, auc
import pandas as pd
import logging

log = logging.getLogger(__name__)

def plot_best_roc_curves_comparison(models_dir: Path, figures_dir: Path, splits: dict):
    """
    Trace les courbes ROC comparées des meilleurs modèles KNN et MICE sur le jeu de validation.

    Paramètres :
    ------------
    models_dir : Path
        Dossier contenant les fichiers pipeline + seuils + df_all_thresholds.csv
    figures_dir : Path
        Dossier de sortie pour enregistrer la figure
    splits : dict
        Dictionnaire contenant les données de validation (X_val, y_val)
        pour les combinaisons : knn_full, knn_reduced, mice_full, mice_reduced
    """
    df_path = models_dir / "df_all_thresholds.csv"
    if not df_path.exists():
        raise FileNotFoundError(f"❌ Fichier df_all_thresholds.csv introuvable : {df_path}")

    df_all = pd.read_csv(df_path)
    df_all.columns = df_all.columns.str.lower()

    if "imputation" not in df_all.columns or "version" not in df_all.columns:
        raise KeyError("❌ Les colonnes 'imputation' et 'version' sont requises dans df_all_thresholds.csv")

    best_combinations = (
        df_all.sort_values("f1", ascending=False)
              .groupby(["imputation", "version"], as_index=False)
              .first()
    )

    color_map = {
        ("knn", "full"): "#1f77b4",
        ("knn", "reduced"): "#2ca02c",
        ("mice", "full"): "#ff7f0e",
        ("mice", "reduced"): "#d62728",
    }

    plt.figure(figsize=(8, 6))

    for _, row in best_combinations.iterrows():
        model = row["model"]
        imp = row["imputation"]
        version = row["version"]

        key = f"{imp}_{version}".lower() # Convert key to lowercase

        # --- Debugging Print Statements ---
        log.info(f"Attempting to get split for key: {key}")
        split = splits.get(key)
        if split is None:
            log.warning(f"⚠️ Data missing for key: {key}. Split dictionary is None.")
            continue
        log.info(f"Split dictionary for key {key}: {split.keys()}")
        if "val" not in split:
             log.warning(f"⚠️ Data missing for key: {key}. 'val' not in split dictionary.")
             continue
        log.info(f"Accessing 'val' key. Contains: {split['val'].keys()}")
        # --- End Debugging Print Statements ---


        # Corrected pipeline path construction
        model_specific_dir = models_dir / model.lower() / version.lower()
        pipe_path = model_specific_dir / f"pipeline_{model.lower()}_{imp.lower()}_{version.lower()}.joblib"


        if not pipe_path.exists():
            print(f"❌ Pipeline manquant : {pipe_path.name}")
            continue

        pipe = joblib.load(pipe_path)


        # Access validation data using the consistent structure
        X_val = split["val"].get("X")
        y_val = split["val"].get("y")


        if X_val is None or y_val is None:
            print(f"❌ Données de validation incomplètes pour {key}")
            continue

        y_scores = pipe.predict_proba(X_val)[:, 1] if hasattr(pipe, "predict_proba") else pipe.decision_function(X_val)
        fpr, tpr, _ = roc_curve(y_val, y_scores)
        roc_auc = auc(fpr, tpr)

        label = f"{model} ({imp.upper()}-{version.upper()}) – AUC={roc_auc:.3f}"
        color = color_map.get((imp, version), None)
        plt.plot(fpr, tpr, lw=2, label=label, color=color)

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.5)
    plt.xlabel("Taux de faux positifs")
    plt.ylabel("Taux de vrais positifs")
    plt.title("Comparaison des meilleures courbes ROC (Validation)", fontsize=13)
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    figures_dir.mkdir(parents=True, exist_ok=True)
    save_path = figures_dir / "roc_comparison_best_models.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✅ Courbe ROC sauvegardée → {save_path}")
    plt.show()