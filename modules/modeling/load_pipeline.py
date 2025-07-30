#modules/medeling/load_pipeline

import joblib
from pathlib import Path
from modules.config.env_setup import init_project, set_display_options

# 🚀 Initialisation complète (gère tout : Colab, Drive, dépendances, chemins)
init_result = init_project()
paths = init_result["paths"]


def load_pipeline(model_type, imputation_method, model_variant="full", models_dir=paths["MODELS_DIR"]):
    """
    Charge un pipeline optimisé depuis le notebook 2.

    Args:
        model_type (str): Type de modèle (e.g., 'randforest', 'xgboost', 'svm', 'mlp', 'gradboost').
        imputation_method (str): Méthode d'imputation ('knn' ou 'mice').
        model_variant (str, optional): Variante du modèle ('full', 'reduced'). Defaults to "full".
        models_dir (Path, optional): Chemin du répertoire des modèles. Defaults to MODELS_DIR.

    Returns:
        object: Le pipeline chargé.
    """
    if model_variant == "full":
        path = models_dir / "notebook2" / f"pipeline_{model_type}_{imputation_method}_{model_variant}.joblib"
    elif model_variant == "reduced":
        # Ajustez le nommage si nécessaire
        path = models_dir / "notebook2" / imputation_method / "reduced" / f"pipeline_{model_type}_{imputation_method}_{model_variant}.joblib"
    else:
        raise ValueError("model_variant doit être 'full' ou 'reduced'")

    if path.exists():
        pipeline = joblib.load(path)
        print(f"✅ Pipeline {model_type} ({imputation_method}, {model_variant}) chargé.")
        return pipeline
    else:
        print(f"❌ Chemin du pipeline non trouvé : {path}")
        return None

# Exemple d'utilisation
# pipeline_rf_knn_full = load_pipeline('randforest', 'knn', 'full')