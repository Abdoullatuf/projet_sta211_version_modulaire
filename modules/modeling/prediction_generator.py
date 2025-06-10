
#modules/modeling/predictiongenerator.py

from sklearn.experimental import enable_iterative_imputer  # ⚠️ doit être tout en haut
import pandas as pd
import joblib
from pathlib import Path
from preprocessing.final_preprocessing import prepare_final_dataset




def generate_challenge_predictions(
    model_path: Path,
    raw_data_path: Path,
    save_dir: Path,
    dataset_name: str = "mice_with_outliers",
    imputation_method: str = "mice",
    remove_outliers: bool = False,
    verbose: bool = True
):
    """
    Génère les prédictions pour le challenge final à partir du pipeline EDA + modèle déjà entraîné.

    Args:
        model_path (Path): Chemin vers le fichier .joblib du modèle sauvegardé.
        raw_data_path (Path): Chemin vers le fichier CSV brut (ex: data_test.csv).
        save_dir (Path): Dossier où sauvegarder le fichier de prédictions.
        dataset_name (str): Nom du dataset (utilisé pour nommer le fichier prédiction).
        imputation_method (str): "mice" ou "knn".
        remove_outliers (bool): True si le dataset doit être nettoyé des outliers.
        verbose (bool): Affiche les étapes si True.

    Returns:
        pd.DataFrame: DataFrame contenant les prédictions avec colonnes [id, prediction].
    """
    if verbose:
        print(f"\n📂 Chargement des données brutes depuis : {raw_data_path}")
    df_raw = pd.read_csv(raw_data_path)

    if verbose:
        print("🧼 Application du pipeline de prétraitement EDA...")
    df_preprocessed = prepare_final_dataset(
        df_raw,
        method=imputation_method,
        remove_outliers=remove_outliers,
        verbose=verbose
    )

    if verbose:
        print(f"🤖 Chargement du modèle depuis : {model_path}")
    model = joblib.load(model_path)

    X_test = df_preprocessed.drop(columns=["outcome"], errors="ignore")
    y_pred = model.predict(X_test)

    predictions = pd.DataFrame({
        "id": range(1, len(y_pred) + 1),
        "prediction": y_pred
    })

    output_file = save_dir / f"predictions_{dataset_name}.csv"
    if verbose:
        print(f"💾 Sauvegarde dans : {output_file}")
    predictions.to_csv(output_file, index=False)

    return predictions
