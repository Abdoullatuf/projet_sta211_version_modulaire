from modules.config.env_setup import init_project
from modules.preprocessing.data_loader import load_data

import pandas as pd
import joblib
import json
import logging
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

# Initialisation environnement
project = init_project()
paths = project["paths"]
ROOT_DIR = paths["ROOT_DIR"]  # Récupérer ROOT_DIR depuis le dictionnaire
log.info("✅ Initialisation chemins via env_setup OK")

def generate_submission_file(test_data_path: Path,
                              columns_path: Path,
                              models_dir: Path,
                              output_path: Path) -> None:
    log.info("🚀 Démarrage du processus de prédiction pour la soumission...")

    champion_path = models_dir / "notebook2" / "meilleur_modele" / "champion_info.json"
    with open(champion_path) as f:
        champion_info = json.load(f)

    champ_name = champion_info["model_name"]
    champ_imp = champion_info["imputation"].lower()

    champ_pipe_path = ROOT_DIR / champion_info["pipeline_path"]
    threshold_value = champion_info["performance"]["threshold"]

    log.info(f"🏆 Champion chargé : {champ_name}, Seuil: {threshold_value:.3f}")

    # 1️⃣ Chargement des colonnes officielles
    columns = joblib.load(columns_path)
    log.info(f"✅ {len(columns)} noms de colonnes chargés depuis {columns_path.name}")

    # 2️⃣ Chargement des données brutes
    df_test_raw = load_data(test_data_path, require_outcome=False, display_info=False)

    # 3️⃣ Chargement des transformers et paramètres liés à l'imputation choisie
    transf_dir = models_dir / "notebook1" / champ_imp / f"{champ_imp}_transformers"

    yj_path = transf_dir / "yeo_johnson_X1_X2.pkl"
    bc_path = transf_dir / "box_cox_X3.pkl"
    poly_path = models_dir / "notebook1" / champ_imp / f"poly_transformer_{champ_imp}.pkl"

    transformer_yj = joblib.load(yj_path)
    transformer_bc = joblib.load(bc_path)
    poly_transformer = joblib.load(poly_path)

    # 4️⃣ Application des transformations
    df_x = df_test_raw.copy()

    # Yeo-Johnson + Box-Cox
    df_x['X1_transformed'] = transformer_yj.transform(df_x[['X1', 'X2']])[:, 0]
    df_x['X2_transformed'] = transformer_yj.transform(df_x[['X1', 'X2']])[:, 1]
    df_x3 = df_x[['X3']].clip(lower=1e-6)  # Protection avant Box-Cox
    df_x['X3_transformed'] = transformer_bc.transform(df_x3)

    # Imputation des valeurs manquantes avant les transformations polynomiales
    continuous_cols = ['X1_transformed', 'X2_transformed', 'X3_transformed']
    
    # Créer un nouvel imputer avec les mêmes paramètres que l'entraînement
    if champ_imp == "mice":
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer
        from sklearn.ensemble import RandomForestRegressor
        
        imputer = IterativeImputer(
            estimator=RandomForestRegressor(
                n_estimators=400,
                max_depth=20,
                min_samples_leaf=2,
                max_features=0.5,
                random_state=42,
                n_jobs=-1
            ),
            max_iter=10,
            random_state=42
        )
    else:  # knn
        from sklearn.impute import KNNImputer
        imputer = KNNImputer(n_neighbors=7)  # Même k que dans l'imputer sauvegardé
    
    df_x[continuous_cols] = imputer.fit_transform(df_x[continuous_cols])

    # Vérifier et traiter les valeurs NaN restantes
    if df_x.isnull().any().any():
        log.warning(f"⚠️ Valeurs NaN détectées après imputation. Colonnes avec NaN: {df_x.columns[df_x.isnull().any()].tolist()}")
        # Imputer les valeurs NaN restantes avec la médiane
        df_x = df_x.fillna(df_x.median())
        log.info("✅ Valeurs NaN restantes imputées avec la médiane")

    df_x.drop(columns=['X1', 'X2', 'X3'], inplace=True)

    # Capping
    capping_params = joblib.load(models_dir / "notebook1" / champ_imp / f"capping_params_{champ_imp}.pkl")
    for col in ['X1_transformed', 'X2_transformed', 'X3_transformed']:
        bounds = capping_params.get(col, {})
        df_x[col] = np.clip(df_x[col], bounds.get('lower_bound'), bounds.get('upper_bound'))

    # Suppression colinéarité
    cols_to_drop = joblib.load(models_dir / "notebook1" / champ_imp / f"cols_to_drop_corr_{champ_imp}.pkl")
    df_x.drop(columns=cols_to_drop, errors='ignore', inplace=True)

    # Ingénierie polynomial
    poly_features = poly_transformer.transform(df_x[continuous_cols])
    poly_feature_names = poly_transformer.get_feature_names_out(continuous_cols)
    df_poly = pd.DataFrame(poly_features, columns=poly_feature_names, index=df_x.index)

    df_engineered = df_x.drop(columns=continuous_cols).join(df_poly)

    # Vérifier et traiter les valeurs NaN après les transformations polynomiales
    if df_engineered.isnull().any().any():
        log.warning(f"⚠️ Valeurs NaN détectées après transformations polynomiales. Colonnes avec NaN: {df_engineered.columns[df_engineered.isnull().any()].tolist()}")
        # Imputer les valeurs NaN restantes avec la médiane
        df_engineered = df_engineered.fillna(df_engineered.median())
        log.info("✅ Valeurs NaN restantes imputées avec la médiane")

    # 5️⃣ Sélection finale des colonnes + modèle
    missing = [col for col in columns if col not in df_engineered.columns]
    if missing:
        log.warning(f"⚠️ Colonnes manquantes dans data_test après transformation : {missing}")

    final_cols = [col for col in columns if col in df_engineered.columns]
    df_engineered = df_engineered[final_cols].copy()

    final_pipeline = joblib.load(champ_pipe_path)
    probas = final_pipeline.predict_proba(df_engineered)[:, 1]
    predictions = (probas >= threshold_value).astype(int)

    ids_test = df_test_raw.index if df_test_raw.index.name == "ID" else df_test_raw.iloc[:, 0]

    # Conversion des prédictions au format attendu (ad./noad.)
    predictions_formatted = ["ad." if p == 1 else "noad." for p in predictions]
    
    # Création du DataFrame au bon format (sans en-tête, seulement les prédictions)
    submission_df = pd.DataFrame(predictions_formatted)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(output_path, index=False, header=False)

    log.info(f"🎉 Fichier de soumission créé avec succès : {output_path}")

if __name__ == "__main__":
    TEST_DATA_PATH = paths["RAW_DATA_DIR"] / "data_test.csv"
    COLUMNS_PATH = paths["MODELS_DIR"] / "notebook2" / "mice" / "columns_mice.pkl"
    MODELS_DIR = paths["MODELS_DIR"]
    SUBMISSION_PATH = paths["OUTPUTS_DIR"] / "predictions" / "submission.csv"

    generate_submission_file(TEST_DATA_PATH, COLUMNS_PATH, MODELS_DIR, SUBMISSION_PATH)
