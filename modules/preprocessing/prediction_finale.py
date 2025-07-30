# Fichier : prediction_finale.py (Version finale avec Stacking)

import pandas as pd
import joblib
import json
import logging
from pathlib import Path
import numpy as np

# --- Configuration du Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def load_challenge_data(data_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """
    Charge les données de test du challenge avec le bon format.
    """
    try:
        # Charger les données de test brutes avec le bon séparateur (tabulation)
        df_test_raw = pd.read_csv(data_path, header=0, na_values="?", sep='\t', engine='python')
        
        # Créer les IDs (index + 1) et utiliser toutes les colonnes comme features
        ids_test = pd.Series(range(1, len(df_test_raw) + 1), name='ID')
        df_test_features = df_test_raw.copy()
        
        log.info(f"✅ Données de test brutes chargées ({df_test_features.shape})")
        log.info(f"📋 Colonnes disponibles : {list(df_test_features.columns[:5])}...")
        return df_test_features, ids_test

    except Exception as e:
        log.error(f"❌ Erreur critique lors du chargement des données : {e}")
        raise

def preprocess_data_stream(df_raw: pd.DataFrame, imputation_method: str, models_dir: Path) -> pd.DataFrame:
    """
    Applique le pipeline de prétraitement complet (Notebook 01) à un DataFrame brut
    pour une méthode d'imputation donnée (mice ou knn).
    """
    log.info(f"--- Démarrage du prétraitement pour '{imputation_method.upper()}' ---")
    df = df_raw.copy()
    
    # Étape 1: Imputation X4 (médiane)
    if 'X4' in df.columns:
        median_path = models_dir / "notebook1" / "median_imputer_X4.pkl"
        median_value = joblib.load(median_path)
        df['X4'].fillna(median_value, inplace=True)

    # Étape 2: Imputation MICE/KNN
    if imputation_method == "mice":
        imputer_path = models_dir / "notebook1" / imputation_method / "imputer_mice_custom.pkl"
    else:  # knn
        imputer_path = models_dir / "notebook1" / imputation_method / "imputer_knn_k7.pkl"
    
    imputer = joblib.load(imputer_path)
    df = pd.DataFrame(imputer.transform(df), columns=df.columns)
    
    # Étape 3: Transformation (Yeo-Johnson + Box-Cox)
    yj_path = models_dir / "notebook1" / imputation_method / f"{imputation_method}_transformers" / "yeo_johnson_X1_X2.pkl"
    bc_path = models_dir / "notebook1" / imputation_method / f"{imputation_method}_transformers" / "box_cox_X3.pkl"
    transformer_yj = joblib.load(yj_path)
    transformer_bc = joblib.load(bc_path)
    df[['X1', 'X2']] = transformer_yj.transform(df[['X1', 'X2']])
    df_x3 = df[['X3']].copy()
    if (df_x3['X3'] <= 0).any(): df_x3['X3'] += 1e-6
    df['X3'] = transformer_bc.transform(df_x3)
    df.rename(columns={'X1': 'X1_transformed', 'X2': 'X2_transformed', 'X3': 'X3_transformed'}, inplace=True)
    
    # Étape 4: Capping
    capping_path = models_dir / "notebook1" / imputation_method / f"capping_params_{imputation_method}.pkl"
    capping_params = joblib.load(capping_path)
    for col in ['X1_transformed', 'X2_transformed', 'X3_transformed']:
        bounds = capping_params.get(col, {})
        df[col] = np.clip(df[col], bounds.get('lower_bound'), bounds.get('upper_bound'))

    # Étape 5: Suppression de la colinéarité
    corr_path = models_dir / "notebook1" / imputation_method / f"cols_to_drop_corr_{imputation_method}.pkl"
    cols_to_drop = joblib.load(corr_path)
    df = df.drop(columns=cols_to_drop, errors='ignore')
    
    # Étape 6: Ingénierie de caractéristiques
    poly_path = models_dir / "notebook1" / imputation_method / f"poly_transformer_{imputation_method}.pkl"
    poly_transformer = joblib.load(poly_path)
    continuous_cols = ['X1_transformed', 'X2_transformed', 'X3_transformed']
    continuous_features = df[continuous_cols]
    poly_features = poly_transformer.transform(continuous_features)
    poly_feature_names = poly_transformer.get_feature_names_out(continuous_cols)
    df_poly = pd.DataFrame(poly_features, columns=poly_feature_names, index=df.index)
    df = df.drop(columns=continuous_cols).join(df_poly)
    
    log.info(f"--- Prétraitement pour '{imputation_method.upper()}' terminé. Shape final : {df.shape} ---")
    return df


def generate_submission_file(
    test_data_path: Path,
    models_dir: Path,
    output_path: Path
):
    """
    Exécute le pipeline de stacking complet pour générer le fichier de soumission.
    """
    log.info("🚀 Démarrage du processus de prédiction avec STACKING...")
    
    # --- 1. Charger les données de test brutes ---
    df_test, ids_test = load_challenge_data(test_data_path)

    # --- 2. Prétraitement en parallèle pour MICE et KNN ---
    X_test_mice_processed = preprocess_data_stream(df_test, 'mice', models_dir)
    X_test_knn_processed = preprocess_data_stream(df_test, 'knn', models_dir)

    # --- 3. Générer les prédictions des modèles de base (Niveau 0) ---
    log.info("⚙️ Génération des prédictions des modèles de base (Niveau 0)...")
    base_model_names = ["SVM", "XGBoost", "RandForest", "GradBoost"]
    meta_features = {}

    for model_name in base_model_names:
        # Prédictions sur les données MICE
        pipe_mice = joblib.load(models_dir / "notebook2" / f"pipeline_{model_name.lower()}_mice.joblib")
        meta_features[f'{model_name}_mice'] = pipe_mice.predict_proba(X_test_mice_processed)[:, 1]

        # Prédictions sur les données KNN
        pipe_knn = joblib.load(models_dir / "notebook2" / f"pipeline_{model_name.lower()}_knn.joblib")
        meta_features[f'{model_name}_knn'] = pipe_knn.predict_proba(X_test_knn_processed)[:, 1]

    df_meta_features = pd.DataFrame(meta_features)
    log.info(f"✅ Méta-caractéristiques générées. Shape : {df_meta_features.shape}")

    # --- 4. Prédiction finale avec le méta-modèle (Niveau 1) ---
    try:
        # Charger le champion du stacking (ex: la version avec refit)
        stacking_model_path = models_dir / "notebook3" / "stacking_champion_model.joblib"
        stacking_threshold_path = models_dir / "notebook3" / "stacking_champion_threshold.pkl"
        
        stacking_model = joblib.load(stacking_model_path)
        stacking_threshold = joblib.load(stacking_threshold_path)
        log.info(f"🏆 Méta-modèle de stacking chargé. Seuil optimal : {stacking_threshold:.3f}")
        
        # Prédiction finale
        final_probas = stacking_model.predict_proba(df_meta_features)[:, 1]
        final_predictions = (final_probas >= stacking_threshold).astype(int)
        log.info("✅ Prédictions finales du stacking générées.")
    except Exception as e:
        log.error(f"❌ Erreur lors de la prédiction avec le méta-modèle : {e}")
        raise

    # --- 5. Création et sauvegarde du fichier de soumission ---
    submission_df = pd.DataFrame({'ID': ids_test, 'prediction': final_predictions})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    
    log.info(f"🎉 Fichier de soumission créé avec succès : {output_path}")
    print("\n--- Récapitulatif de la soumission ---")
    print(f"Modèle utilisé : Stacking Classifier")
    print(f"Nombre de prédictions : {len(submission_df)}")
    print(f"Distribution des prédictions : \n{submission_df['prediction'].value_counts(normalize=True)}")


if __name__ == '__main__':
    # --- Configuration des chemins (absolus depuis la racine du projet) ---
    # Obtenir le chemin racine du projet (remonter de modules/preprocessing vers la racine)
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    TEST_DATA_PATH = PROJECT_ROOT / "data/raw/data_test.csv"
    MODELS_DIR = PROJECT_ROOT / "models"
    SUBMISSION_PATH = PROJECT_ROOT / "outputs/predictions/submission_stacking.csv"
    
    log.info(f"📁 Répertoire racine du projet : {PROJECT_ROOT}")
    log.info(f"📁 Chemin des données de test : {TEST_DATA_PATH}")
    log.info(f"📁 Répertoire des modèles : {MODELS_DIR}")
    log.info(f"📁 Chemin de sortie : {SUBMISSION_PATH}")
    
    # --- Lancement du processus ---
    generate_submission_file(
        test_data_path=TEST_DATA_PATH,
        models_dir=MODELS_DIR,
        output_path=SUBMISSION_PATH
    )