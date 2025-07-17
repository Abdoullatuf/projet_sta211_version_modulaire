# modules/modeling/generate_final_predictions.py
"""
🎯 MODULE DE PRÉDICTIONS FINALES POUR LE CHALLENGE STA211
========================================================

Module simplifié basé sur l'analyse des notebooks 01, 02 et 03.
Encapsule toute la logique de prédiction finale pour le challenge.

Pipeline complet :
1. Notebook 01 : Transformations optimales (Yeo-Johnson + Box-Cox)
2. Notebook 02 : Imputation KNN, standardisation, modélisation
3. Notebook 03 : Stacking, optimisation des seuils, prédictions finales

Auteur : Analyse des notebooks STA211 - Challenge 2025
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier

# Configuration
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constantes du projet
RANDOM_STATE = 42
OPTIMAL_K_KNN = 19
DEFAULT_THRESHOLD = 0.340

# =============================================================================
# 1. CONFIGURATION ET CHEMINS
# =============================================================================

def setup_project_paths(root_dir: Optional[str] = None) -> Dict[str, Path]:
    """
    Configure les chemins du projet STA211.
    
    Args:
        root_dir: Répertoire racine du projet (optionnel)
        
    Returns:
        Dictionnaire des chemins configurés
    """
    if root_dir is None:
        # Détection automatique du répertoire racine
        current_path = Path(__file__).resolve()
        for parent in [current_path.parent.parent.parent, *current_path.parents]:
            if (parent / "modules").exists():
                root_dir = parent
                break
        
        if root_dir is None:
            raise FileNotFoundError("Impossible de localiser le répertoire racine du projet")
    
    root = Path(root_dir)
    
    paths = {
        "ROOT_DIR": root,
        "DATA_RAW": root / "data" / "raw",
        "DATA_PROCESSED": root / "data" / "processed",
        "MODELS_DIR": root / "models",
        "NOTEBOOKS_MODELS": root / "notebooks" / "models",
        "OUTPUTS_DIR": root / "outputs",
        "OUTPUTS_PREDICTIONS": root / "outputs" / "predictions"
    }
    
    # Créer les répertoires s'ils n'existent pas
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    
    return paths

# =============================================================================
# 2. CHARGEMENT DES DONNÉES
# =============================================================================

def load_test_data(test_file: Optional[str] = None) -> pd.DataFrame:
    """
    Charge les données de test pour les prédictions finales.
    
    Args:
        test_file: Chemin vers le fichier de test (optionnel)
        
    Returns:
        DataFrame des données de test
    """
    if test_file is None:
        # Chemins par défaut
        paths = setup_project_paths()
        test_candidates = [
            paths["DATA_RAW"] / "data_test.csv",
            paths["ROOT_DIR"] / "data" / "raw" / "data_test.csv",
            paths["ROOT_DIR"] / "SOUMISSION_STA211_2025" / "data" / "raw" / "data_test.csv"
        ]
        
        for candidate in test_candidates:
            if candidate.exists():
                test_file = candidate
                break
        
        if test_file is None:
            raise FileNotFoundError("Fichier data_test.csv introuvable dans les emplacements standards")
    
    logger.info(f"📊 Chargement des données de test : {test_file}")
    df_test = pd.read_csv(test_file)
    logger.info(f"✅ Données chargées : {df_test.shape}")
    
    return df_test

# =============================================================================
# 3. PREPROCESSING - TRANSFORMATIONS OPTIMALES (NOTEBOOK 01)
# =============================================================================

def load_optimal_transformers(paths: Dict[str, Path]) -> Tuple[Optional[PowerTransformer], Optional[PowerTransformer]]:
    """
    Charge les transformateurs optimaux depuis le notebook 01.
    
    Args:
        paths: Dictionnaire des chemins
        
    Returns:
        Tuple (transformer_yeo_johnson, transformer_box_cox)
    """
    yj_transformer = None
    bc_transformer = None
    
    # Chargement Yeo-Johnson (X1, X2)
    yj_path = paths["NOTEBOOKS_MODELS"] / "yeo_johnson_X1_X2.pkl"
    if yj_path.exists():
        yj_transformer = joblib.load(yj_path)
        logger.info("✅ Transformateur Yeo-Johnson (X1, X2) chargé")
    else:
        logger.warning(f"⚠️ Transformateur Yeo-Johnson manquant : {yj_path}")
    
    # Chargement Box-Cox (X3)
    bc_path = paths["NOTEBOOKS_MODELS"] / "box_cox_X3.pkl"
    if bc_path.exists():
        bc_transformer = joblib.load(bc_path)
        logger.info("✅ Transformateur Box-Cox (X3) chargé")
    else:
        logger.warning(f"⚠️ Transformateur Box-Cox manquant : {bc_path}")
    
    return yj_transformer, bc_transformer

def apply_optimal_transformations(df: pd.DataFrame, yj_transformer: Optional[PowerTransformer], 
                                bc_transformer: Optional[PowerTransformer]) -> pd.DataFrame:
    """
    Applique les transformations optimales (Notebook 01).
    
    Args:
        df: DataFrame original
        yj_transformer: Transformateur Yeo-Johnson pour X1, X2
        bc_transformer: Transformateur Box-Cox pour X3
        
    Returns:
        DataFrame avec variables transformées
    """
    logger.info("🔄 Application des transformations optimales...")
    df_transformed = df.copy()
    
    # Vérifier la présence des variables continues
    continuous_vars = ['X1', 'X2', 'X3']
    missing_vars = [var for var in continuous_vars if var not in df.columns]
    if missing_vars:
        logger.warning(f"⚠️ Variables continues manquantes : {missing_vars}")
        return df_transformed
    
    # Transformation Yeo-Johnson pour X1, X2
    if yj_transformer is not None:
        logger.info("📊 Application Yeo-Johnson sur X1, X2...")
        X1_X2_data = df[['X1', 'X2']].values
        X1_X2_transformed = yj_transformer.transform(X1_X2_data)
        df_transformed['X1_trans'] = X1_X2_transformed[:, 0]
        df_transformed['X2_trans'] = X1_X2_transformed[:, 1]
        logger.info("✅ Transformation Yeo-Johnson appliquée")
    else:
        logger.warning("⚠️ Transformateur Yeo-Johnson manquant - utilisation valeurs originales")
        df_transformed['X1_trans'] = df['X1']
        df_transformed['X2_trans'] = df['X2']
    
    # Transformation Box-Cox pour X3
    if bc_transformer is not None:
        logger.info("📊 Application Box-Cox sur X3...")
        X3_data = df[['X3']].values
        X3_transformed = bc_transformer.transform(X3_data)
        df_transformed['X3_trans'] = X3_transformed.ravel()
        logger.info("✅ Transformation Box-Cox appliquée")
    else:
        logger.warning("⚠️ Transformateur Box-Cox manquant - utilisation valeur originale")
        df_transformed['X3_trans'] = df['X3']
    
    return df_transformed

# =============================================================================
# 4. PREPROCESSING - IMPUTATION ET STANDARDISATION (NOTEBOOK 02)
# =============================================================================

def load_preprocessing_pipeline(paths: Dict[str, Path]) -> Tuple[Optional[List[str]], Optional[object]]:
    """
    Charge le pipeline de preprocessing (colonnes + standardiser).
    
    Args:
        paths: Dictionnaire des chemins
        
    Returns:
        Tuple (feature_columns, preprocessor)
    """
    feature_columns = None
    preprocessor = None
    
    # Chargement des colonnes utilisées
    columns_path = paths["MODELS_DIR"] / "columns_knn_used.pkl"
    if columns_path.exists():
        feature_columns = joblib.load(columns_path)
        logger.info(f"✅ Colonnes chargées : {len(feature_columns)} features")
    else:
        logger.warning(f"⚠️ Colonnes manquantes : {columns_path}")
    
    # Chargement du preprocesseur
    preproc_path = paths["MODELS_DIR"] / "preproc_knn.joblib"
    if preproc_path.exists():
        preprocessor = joblib.load(preproc_path)
        logger.info("✅ Preprocesseur chargé")
    else:
        logger.warning(f"⚠️ Preprocesseur manquant : {preproc_path}")
    
    return feature_columns, preprocessor

def apply_preprocessing_pipeline(df: pd.DataFrame, feature_columns: Optional[List[str]], 
                               preprocessor: Optional[object]) -> pd.DataFrame:
    """
    Applique le pipeline de preprocessing complet.
    
    Args:
        df: DataFrame avec transformations appliquées
        feature_columns: Liste des colonnes à utiliser
        preprocessor: Pipeline de preprocessing
        
    Returns:
        DataFrame preprocessé
    """
    logger.info("🔄 Application du pipeline de preprocessing...")
    
    # Alignement des colonnes
    if feature_columns is not None:
        available_cols = set(df.columns)
        required_cols = set(feature_columns)
        missing_cols = required_cols - available_cols
        
        if missing_cols:
            logger.warning(f"⚠️ Colonnes manquantes : {len(missing_cols)} colonnes")
            # Ajouter les colonnes manquantes avec des valeurs par défaut
            for col in missing_cols:
                df[col] = 0
        
        # Sélectionner les colonnes requises
        df_aligned = df[feature_columns]
        logger.info(f"✅ Colonnes alignées : {df_aligned.shape}")
    else:
        df_aligned = df
        logger.warning("⚠️ Pas de colonnes de référence - utilisation de toutes les colonnes")
    
    # Application du preprocesseur
    if preprocessor is not None:
        X_processed = preprocessor.transform(df_aligned)
        
        # Conversion en DataFrame
        if hasattr(X_processed, 'toarray'):
            X_processed = X_processed.toarray()
        
        df_processed = pd.DataFrame(
            X_processed,
            columns=feature_columns if feature_columns else df_aligned.columns,
            index=df_aligned.index
        )
        
        # Vérifications de qualité
        nan_count = df_processed.isnull().sum().sum()
        if nan_count > 0:
            logger.warning(f"⚠️ {nan_count} valeurs NaN détectées après preprocessing")
            df_processed = df_processed.fillna(0)
        
        inf_count = np.isinf(df_processed.values).sum()
        if inf_count > 0:
            logger.warning(f"⚠️ {inf_count} valeurs infinies détectées")
            df_processed = df_processed.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        logger.info(f"✅ Preprocessing terminé : {df_processed.shape}")
        return df_processed
    
    else:
        logger.warning("⚠️ Pas de preprocesseur - retour des données alignées")
        return df_aligned

# =============================================================================
# 5. CHARGEMENT DU MODÈLE FINAL (NOTEBOOK 03)
# =============================================================================

def load_final_model(paths: Dict[str, Path]) -> Optional[object]:
    """
    Charge le modèle final optimisé.
    
    Args:
        paths: Dictionnaire des chemins
        
    Returns:
        Modèle final chargé
    """
    model_candidates = [
        paths["MODELS_DIR"] / "final_model_optimized.pkl",
        paths["MODELS_DIR"] / "model_final_knn_stacking_best_model.joblib",
        paths["MODELS_DIR"] / "rf_knn_best_model_seuil_optimise.joblib",
        paths["MODELS_DIR"] / "best_rf_knn.joblib"
    ]
    
    for model_path in model_candidates:
        if model_path.exists():
            model = joblib.load(model_path)
            logger.info(f"✅ Modèle final chargé : {model_path.name}")
            return model
    
    logger.error("❌ Aucun modèle final trouvé")
    return None

# =============================================================================
# 6. OPTIMISATION DES SEUILS
# =============================================================================

def optimize_prediction_threshold(y_proba: np.ndarray, default_threshold: float = DEFAULT_THRESHOLD) -> float:
    """
    Optimise le seuil de prédiction (version simplifiée).
    
    Args:
        y_proba: Probabilités prédites
        default_threshold: Seuil par défaut
        
    Returns:
        Seuil optimal
    """
    # Pour les prédictions finales, on utilise le seuil optimisé dans les notebooks
    # Cette fonction pourrait être étendue pour une optimisation dynamique
    return default_threshold

# =============================================================================
# 7. GÉNÉRATION DES PRÉDICTIONS FINALES
# =============================================================================

def generate_final_predictions(model: object, X_test: pd.DataFrame, 
                             optimal_threshold: float = DEFAULT_THRESHOLD) -> Dict[str, np.ndarray]:
    """
    Génère les prédictions finales.
    
    Args:
        model: Modèle entraîné
        X_test: Données de test preprocessées
        optimal_threshold: Seuil optimal
        
    Returns:
        Dictionnaire avec prédictions et probabilités
    """
    logger.info("🎯 Génération des prédictions finales...")
    
    # Prédictions probabilistes
    y_proba = model.predict_proba(X_test)
    probabilities = y_proba[:, 1] if y_proba.shape[1] == 2 else y_proba.ravel()
    
    # Prédictions binaires avec seuil optimal
    predictions_binary = (probabilities >= optimal_threshold).astype(int)
    
    # Conversion en labels textuels
    label_mapping = {0: 'noad.', 1: 'ad.'}
    predictions_labels = np.array([label_mapping[pred] for pred in predictions_binary])
    
    # Statistiques
    n_positive = np.sum(predictions_binary)
    n_total = len(predictions_binary)
    positive_rate = n_positive / n_total * 100
    
    logger.info(f"📈 Résultats des prédictions :")
    logger.info(f"   • Total échantillons : {n_total}")
    logger.info(f"   • Publicités (ad.) : {n_positive} ({positive_rate:.1f}%)")
    logger.info(f"   • Non-publicités (noad.) : {n_total - n_positive} ({100 - positive_rate:.1f}%)")
    logger.info(f"   • Probabilité moyenne : {probabilities.mean():.3f}")
    logger.info(f"   • Seuil utilisé : {optimal_threshold:.3f}")
    
    return {
        'predictions_labels': predictions_labels,
        'predictions_binary': predictions_binary,
        'probabilities': probabilities,
        'threshold': optimal_threshold,
        'n_positive': n_positive,
        'n_total': n_total,
        'positive_rate': positive_rate
    }

# =============================================================================
# 8. SAUVEGARDE DES RÉSULTATS
# =============================================================================

def save_predictions(predictions: Dict[str, np.ndarray], paths: Dict[str, Path]) -> List[str]:
    """
    Sauvegarde les prédictions dans différents formats.
    
    Args:
        predictions: Dictionnaire des prédictions
        paths: Dictionnaire des chemins
        
    Returns:
        Liste des fichiers créés
    """
    logger.info("💾 Sauvegarde des prédictions...")
    
    output_dir = paths["OUTPUTS_PREDICTIONS"]
    output_dir.mkdir(parents=True, exist_ok=True)
    
    files_created = []
    
    # 1. Fichier principal de soumission
    submission_df = pd.DataFrame({'prediction': predictions['predictions_labels']})
    submission_file = output_dir / "submission.csv"
    submission_df.to_csv(submission_file, index=False)
    files_created.append(str(submission_file))
    
    # 2. Fichier détaillé avec probabilités
    detailed_df = pd.DataFrame({
        'prediction': predictions['predictions_labels'],
        'prediction_binary': predictions['predictions_binary'],
        'probability': predictions['probabilities']
    })
    detailed_file = output_dir / "predictions_detaillees.csv"
    detailed_df.to_csv(detailed_file, index=False)
    files_created.append(str(detailed_file))
    
    # 3. Fichier de métadonnées
    metadata = {
        'threshold': predictions['threshold'],
        'n_positive': predictions['n_positive'],
        'n_total': predictions['n_total'],
        'positive_rate': predictions['positive_rate'],
        'mean_probability': predictions['probabilities'].mean()
    }
    
    metadata_df = pd.DataFrame([metadata])
    metadata_file = output_dir / "predictions_metadata.csv"
    metadata_df.to_csv(metadata_file, index=False)
    files_created.append(str(metadata_file))
    
    logger.info(f"✅ Fichiers créés :")
    for file in files_created:
        file_path = Path(file)
        size_kb = file_path.stat().st_size / 1024
        logger.info(f"   • {file_path.name} ({size_kb:.1f} KB)")
    
    return files_created

# =============================================================================
# 9. PIPELINE COMPLET
# =============================================================================

def run_complete_prediction_pipeline(test_file: Optional[str] = None, 
                                   output_dir: Optional[str] = None,
                                   threshold: float = DEFAULT_THRESHOLD,
                                   verbose: bool = True) -> Dict[str, Union[bool, str, List[str]]]:
    """
    Exécute le pipeline complet de prédictions finales.
    
    Args:
        test_file: Chemin vers le fichier de test (optionnel)
        output_dir: Répertoire de sortie (optionnel)
        threshold: Seuil de prédiction
        verbose: Affichage détaillé
        
    Returns:
        Dictionnaire avec le statut et les résultats
    """
    try:
        if verbose:
            logger.info("🚀 DÉMARRAGE DU PIPELINE DE PRÉDICTIONS FINALES")
            logger.info("=" * 60)
        
        # 1. Configuration des chemins
        paths = setup_project_paths()
        if output_dir:
            paths["OUTPUTS_PREDICTIONS"] = Path(output_dir)
        
        # 2. Chargement des données de test
        df_test = load_test_data(test_file)
        
        # 3. Chargement des transformateurs (Notebook 01)
        yj_transformer, bc_transformer = load_optimal_transformers(paths)
        
        # 4. Application des transformations optimales
        df_transformed = apply_optimal_transformations(df_test, yj_transformer, bc_transformer)
        
        # 5. Chargement du pipeline de preprocessing (Notebook 02)
        feature_columns, preprocessor = load_preprocessing_pipeline(paths)
        
        # 6. Application du preprocessing
        X_test_processed = apply_preprocessing_pipeline(df_transformed, feature_columns, preprocessor)
        
        # 7. Chargement du modèle final (Notebook 03)
        final_model = load_final_model(paths)
        if final_model is None:
            raise ValueError("Modèle final non trouvé")
        
        # 8. Génération des prédictions
        predictions = generate_final_predictions(final_model, X_test_processed, threshold)
        
        # 9. Sauvegarde des résultats
        files_created = save_predictions(predictions, paths)
        
        if verbose:
            logger.info("=" * 60)
            logger.info("🏆 PIPELINE TERMINÉ AVEC SUCCÈS !")
            logger.info(f"📊 Résultats : {predictions['n_positive']}/{predictions['n_total']} publicités prédites")
            logger.info(f"📁 Fichiers créés : {len(files_created)}")
        
        return {
            'success': True,
            'n_predictions': predictions['n_total'],
            'n_positive': predictions['n_positive'],
            'positive_rate': predictions['positive_rate'],
            'threshold': threshold,
            'files_created': files_created
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur dans le pipeline : {e}")
        return {
            'success': False,
            'error': str(e),
            'files_created': []
        }

# =============================================================================
# 10. FONCTION PRINCIPALE
# =============================================================================

def main():
    """
    Fonction principale pour exécuter les prédictions finales.
    """
    print("🎯 PRÉDICTIONS FINALES - CHALLENGE STA211 2025")
    print("=" * 50)
    print("📋 Pipeline basé sur l'analyse des notebooks 01, 02, 03")
    print("🔧 Transformations optimales + Imputation KNN + Modèle final")
    print("=" * 50)
    
    # Exécution du pipeline complet
    results = run_complete_prediction_pipeline()
    
    if results['success']:
        print("\n✅ SUCCÈS - Prédictions générées !")
        print(f"📊 {results['n_positive']}/{results['n_predictions']} publicités prédites")
        print(f"🎯 Taux de publicités : {results['positive_rate']:.1f}%")
        print(f"📁 Fichiers créés : {len(results['files_created'])}")
        
        for file in results['files_created']:
            print(f"   • {Path(file).name}")
    else:
        print(f"\n❌ ÉCHEC - Erreur : {results['error']}")
    
    return results

# =============================================================================
# POINT D'ENTRÉE
# =============================================================================

if __name__ == "__main__":
    main()