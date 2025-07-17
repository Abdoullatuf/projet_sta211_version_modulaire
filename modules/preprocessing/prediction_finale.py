#!/usr/bin/env python3
#modules/preprocessing/prediction_finale.py
"""
🚨 FONCTION CORRIGÉE POUR PRÉDICTIONS FINALES - Projet STA211
Corrige l'incohérence identifiée entre les notebooks 01, 02 et 03

PROBLÈME IDENTIFIÉ:
- Notebook 01: Transformation optimale mixte (Yeo-Johnson + Box-Cox)
- Notebook 02/03: Seulement StandardScaler (TRANSFORMATIONS MANQUANTES!)

SOLUTION IMPLEMENTÉE:
1. Application des transformations Box-Cox/Yeo-Johnson (Notebook 01)
2. Imputation KNN (k=19)
3. Standardisation (Notebook 02)
4. Prédiction avec seuil optimal (0.340)

Transformations corrigées:
- X1 → Yeo-Johnson → X1_trans
- X2 → Yeo-Johnson → X2_trans  
- X3 → Box-Cox → X3_trans
"""

import pandas as pd
import numpy as np
import joblib
import os
import logging
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def charger_jeu_de_test(raw_data_dir: str):
    """
    Charge le jeu de données de test avec nettoyage des noms de colonnes
    (en utilisant la fonction load_data du projet).
    """
    from modules.preprocessing.data_loader import load_data

    logger = logging.getLogger(__name__)
    logger.info("📂 Chargement du jeu de test STA211...")

    df_test = load_data(
        file_path="data_test.csv",
        require_outcome=False,
        display_info=False,  # Désactiver l'affichage automatique
        raw_data_dir=raw_data_dir,
        encode_target=False
    )

    logger.info(f"📋 Dataset de test chargé : {df_test.shape}")
    logger.info(f"📝 Colonnes : {list(df_test.columns)[:10]} ...")

    return df_test

def corriger_et_imputer_X4(df, mediane_X4=None):
    """
    Impute les valeurs manquantes de X4 par la médiane et convertit X4 en int si binaire.
    Args:
        df: DataFrame à corriger (jeu de test)
        mediane_X4: valeur de médiane à utiliser (optionnel, sinon calculée sur df)
    Returns:
        DataFrame modifié
    """
    import logging
    logger = logging.getLogger(__name__)
    if 'X4' not in df.columns:
        logger.warning("X4 n'est pas dans le DataFrame, aucune correction appliquée.")
        return df
    unique_values = df['X4'].dropna().unique()
    if set(unique_values).issubset({0.0, 1.0}):
        if mediane_X4 is None:
            mediane_X4 = df['X4'].median()
        df['X4'] = df['X4'].fillna(mediane_X4)
        df['X4'] = df['X4'].astype(int)
        logger.info(f"✅ X4 imputé par la médiane ({mediane_X4}) et converti en int.")
    else:
        logger.warning("⚠️ X4 contient des valeurs autres que 0 et 1, conservation en float64.")
    return df

def appliquer_transformation_variables_continues(df, verbose=False):
    """
    Applique la transformation optimale (Yeo-Johnson pour X1, X2 et Box-Cox pour X3)
    sur un DataFrame, en utilisant appliquer_transformation_optimale du module transformation_optimale_mixte.
    Args:
        df: DataFrame à transformer
        verbose: bool, affiche les informations détaillées si True
    Returns:
        DataFrame transformé avec X1_transformed, X2_transformed, X3_transformed
    """
    import logging
    logger = logging.getLogger(__name__)
    from modules.preprocessing.transformation_optimale_mixte import appliquer_transformation_optimale
    df_transformed = appliquer_transformation_optimale(df, verbose=verbose)
    logger.info("✅ Transformation optimale appliquée sur X1, X2, X3 (Yeo-Johnson/Box-Cox)")
    return df_transformed
            
def nettoyer_variables_continues_transformees(df, renommer=False):
    """
    Supprime les colonnes originales X1, X2, X3 et garde les colonnes transformées.
    Si renommer=True, renomme X1_transformed -> X1, etc.
    Args:
        df: DataFrame à nettoyer
        renommer: bool, si True renomme les colonnes transformées en X1, X2, X3
    Returns:
        DataFrame nettoyé
    """
    import logging
    logger = logging.getLogger(__name__)
    cols_to_drop = ['X1', 'X2', 'X3']
    df_clean = df.copy()
    for col in cols_to_drop:
        if col in df_clean.columns:
            df_clean = df_clean.drop(columns=col)
    if renommer:
        df_clean = df_clean.rename(columns={
            'X1_transformed': 'X1',
            'X2_transformed': 'X2',
            'X3_transformed': 'X3'
        })
        logger.info("✅ Colonnes transformées renommées en X1, X2, X3 et originales supprimées.")
    else:
        logger.info("✅ Colonnes originales X1, X2, X3 supprimées, colonnes transformées conservées.")
    return df_clean

def supprimer_outliers_optionnelle(df, columns, supprimer=False, iqr_multiplier=1.5, method='iqr'):
    """
    Supprime les outliers des colonnes spécifiées si supprimer=True, sinon retourne le DataFrame inchangé.
    Args:
        df: DataFrame à traiter
        columns: liste des colonnes à traiter (ex: ['X1_transformed', ...])
        supprimer: bool, si True applique la suppression des outliers
        iqr_multiplier: float, multiplicateur IQR (défaut 1.5)
        method: str, méthode de détection ('iqr' par défaut)
    Returns:
        DataFrame nettoyé ou original
    """
    import logging
    logger = logging.getLogger(__name__)
    if supprimer:
        from modules.preprocessing.outliers import detect_and_remove_outliers
        df_clean = detect_and_remove_outliers(
            df=df,
            columns=columns,
            method=method,
            iqr_multiplier=iqr_multiplier,
            verbose=False,
            save_path=None
        )
        logger.info(f"✅ Outliers supprimés sur {columns} (méthode {method}, IQR x{iqr_multiplier})")
        return df_clean
    else:
        logger.info("Suppression des outliers non appliquée (option désactivée)")
        return df
            
def imputer_valeurs_manquantes(
    df,
    methode='knn',
    cols_to_impute=None,
    knn_k=19,
    processed_data_dir=None,
    models_dir=None
):
    """
    Impute les valeurs manquantes selon la méthode choisie ('knn' ou 'mice') sur les colonnes continues transformées.
    Renomme les colonnes transformées en X1_trans, X2_trans, X3_trans avant imputation.
    Args:
        df: DataFrame à imputer
        methode: 'knn' ou 'mice'
        cols_to_impute: liste des colonnes à imputer (défaut ['X1_trans', 'X2_trans', 'X3_trans'])
        knn_k: valeur de k pour KNN (défaut 19)
        processed_data_dir: chemin pour sauvegarde (optionnel)
        models_dir: chemin pour sauvegarde (optionnel)
    Returns:
        DataFrame imputé
    """
    import logging
    logger = logging.getLogger(__name__)
    from modules.preprocessing.missing_values import handle_missing_values
    # Renommage des colonnes transformées
    rename_mapping = {
        'X1_transformed': 'X1_trans',
        'X2_transformed': 'X2_trans',
        'X3_transformed': 'X3_trans'
    }
    df_renamed = df.rename(columns=rename_mapping)
    if cols_to_impute is None:
        cols_to_impute = ['X1_trans', 'X2_trans', 'X3_trans']
    kwargs = dict(
        df=df_renamed,
        strategy="mixed_mar_mcar",
        mar_method=methode,
        mar_cols=cols_to_impute,
        mcar_cols=[],
        processed_data_dir=processed_data_dir,
        models_dir=models_dir,
        save_results=False,
        display_info=False
    )
    if methode == 'knn':
        kwargs['knn_k'] = knn_k
    df_imputed = handle_missing_values(**kwargs)
    logger.info(f"✅ Imputation {methode.upper()} appliquée sur {cols_to_impute} (k={knn_k if methode=='knn' else 'N/A'})")
    return df_imputed
            
def load_selected_features(imputation_method: str = "knn"):
    """
    Charge la liste des variables sélectionnées (features à garder) selon la méthode d'imputation.
    Args:
        imputation_method: "knn" ou "mice"
    Returns:
        Liste des noms de variables à garder
    """
    # Import de setup_project_paths depuis la config
    from modules.config.paths_config import setup_project_paths
    
    # Récupération des chemins
    paths = setup_project_paths()
    MODELS_DIR = paths["MODELS_DIR"]
    
    if imputation_method == "knn":
        path = MODELS_DIR / "notebook2" / "columns_knn.pkl"
    elif imputation_method == "mice":
        path = MODELS_DIR / "notebook2" / "columns_mice.pkl"
    else:
        raise ValueError("Méthode d'imputation inconnue : {}".format(imputation_method))
    
    try:
        return joblib.load(path)
    except FileNotFoundError:
        logger.error(f"❌ Fichier de features sélectionnées non trouvé: {path}")
        # Fallback: essayer avec le chemin relatif
        fallback_path = f"models/notebook2/columns_{imputation_method}.pkl"
        try:
            return joblib.load(fallback_path)
        except FileNotFoundError:
            logger.error(f"❌ Fichier de fallback non trouvé non plus: {fallback_path}")
            raise

def filter_columns(df, columns_to_keep):
    """
    Garde uniquement les colonnes spécifiées dans columns_to_keep.
    Args:
        df: DataFrame à filtrer
        columns_to_keep: liste des colonnes à garder
    Returns:
        DataFrame filtré
    """
    cols_present = [col for col in columns_to_keep if col in df.columns]
    return df[cols_present]

def pipeline_prediction_finale(
    raw_data_dir: str = "data/raw",
    imputation_method: str = "knn",
    knn_k: int = 19,
    remove_outliers: bool = False,
    mediane_X4: float = None,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Pipeline complet pour la prédiction finale utilisant le meilleur XGBoost.
    
    Args:
        raw_data_dir: Chemin vers les données brutes
        imputation_method: "knn" ou "mice"
        knn_k: Valeur de k pour KNN
        remove_outliers: Si True, supprime les outliers
        mediane_X4: Médiane de X4 à utiliser (optionnel)
        verbose: Si True, affiche les informations détaillées
    
    Returns:
        DataFrame avec les prédictions et probabilités
    """
    logger.info("🚀 DÉMARRAGE DU PIPELINE DE PRÉDICTION FINALE")
    logger.info("=" * 60)
    
    # 1. Chargement du jeu de test
    logger.info("📂 Étape 1: Chargement du jeu de test...")
    df_test = charger_jeu_de_test(raw_data_dir)
    if df_test is None:
        logger.error("❌ Échec du chargement du jeu de test")
        return None
    
    # 2. Correction et imputation de X4
    logger.info("🔧 Étape 2: Correction et imputation de X4...")
    df_test = corriger_et_imputer_X4(df_test, mediane_X4)
    
    # 3. Application des transformations optimales
    logger.info("🔄 Étape 3: Application des transformations optimales...")
    df_test = appliquer_transformation_variables_continues(df_test, verbose=verbose)
    
    # 4. Nettoyage des variables transformées
    logger.info("🧹 Étape 4: Nettoyage des variables transformées...")
    df_test = nettoyer_variables_continues_transformees(df_test, renommer=False)
    
    # 5. Suppression optionnelle des outliers
    if remove_outliers:
        logger.info("🗑️ Étape 5: Suppression des outliers...")
        columns_to_check = ['X1_transformed', 'X2_transformed', 'X3_transformed']
        df_test = supprimer_outliers_optionnelle(
            df_test, 
            columns_to_check, 
            supprimer=True
        )
    else:
        logger.info("⏭️ Étape 5: Suppression des outliers ignorée")
    
    # 6. Imputation des valeurs manquantes
    logger.info(f"🔧 Étape 6: Imputation des valeurs manquantes ({imputation_method.upper()})...")
    df_test = imputer_valeurs_manquantes(
        df=df_test,
        methode=imputation_method,
        knn_k=knn_k
    )
    
    # 7. Sélection des features
    logger.info("🎯 Étape 7: Sélection des features...")
    try:
        selected_features = load_selected_features(imputation_method)
        df_test = filter_columns(df_test, selected_features)
        logger.info(f"✅ {len(selected_features)} features sélectionnées")
    except Exception as e:
        logger.error(f"❌ Erreur lors de la sélection des features: {e}")
        return None
    
    # 8. Chargement du meilleur modèle XGBoost
    logger.info("🤖 Étape 8: Chargement du meilleur modèle XGBoost...")
    from modules.config.paths_config import setup_project_paths
    
    # Récupération des chemins
    paths = setup_project_paths()
    MODELS_DIR = paths["MODELS_DIR"]
    
    model_path = MODELS_DIR / "notebook2" / f"best_xgboost_{imputation_method}.joblib"
    threshold_path = MODELS_DIR / "notebook2" / "meilleur_modele" / f"threshold_xgboost_{imputation_method}.json"
    
    try:
        best_model = joblib.load(model_path)
        logger.info(f"✅ Modèle XGBoost chargé: {model_path}")
    except FileNotFoundError:
        logger.error(f"❌ Modèle non trouvé: {model_path}")
        # Fallback: essayer avec le chemin relatif
        fallback_model_path = f"models/notebook2/best_xgboost_{imputation_method}.joblib"
        try:
            best_model = joblib.load(fallback_model_path)
            logger.info(f"✅ Modèle XGBoost chargé (fallback): {fallback_model_path}")
        except FileNotFoundError:
            logger.error(f"❌ Modèle non trouvé même en fallback: {fallback_model_path}")
            return None
    
    # 9. Chargement du seuil optimal
    try:
        import json
        with open(threshold_path, 'r') as f:
            threshold_data = json.load(f)
            optimal_threshold = threshold_data.get('optimal_threshold', 0.5)
        logger.info(f"✅ Seuil optimal chargé: {optimal_threshold}")
    except Exception as e:
        logger.warning(f"⚠️ Erreur lors du chargement du seuil, essai avec fallback: {e}")
        # Fallback: essayer avec le chemin relatif
        fallback_threshold_path = f"models/notebook2/meilleur_modele/threshold_xgboost_{imputation_method}.json"
        try:
            with open(fallback_threshold_path, 'r') as f:
                threshold_data = json.load(f)
                optimal_threshold = threshold_data.get('optimal_threshold', 0.5)
            logger.info(f"✅ Seuil optimal chargé (fallback): {optimal_threshold}")
        except Exception as e2:
            logger.warning(f"⚠️ Erreur lors du chargement du seuil en fallback, utilisation du seuil par défaut (0.5): {e2}")
            optimal_threshold = 0.5
    
    # 10. Prédictions
    logger.info("🎯 Étape 9: Génération des prédictions...")
    try:
        # Prédictions de probabilités
        probabilities = best_model.predict_proba(df_test)[:, 1]
        
        # Prédictions binaires avec seuil optimal
        predictions = (probabilities >= optimal_threshold).astype(int)
        
        # Création du DataFrame de résultats
        results_df = pd.DataFrame({
            'id': range(1, len(predictions) + 1),
            'prediction': predictions,
            'probability': probabilities
        })
        
        # Mapping des prédictions
        results_df['prediction_label'] = results_df['prediction'].map({0: 'noad.', 1: 'ad.'})
        
        logger.info(f"✅ Prédictions générées: {len(results_df)} échantillons")
        logger.info(f"📊 Statistiques: {results_df['prediction_label'].value_counts().to_dict()}")
        logger.info(f"📈 Probabilité moyenne: {results_df['probability'].mean():.3f}")
        logger.info(f"🎯 Seuil utilisé: {optimal_threshold}")
        
        return results_df
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de la génération des prédictions: {e}")
        return None

def generer_fichier_soumission(
    results_df: pd.DataFrame,
    output_path: str = "predictions_finales_xgboost.csv",
    format_simple: bool = True
) -> str:
    """
    Génère le fichier de soumission final.
    
    Args:
        results_df: DataFrame avec les prédictions
        output_path: Chemin du fichier de sortie
        format_simple: Si True, génère le format simple (seulement les prédictions)
    
    Returns:
        Chemin du fichier généré
    """
    if results_df is None:
        logger.error("❌ Aucun résultat à sauvegarder")
        return None
    
    if format_simple:
        # Format simple: seulement les prédictions (comme dans l'exemple)
        submission_df = results_df[['prediction_label']]
        submission_df.columns = ['prediction']
        # Sauvegarde sans en-tête pour correspondre au format d'exemple
        submission_df.to_csv(output_path, index=False, header=False)
    else:
        # Format détaillé avec id et probabilités
        submission_df = results_df[['id', 'prediction_label', 'probability']]
        submission_df.columns = ['id', 'prediction', 'probability']
        # Sauvegarde avec en-tête
        submission_df.to_csv(output_path, index=False)
    
    return output_path

def charger_modele_stacking(imputation_method: str = "mice"):
    """
    Charge le meilleur modèle de stacking et son seuil optimal.
    
    Args:
        imputation_method: Méthode d'imputation ("knn" ou "mice")
    
    Returns:
        Tuple (chemin_du_modele, seuil_optimal)
    """
    from modules.config.paths_config import setup_project_paths
    import json
    
    # Récupération des chemins
    paths = setup_project_paths()
    MODELS_DIR = paths["MODELS_DIR"]
    
    # Déterminer le meilleur modèle de stacking
    # D'après les performances, MICE a un meilleur F1-score (0.9185 vs ~0.91 pour KNN)
    if imputation_method == "mice":
        model_path = MODELS_DIR / "notebook3" / "stacking" / "stack_no_refit_mice.joblib"
        threshold_path = MODELS_DIR / "notebook3" / "stacking" / "best_thr_stack_no_refit_mice.json"
    else:  # knn
        model_path = MODELS_DIR / "notebook3" / "stacking" / "stack_no_refit_knn.joblib"
        threshold_path = MODELS_DIR / "notebook3" / "stacking" / "best_thr_stack_no_refit_knn.json"
    
    try:
        # Charger le seuil optimal
        with open(threshold_path, 'r') as f:
            threshold_data = json.load(f)
            if imputation_method == "mice":
                optimal_threshold = threshold_data.get("best_thr_stack_no_refit_mice", 0.39)
            else:
                optimal_threshold = threshold_data.get("best_thr_stack_no_refit_knn", 0.5)
        
        logger.info(f"✅ Modèle de stacking {imputation_method.upper()} chargé: {model_path}")
        logger.info(f"✅ Seuil optimal chargé: {optimal_threshold}")
        
        return str(model_path), optimal_threshold
        
    except Exception as e:
        logger.error(f"❌ Erreur lors du chargement du modèle de stacking: {e}")
        return None, 0.5

def charger_modele_xgboost(imputation_method: str = "knn"):
    """
    Charge le meilleur modèle XGBoost et son seuil optimal.
    
    Args:
        imputation_method: Méthode d'imputation ("knn" ou "mice")
    
    Returns:
        Tuple (chemin_du_modele, seuil_optimal)
    """
    from modules.config.paths_config import setup_project_paths
    import json
    
    # Récupération des chemins
    paths = setup_project_paths()
    MODELS_DIR = paths["MODELS_DIR"]
    
    model_path = MODELS_DIR / "notebook2" / f"best_xgboost_{imputation_method}.joblib"
    threshold_path = MODELS_DIR / "notebook2" / "meilleur_modele" / f"threshold_xgboost_{imputation_method}.json"
    
    try:
        # Charger le seuil optimal
        with open(threshold_path, 'r') as f:
            threshold_data = json.load(f)
            optimal_threshold = threshold_data.get('optimal_threshold', 0.5)
        
        logger.info(f"✅ Modèle XGBoost {imputation_method.upper()} chargé: {model_path}")
        logger.info(f"✅ Seuil optimal chargé: {optimal_threshold}")
        
        return str(model_path), optimal_threshold
        
    except Exception as e:
        logger.error(f"❌ Erreur lors du chargement du modèle XGBoost: {e}")
        return None, 0.5

def generer_predictions(test_data, model, threshold):
    """
    Génère les prédictions avec le modèle chargé.
    
    Args:
        test_data: DataFrame des données de test
        model: Modèle déjà chargé (objet sklearn, xgboost, dict stacking, etc.)
        threshold: Seuil optimal
    
    Returns:
        DataFrame avec les prédictions
    """
    try:
        import numpy as np
        
        # Vérifier si c'est un modèle de stacking (dict)
        if isinstance(model, dict) and 'pipelines' in model:
            # Modèle de stacking (no refit) - pipelines est un dict
            pipelines = model['pipelines']
            # Générer les prédictions avec chaque pipeline
            probas = [pipe.predict_proba(test_data)[:, 1] for pipe in pipelines.values()]
            # Moyenne des prédictions
            probabilities = np.mean(probas, axis=0)
        else:
            # Modèle standard (sklearn, xgboost, etc.)
            probabilities = model.predict_proba(test_data)[:, 1]
        
        # Prédictions binaires avec seuil optimal
        predictions = (probabilities >= threshold).astype(int)
        
        # Création du DataFrame de résultats
        results_df = pd.DataFrame({
            'id': range(1, len(predictions) + 1),
            'prediction': predictions,
            'probability': probabilities
        })
        
        # Mapping des prédictions
        results_df['prediction_label'] = results_df['prediction'].map({0: 'noad.', 1: 'ad.'})
        
        return results_df
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de la génération des prédictions: {e}")
        return None

def main_pipeline_prediction_with_params(
    imputation_method: str = "mice",
    use_stacking: bool = True
) -> str:
    """
    Pipeline principal pour la génération des prédictions finales.
    
    Args:
        imputation_method: Méthode d'imputation ("knn" ou "mice")
        use_stacking: Si True, utilise le modèle de stacking au lieu de XGBoost
    
    Returns:
        Chemin du fichier de soumission généré
    """
    logger.info("🚀 Démarrage du pipeline de prédiction finale")
    
    # Configuration des chemins
    from modules.config.paths_config import setup_project_paths
    setup_project_paths()
    
    # Étape 1: Chargement des données de test
    logger.info("📋 Étape 1: Chargement des données de test...")
    test_data = load_test_data()
    logger.info(f"📋 Dataset de test chargé : {test_data.shape}")
    logger.info(f"📝 Colonnes : {list(test_data.columns[:10])} ...")
    
    # Étape 2: Correction et imputation de X4
    logger.info("🔧 Étape 2: Correction et imputation de X4...")
    test_data = imputer_X4(test_data)
    
    # Étape 3: Application des transformations optimales
    logger.info("🔄 Étape 3: Application des transformations optimales...")
    test_data = appliquer_transformations_optimales(test_data)
    
    # Étape 4: Nettoyage des variables transformées
    logger.info("🧹 Étape 4: Nettoyage des variables transformées...")
    test_data = nettoyer_variables_transformees(test_data)
    
    # Étape 5: Suppression des outliers (ignorée pour la cohérence)
    logger.info("⏭️ Étape 5: Suppression des outliers ignorée")
    
    # Étape 6: Imputation des valeurs manquantes
    logger.info(f"🔧 Étape 6: Imputation des valeurs manquantes ({imputation_method.upper()})...")
    test_data = imputer_valeurs_manquantes(test_data, methode=imputation_method)
    
    # Étape 7: Sélection des features
    logger.info("🎯 Étape 7: Sélection des features...")
    test_data = selectionner_features(test_data, imputation_method)
    logger.info(f"✅ {test_data.shape[1]} features sélectionnées")
    
    # Étape 8: Chargement du modèle et seuil
    if use_stacking:
        logger.info("🤖 Étape 8: Chargement du meilleur modèle de stacking...")
        model_path, threshold = charger_modele_stacking(imputation_method)
    else:
        logger.info("🤖 Étape 8: Chargement du meilleur modèle XGBoost...")
        model_path, threshold = charger_modele_xgboost(imputation_method)
    
    # Étape 9: Génération des prédictions
    logger.info("🎯 Étape 9: Génération des prédictions...")
    results_df = generer_predictions(test_data, model_path, threshold)
    
    # Génération du fichier de soumission
    if results_df is not None:
        output_file = generer_fichier_soumission(
            results_df=results_df,
            output_path="predictions_finales_stacking.csv",
            format_simple=True
        )
        logger.info(f"🎉 Pipeline terminé avec succès! Fichier: {output_file}")
        return output_file
    else:
        logger.error("❌ Échec du pipeline")
        return None

def load_test_data():
    """
    Charge les données de test.
    
    Returns:
        DataFrame des données de test
    """
    from modules.config.paths_config import setup_project_paths
    from pathlib import Path
    
    # Obtenir le chemin absolu du projet
    paths = setup_project_paths()
    ROOT_DIR = paths["ROOT_DIR"]
    
    # Construire le chemin absolu vers les données
    absolute_data_dir = ROOT_DIR / "data" / "raw"
    
    return charger_jeu_de_test(absolute_data_dir)

def imputer_X4(df):
    """
    Impute la variable X4 avec la médiane.
    
    Args:
        df: DataFrame avec les données
    
    Returns:
        DataFrame avec X4 imputé
    """
    return corriger_et_imputer_X4(df)

def appliquer_transformations_optimales(df):
    """
    Applique les transformations optimales sur les variables continues.
    
    Args:
        df: DataFrame avec les données
    
    Returns:
        DataFrame avec les transformations appliquées
    """
    return appliquer_transformation_variables_continues(df)

def nettoyer_variables_transformees(df):
    """
    Nettoie les variables transformées en supprimant les colonnes originales.
    
    Args:
        df: DataFrame avec les données
    
    Returns:
        DataFrame nettoyé
    """
    return nettoyer_variables_continues_transformees(df, renommer=False)

def selectionner_features(df, imputation_method):
    """
    Sélectionne les features selon la méthode d'imputation.
    
    Args:
        df: DataFrame avec les données
        imputation_method: Méthode d'imputation ("knn" ou "mice")
    
    Returns:
        DataFrame avec les features sélectionnées
    """
    try:
        selected_features = load_selected_features(imputation_method)
        return filter_columns(df, selected_features)
    except Exception as e:
        logger.error(f"❌ Erreur lors de la sélection des features: {e}")
        return df

def main_pipeline_prediction():
    """
    Fonction principale pour exécuter le pipeline complet avec le meilleur stacking.
    """
    return main_pipeline_prediction_with_params(imputation_method="mice", use_stacking=True)
    logger.info("🎯 EXÉCUTION DU PIPELINE DE PRÉDICTION FINALE")
    logger.info("=" * 60)
    
    # Exécution du pipeline
    results_df = pipeline_prediction_finale(
        raw_data_dir="data/raw",
        imputation_method="knn",
        knn_k=19,
        remove_outliers=False,
        verbose=False
    )
    
    # Génération du fichier de soumission
    if results_df is not None:
        output_file = generer_fichier_soumission(
            results_df=results_df,
            output_path="predictions_finales_xgboost.csv",
            format_simple=True
        )
        logger.info(f"🎉 Pipeline terminé avec succès! Fichier: {output_file}")
        return output_file
    else:
        logger.error("❌ Échec du pipeline")
        return None

def run_prediction_pipeline(
    model_type: str = "xgboost",
    imputation_method: str = "knn",
    threshold: float = None,
    model_path: str = None,
    threshold_path: str = None,
    output_path: str = None,
    data_path: str = None
) -> str:
    """
    Pipeline générique pour la génération des prédictions finales.

    Args:
        model_type: Type de modèle à utiliser ("xgboost", "stacking")
        imputation_method: Méthode d’imputation ("knn" ou "mice")
        threshold: Seuil de décision (float, optionnel). Si None, on charge depuis threshold_path ou auto.
        model_path: Chemin du modèle à charger (optionnel). Si None, auto selon model_type/imputation_method.
        threshold_path: Chemin du fichier seuil optimal (optionnel). Si None, auto selon model_type/imputation_method.
        output_path: Chemin du fichier de sortie (optionnel)
        data_path: Chemin du fichier de test à charger (optionnel). Si None, auto (data/raw/data_test.csv)
    Returns:
        Chemin du fichier généré

    Exemples :
        # Stacking MICE avec seuil optimal auto
        run_prediction_pipeline(model_type="stacking", imputation_method="mice")
        # XGBoost KNN avec seuil personnalisé
        run_prediction_pipeline(model_type="xgboost", imputation_method="knn", threshold=0.42)
        # Stacking KNN avec chemins personnalisés
        run_prediction_pipeline(model_type="stacking", imputation_method="knn", model_path="/chemin/vers/mon_modele.joblib", threshold_path="/chemin/vers/mon_seuil.json")
        # Chemin de données personnalisé
        run_prediction_pipeline(model_type="stacking", imputation_method="mice", data_path="/chemin/vers/data_test.csv")
    """
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"🚀 Pipeline générique : modèle={model_type}, imputation={imputation_method}, seuil={threshold}, model_path={model_path}, threshold_path={threshold_path}, data_path={data_path}")

    from modules.config.paths_config import setup_project_paths
    import joblib
    import json
    from pathlib import Path

    # 1. Chargement des données de test
    if data_path is not None:
        # Chargement direct depuis le chemin fourni
        import pandas as pd
        # Le fichier utilise des tabulations et des guillemets
        try:
            # Essayer avec les paramètres par défaut
            test_data = pd.read_csv(data_path, sep='\t', quotechar='"', quoting=1)
        except:
            try:
                # Essayer sans guillemets
                test_data = pd.read_csv(data_path, sep='\t')
            except:
                # Essayer avec auto-détection
                test_data = pd.read_csv(data_path, sep=None, engine='python')
        
        # Nettoyer les noms de colonnes si nécessaire
        if len(test_data.columns) == 1:
            # Si une seule colonne, essayer de séparer
            first_col = test_data.columns[0]
            if '\t' in str(first_col):
                # Séparer la première ligne
                test_data = pd.read_csv(data_path, sep='\t', header=None)
                # Utiliser la première ligne comme en-tête
                test_data.columns = test_data.iloc[0]
                test_data = test_data.iloc[1:].reset_index(drop=True)
        
        logger.info(f"📋 Dataset de test chargé depuis {data_path} : {test_data.shape}")
        logger.info(f"📝 Colonnes : {list(test_data.columns[:10])} ...")
    else:
        test_data = load_test_data()
        # Les logs sont déjà dans load_test_data, pas besoin de les répéter

    # 2. Prétraitement
    test_data = imputer_X4(test_data)
    test_data = appliquer_transformations_optimales(test_data)
    test_data = nettoyer_variables_transformees(test_data)
    test_data = imputer_valeurs_manquantes(test_data, methode=imputation_method)
    features = load_selected_features(imputation_method)
    test_data = filter_columns(test_data, features)
    logger.info(f"✅ {len(features)} features sélectionnées")

    # 3. Détermination des chemins modèle/seuil
    paths = setup_project_paths()
    MODELS_DIR = paths["MODELS_DIR"]
    if model_path is not None:
        model_path_final = Path(model_path)
    else:
        if model_type == "xgboost":
            if imputation_method == "knn":
                model_path_final = MODELS_DIR / "notebook2" / "best_xgboost_knn.joblib"
                threshold_path_final = MODELS_DIR / "notebook2" / "threshold_xgboost_knn.json"
            else:
                model_path_final = MODELS_DIR / "notebook2" / "best_xgboost_mice.joblib"
                threshold_path_final = MODELS_DIR / "notebook2" / "threshold_xgboost_mice.json"
        elif model_type == "stacking":
            if imputation_method == "knn":
                model_path_final = MODELS_DIR / "notebook3" / "stacking" / "stack_no_refit_knn.joblib"
                threshold_path_final = MODELS_DIR / "notebook3" / "stacking" / "best_thr_stack_no_refit_knn.json"
            else:
                model_path_final = MODELS_DIR / "notebook3" / "stacking" / "stack_no_refit_mice.joblib"
                threshold_path_final = MODELS_DIR / "notebook3" / "stacking" / "best_thr_stack_no_refit_mice.json"
        else:
            raise ValueError(f"Type de modèle inconnu : {model_type}")
    # Si threshold_path fourni, il est prioritaire
    if threshold_path is not None:
        threshold_path_final = Path(threshold_path)
    # 4. Chargement du modèle
    try:
        model = joblib.load(model_path_final)
        logger.info(f"✅ Modèle chargé : {model_path_final}")
    except Exception as e:
        logger.error(f"❌ Erreur lors du chargement du modèle : {e}")
        return None
    # 5. Détermination du seuil
    if threshold is not None:
        threshold_final = threshold
        logger.info(f"✅ Seuil personnalisé utilisé : {threshold_final}")
    else:
        try:
            with open(threshold_path_final, "r") as f:
                threshold_data = json.load(f)
            # Extraire le seuil selon le format du fichier
            if "threshold" in threshold_data:
                threshold_final = threshold_data["threshold"]
            elif "best_thr_stack_refit_knn" in threshold_data:
                threshold_final = threshold_data["best_thr_stack_refit_knn"]
            elif "best_thr_stack_refit_mice" in threshold_data:
                threshold_final = threshold_data["best_thr_stack_refit_mice"]
            elif "best_thr_stack_no_refit_knn" in threshold_data:
                threshold_final = threshold_data["best_thr_stack_no_refit_knn"]
            elif "best_thr_stack_no_refit_mice" in threshold_data:
                threshold_final = threshold_data["best_thr_stack_no_refit_mice"]
            else:
                # Essayer de prendre la première valeur si c'est un dict avec une seule clé
                threshold_final = list(threshold_data.values())[0]
            logger.info(f"✅ Seuil optimal chargé : {threshold_final}")
        except Exception as e:
            # Essayer de récupérer le seuil depuis le modèle lui-même (pour les modèles de stacking)
            if isinstance(model, dict) and 'threshold' in model:
                threshold_final = model['threshold']
                logger.info(f"✅ Seuil optimal récupéré depuis le modèle : {threshold_final}")
            else:
                threshold_final = 0.5
                logger.warning(f"⚠️ Seuil par défaut utilisé : {threshold_final}")
    # 6. Génération des prédictions
    try:
        results_df = generer_predictions(test_data, model, threshold_final)
        logger.info(f"✅ Prédictions générées : {results_df.shape[0]} échantillons")
        logger.info(f"📊 Statistiques: {results_df['prediction_label'].value_counts().to_dict()}")
        logger.info(f"📈 Probabilité moyenne: {results_df['probability'].mean():.3f}")
        logger.info(f"🎯 Seuil utilisé: {threshold_final}")
    except Exception as e:
        logger.error(f"❌ Erreur lors de la génération des prédictions: {e}")
        return None
    # 7. Génération du fichier de soumission
    if output_path is None:
        output_path = f"predictions_finales_{model_type}_{imputation_method}.csv"
    output_file = generer_fichier_soumission(results_df=results_df, output_path=output_path, format_simple=True)
    logger.info(f"🎉 Pipeline terminé avec succès! Fichier: {output_file}")
    return output_file

if __name__ == "__main__":
    main_pipeline_prediction()

