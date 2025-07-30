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

def appliquer_transformation_variables_continues(df, verbose=False, models_dir=None):
    """
    Applique la transformation optimale (Yeo-Johnson pour X1, X2 et Box-Cox pour X3)
    sur un DataFrame, en utilisant appliquer_transformation_optimale du module transformation_optimale_mixte.
    Args:
        df: DataFrame à transformer
        verbose: bool, affiche les informations détaillées si True
        models_dir: chemin vers le dossier des modèles (optionnel)
    Returns:
        DataFrame transformé avec X1_transformed, X2_transformed, X3_transformed
    """
    import logging
    logger = logging.getLogger(__name__)
    from modules.preprocessing.transformation_optimale_mixte import appliquer_transformation_optimale
    
    # Si models_dir n'est pas fourni, utiliser le chemin par défaut
    if models_dir is None:
        from modules.config.paths_config import setup_project_paths
        paths = setup_project_paths()
        models_dir = paths["MODELS_DIR"] / "notebook1"
    
    df_transformed = appliquer_transformation_optimale(df, models_dir=models_dir, verbose=verbose)
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

def generer_fichier_soumission(results_df, output_path="predictions_finales.csv", format_simple=False):
    """
    Génère le fichier de soumission final.
    
    Args:
        results_df: DataFrame avec les prédictions
        output_path: Chemin du fichier de sortie
        format_simple: Si True, génère seulement les prédictions (format soumission)
                      Si False, génère le fichier détaillé complet
    
    Returns:
        Chemin du fichier généré
    """
    try:
        if format_simple:
            # Format de soumission : seulement les prédictions sans en-tête
            submission_df = results_df['prediction_label']
            submission_df.to_csv(output_path, index=False, header=False)
            logger.info(f"📄 Fichier de soumission généré : {output_path}")
        else:
            # Format détaillé : toutes les colonnes
            results_df.to_csv(output_path, index=False, header=False)
            logger.info(f"📊 Fichier détaillé généré : {output_path}")
        
        return output_path
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de la génération du fichier : {e}")
        return None

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

def generer_predictions(test_data, model, threshold, imputation_method="knn"):
    """
    Génère les prédictions avec le modèle chargé.
    
    Args:
        test_data: DataFrame des données de test
        model: Modèle déjà chargé (objet sklearn, xgboost, dict stacking, etc.)
        threshold: Seuil optimal
        imputation_method: Méthode d'imputation pour charger les colonnes d'entraînement
    
    Returns:
        DataFrame avec les prédictions
    """
    try:
        import numpy as np
        
        # Charger les colonnes d'entraînement pour s'assurer de l'ordre exact
        training_columns = load_training_columns(imputation_method)
        
        if training_columns is not None:
            # S'assurer que les features sont dans l'ordre exact d'entraînement
            missing_cols = [col for col in training_columns if col not in test_data.columns]
            if missing_cols:
                logger.error(f"❌ Colonnes manquantes pour la prédiction : {missing_cols}")
                return None
            
            # Réorganiser les colonnes dans l'ordre exact d'entraînement
            test_data_ordered = test_data[training_columns].copy()
            logger.info(f"✅ Features réorganisées dans l'ordre d'entraînement : {len(test_data_ordered.columns)} colonnes")
        else:
            # Fallback : utiliser les données telles qu'elles sont
            test_data_ordered = test_data
            logger.warning("⚠️ Impossible de charger les colonnes d'entraînement, utilisation des données telles qu'elles sont")
        
        # Vérifier si c'est un modèle de stacking (dict)
        if isinstance(model, dict) and 'pipelines' in model:
            # Modèle de stacking (no refit) - pipelines est un dict
            pipelines = model['pipelines']
            
            logger.info(f"✅ Utilisation de {len(test_data_ordered.columns)} features pour stacking")
            
            # Générer les prédictions avec chaque pipeline
            probas = [pipe.predict_proba(test_data_ordered)[:, 1] for pipe in pipelines.values()]
            # Moyenne des prédictions
            probabilities = np.mean(probas, axis=0)
        else:
            # Modèle standard (sklearn, xgboost, etc.)
            logger.info(f"✅ Utilisation de {len(test_data_ordered.columns)} features pour modèle standard")
            
            probabilities = model.predict_proba(test_data_ordered)[:, 1]
        
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

def load_training_columns(imputation_method: str = "knn"):
    """
    Charge les colonnes utilisées lors de l'entraînement du modèle.
    
    Args:
        imputation_method: Méthode d'imputation ("knn" ou "mice")
    
    Returns:
        Liste des colonnes dans l'ordre d'entraînement
    """
    from modules.config.paths_config import setup_project_paths
    import joblib
    
    paths = setup_project_paths()
    columns_path = paths["MODELS_DIR"] / "notebook2" / f"columns_{imputation_method}.pkl"
    
    try:
        columns = joblib.load(columns_path)
        logger.info(f"✅ Colonnes d'entraînement chargées : {len(columns)} features")
        return columns
        
    except FileNotFoundError:
        logger.warning(f"⚠️ Fichier columns_{imputation_method}.pkl non trouvé")
        return None

def main_pipeline_prediction_with_params(
    imputation_method: str = "mice",
    use_stacking: bool = True
) -> str:
    """
    Pipeline principal pour la génération des prédictions finales.
    Version mise à jour utilisant run_prediction_pipeline avec toutes les transformations du notebook 3.
    
    Args:
        imputation_method: Méthode d'imputation ("knn" ou "mice")
        use_stacking: Si True, utilise le modèle de stacking au lieu de XGBoost
    
    Returns:
        Chemin du fichier de soumission généré
    """
    logger.info("🚀 Démarrage du pipeline de prédiction finale")
    logger.info(f"📊 Configuration : imputation={imputation_method}, stacking={use_stacking}")
    
    # Déterminer le type de modèle à utiliser
    if use_stacking:
        model_type = "stacking"  # Utilise stacking sans refit par défaut
        logger.info("🤖 Utilisation du modèle de stacking (sans refit)")
    else:
        model_type = "xgboost"
        logger.info("🤖 Utilisation du modèle XGBoost")
    
    # Utiliser la fonction run_prediction_pipeline mise à jour
    try:
        output_file = run_prediction_pipeline(
            model_type=model_type,
            imputation_method=imputation_method,
            threshold=None,  # Utilise le seuil optimal automatiquement
            model_path=None,  # Utilise le chemin automatique
            threshold_path=None,  # Utilise le chemin automatique
            output_path=None,  # Utilise le nom automatique
            data_path=None  # Utilise le chemin automatique
        )
        
        if output_file:
            logger.info(f"🎉 Pipeline terminé avec succès! Fichier: {output_file}")
            return output_file
        else:
            logger.error("❌ Échec du pipeline")
            return None
            
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'exécution du pipeline: {e}")
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

def appliquer_transformations_optimales(df, imputation_method="knn"):
    """
    Applique les transformations optimales sur les variables continues.
    Charge les transformers déjà sauvegardés dans le notebook 1.
    
    Args:
        df: DataFrame avec les données
        imputation_method: Méthode d'imputation ("knn" ou "mice")
    
    Returns:
        DataFrame avec les transformations appliquées
    """
    import logging
    logger = logging.getLogger(__name__)
    from modules.config.paths_config import setup_project_paths
    import joblib
    
    # Récupérer le chemin des modèles
    paths = setup_project_paths()
    models_dir = paths["MODELS_DIR"] / "notebook1" / imputation_method / f"{imputation_method}_transformers"
    
    # Charger les transformers déjà sauvegardés
    try:
        yj_transformer = joblib.load(models_dir / "yeo_johnson_X1_X2.pkl")
        bc_transformer = joblib.load(models_dir / "box_cox_X3.pkl")
        logger.info(f"✅ Transformers chargés depuis le notebook 1 ({imputation_method})")
    except FileNotFoundError:
        logger.error(f"❌ Transformers du notebook 1 non trouvés pour {imputation_method}. Vérifiez que le notebook 1 a été exécuté.")
        raise
    
    # Appliquer les transformations
    df_transformed = df.copy()
    
    # Yeo-Johnson pour X1, X2
    X1_X2_data = df[['X1', 'X2']].values
    X1_X2_transformed = yj_transformer.transform(X1_X2_data)
    df_transformed['X1_transformed'] = X1_X2_transformed[:, 0]
    df_transformed['X2_transformed'] = X1_X2_transformed[:, 1]
    
    # Box-Cox pour X3
    X3_data = df[['X3']].values
    X3_transformed = bc_transformer.transform(X3_data)
    df_transformed['X3_transformed'] = X3_transformed.ravel()
    
    logger.info("✅ Transformation optimale appliquée sur X1, X2, X3 (Yeo-Johnson/Box-Cox)")
    return df_transformed

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
    Pipeline générique pour la génération de prédictions.
    Utilise directement prediction_submission.py qui fonctionne parfaitement.
    
    Args:
        model_type: Type de modèle ("xgboost", "gradboost", "svm", "randforest", "mlp", "logreg", "stacking", "stacking_refit")
        imputation_method: Méthode d'imputation ("knn" ou "mice")
        threshold: Seuil personnalisé (si None, utilise le seuil optimal)
        model_path: Chemin vers le modèle (si None, utilise le chemin automatique)
        threshold_path: Chemin vers le fichier de seuil (si None, utilise le chemin automatique)
        output_path: Chemin de sortie (si None, utilise le nom automatique)
        data_path: Chemin vers les données de test (si None, utilise le chemin automatique)
    
    Returns:
        Dictionnaire avec les chemins des fichiers générés
    """
    logger.info(f"🚀 Pipeline générique : modèle={model_type}, imputation={imputation_method}, seuil={threshold}, model_path={model_path}, threshold_path={threshold_path}, data_path={data_path}")
    
    # Configuration des chemins
    from modules.config.paths_config import setup_project_paths
    paths = setup_project_paths()
    MODELS_DIR = paths["MODELS_DIR"]
    
    # Pour les modèles de stacking, utiliser prediction_submission_with_stacking.py
    if model_type in ["stacking", "stacking_refit"]:
        logger.info("🎯 Modèle stacking détecté - utilisation de prediction_submission_with_stacking.py")
        
        # Import et utilisation de prediction_submission_with_stacking.py
        import subprocess
        import sys
        
        # Construire la commande
        cmd = [
            sys.executable, "prediction_submission_with_stacking.py",
            "--model_type", model_type,
            "--imputation", imputation_method
        ]
        
        if threshold is not None:
            cmd.extend(["--threshold", str(threshold)])
        if model_path is not None:
            cmd.extend(["--model_path", model_path])
        if threshold_path is not None:
            cmd.extend(["--threshold_path", threshold_path])
        if output_path is not None:
            cmd.extend(["--output_path", output_path])
        if data_path is not None:
            cmd.extend(["--data_path", data_path])
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info("✅ prediction_submission_with_stacking.py exécuté avec succès")
            logger.info(f"📄 Sortie : {result.stdout}")
            
            # Retourner les chemins des fichiers générés
            return {
                "detailed": f"outputs/predictions/predictions_detailed_{model_type}_{imputation_method}.csv",
                "submission": f"outputs/predictions/submission_{model_type}_{imputation_method}.csv"
            }
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Erreur lors de l'exécution de prediction_submission_with_stacking.py : {e}")
            logger.error(f"📄 Erreur : {e.stderr}")
            
            # Fallback : essayer d'utiliser directement les pipelines optimisés
            logger.info("🔄 Tentative de fallback avec les pipelines optimisés...")
            return run_stacking_prediction_fallback(model_type, imputation_method, threshold, output_path, data_path)
    
    # Pour les modèles standard, utiliser prediction_submission.py
    else:
        logger.info("🎯 Modèle standard détecté - utilisation de prediction_submission.py")
        
        # Import et utilisation de prediction_submission.py
        from prediction_submission import generate_submission_file
        
        # Déterminer les chemins
        if data_path is None:
            TEST_DATA_PATH = paths["RAW_DATA_DIR"] / "data_test.csv"
        else:
            TEST_DATA_PATH = Path(data_path)
            
        COLUMNS_PATH = paths["MODELS_DIR"] / "notebook2" / f"{imputation_method}" / f"columns_{imputation_method}.pkl"
        
        if output_path is None:
            SUBMISSION_PATH = paths["OUTPUTS_DIR"] / "predictions" / f"submission_{model_type}_{imputation_method}.csv"
        else:
            SUBMISSION_PATH = Path(output_path)
        
        try:
            # Utiliser generate_submission_file qui fonctionne parfaitement
            generate_submission_file(TEST_DATA_PATH, COLUMNS_PATH, paths["MODELS_DIR"], SUBMISSION_PATH)
            
            logger.info("✅ prediction_submission.py exécuté avec succès")
            
            # Retourner les chemins des fichiers générés
            return {
                "detailed": f"outputs/predictions/predictions_detailed_{model_type}_{imputation_method}.csv",
                "submission": str(SUBMISSION_PATH)
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'exécution de prediction_submission.py : {e}")
            return None

def main_pipeline_prediction_with_refit(
    imputation_method: str = "mice"
) -> str:
    """
    Pipeline principal pour la génération des prédictions finales avec stacking refit.
    Version mise à jour utilisant run_prediction_pipeline avec toutes les transformations du notebook 3.
    
    Args:
        imputation_method: Méthode d'imputation ("knn" ou "mice")
    
    Returns:
        Chemin du fichier de soumission généré
    """
    logger.info("🚀 Démarrage du pipeline de prédiction finale avec stacking refit")
    logger.info(f"📊 Configuration : imputation={imputation_method}, stacking=refit")
    
    # Utiliser la fonction run_prediction_pipeline mise à jour avec stacking refit
    try:
        output_file = run_prediction_pipeline(
            model_type="stacking_refit",
            imputation_method=imputation_method,
            threshold=None,  # Utilise le seuil optimal automatiquement
            model_path=None,  # Utilise le chemin automatique
            threshold_path=None,  # Utilise le chemin automatique
            output_path=None,  # Utilise le nom automatique
            data_path=None  # Utilise le chemin automatique
        )
        
        if output_file:
            logger.info(f"🎉 Pipeline terminé avec succès! Fichier: {output_file}")
            return output_file
        else:
            logger.error("❌ Échec du pipeline")
            return None
            
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'exécution du pipeline: {e}")
        return None

def run_stacking_prediction_fallback(model_type, imputation_method, threshold=None, output_path=None, data_path=None):
    """
    Fonction de fallback pour les modèles de stacking qui utilise prediction_stacking_direct.py.
    """
    logger.info(f"🔄 Fallback stacking : modèle={model_type}, imputation={imputation_method}")
    
    try:
        # Utiliser le script dédié pour les stacking
        import subprocess
        import sys
        
        # Construire la commande
        cmd = [
            sys.executable, "prediction_stacking_direct.py",
            "--model_type", model_type,
            "--imputation", imputation_method
        ]
        
        if threshold is not None:
            cmd.extend(["--threshold", str(threshold)])
        if output_path is not None:
            cmd.extend(["--output_path", output_path])
        if data_path is not None:
            cmd.extend(["--data_path", data_path])
        
        logger.info(f"🔄 Exécution de prediction_stacking_direct.py...")
        
        # Exécuter sans capture pour voir les logs en temps réel
        result = subprocess.run(cmd, check=True)
        
        logger.info("✅ prediction_stacking_direct.py exécuté avec succès")
        
        # Retourner les chemins des fichiers générés
        return {
            "detailed": f"outputs/predictions/predictions_detailed_{model_type}_{imputation_method}.csv",
            "submission": f"outputs/predictions/submission_{model_type}_{imputation_method}.csv"
        }
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Erreur lors de l'exécution de prediction_stacking_direct.py : {e}")
        return None
    except Exception as e:
        logger.error(f"❌ Erreur dans le fallback stacking : {e}")
        return None

if __name__ == "__main__":
    main_pipeline_prediction()

