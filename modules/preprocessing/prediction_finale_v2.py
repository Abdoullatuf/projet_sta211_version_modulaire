#!/usr/bin/env python3
# modules/preprocessing/prediction_finale_v2.py

"""
✅ MODULE DE PRÉDICTION FINALE - Version 2 Corrigée
Ce module applique le pipeline de prétraitement complet pour générer des prédictions
sur de nouvelles données, en permettant de choisir entre les stratégies MICE et KNN.
"""

import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
from typing import List, Union

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- FONCTIONS DE PRÉTRAITEMENT INDIVIDUELLES (INCHANGÉES) ---
# (Les fonctions charger_donnees_test, imputer_valeurs_manquantes, etc. sont supposées être ici)
# ... (copiez-collez les fonctions de votre script original ici) ...


# --- PIPELINE DE PRÉDICTION COMPLET (VERSION CORRIGÉE) ---

def generer_predictions_finales(
    chemin_donnees_test: str,
    chemin_modeles: str,
    chemin_sortie_predictions: str,
    methode_imputation: str = 'mice'  # Nouveau paramètre : 'mice' ou 'knn'
):
    """
    Exécute le pipeline complet de prétraitement et de prédiction.
    """
    logger.info(f"🚀 LANCEMENT DU PIPELINE AVEC LA MÉTHODE '{methode_imputation.upper()}' 🚀")

    models_path = Path(chemin_modeles)
    
    # --- Chargement des données brutes ---
    df_test = charger_donnees_test(Path(chemin_donnees_test))

    # --- 1. Imputation ---
    imputer_path = models_path / f"imputer_{methode_imputation}.pkl"
    df_imputed = imputer_valeurs_manquantes(df_test, imputer_path)

    # --- 2. Transformation des variables continues ---
    # NOTE: Votre classe `TransformationOptimaleMixte` sauvegarde deux fichiers.
    # On doit charger les deux et les appliquer.
    df_transformed = df_imputed.copy()
    try:
        transformer_yj = joblib.load(models_path / f"yeo_johnson_X1_X2_{methode_imputation}.pkl")
        transformer_bc = joblib.load(models_path / f"box_cox_X3_{methode_imputation}.pkl")

        df_transformed[['X1', 'X2']] = transformer_yj.transform(df_transformed[['X1', 'X2']])
        # Pour Box-Cox, il faut gérer les valeurs non-positives
        df_x3 = df_transformed[['X3']]
        if (df_x3['X3'] <= 0).any():
            offset = 1e-6 # Le même offset que dans l'entraînement
            df_x3['X3'] = df_x3['X3'] + offset
        df_transformed['X3'] = transformer_bc.transform(df_x3)
        logger.info(f"✅ Transformation mixte ({methode_imputation}) appliquée.")
        
        # Renommer les colonnes pour correspondre au notebook
        df_transformed.rename(columns={'X1': 'X1_transformed', 'X2': 'X2_transformed', 'X3': 'X3_transformed'}, inplace=True)
        
    except FileNotFoundError:
        logger.error("❌ Fichiers de transformation non trouvés. Arrêt.")
        raise
        
    # --- 3. Traitement des outliers par Capping ---
    try:
        capping_params_path = models_path / f"capping_params_{methode_imputation}.pkl"
        capping_params = joblib.load(capping_params_path)
        df_capped = traiter_outliers_par_capping(df_transformed, ['X1_transformed', 'X2_transformed', 'X3_transformed'], capping_params)
    except FileNotFoundError:
        logger.warning("⚠️ Fichier capping_params.pkl non trouvé. Outliers non traités.")
        df_capped = df_transformed.copy()

    # --- 4. Suppression des variables corrélées ---
    try:
        cols_to_drop_path = models_path / f"cols_to_drop_corr_{methode_imputation}.pkl"
        cols_to_drop_corr = joblib.load(cols_to_drop_path)
        df_filtered = df_capped.drop(columns=cols_to_drop_corr, errors='ignore')
        logger.info(f"✅ Suppression de {len(cols_to_drop_corr)} variables corrélées.")
    except FileNotFoundError:
        logger.warning("⚠️ Fichier des colonnes corrélées non trouvé.")
        df_filtered = df_capped.copy()

    # --- 5. Ingénierie de caractéristiques ---
    poly_path = models_path / f"poly_transformer_{methode_imputation}.pkl"
    df_engineered = creer_features_polynomiales(
        df_filtered, 
        poly_path, 
        ['X1_transformed', 'X2_transformed', 'X3_transformed']
    )

    # --- 6. Sélection des caractéristiques finales ---
    features_path = models_path / f"selected_features_{methode_imputation}.pkl"
    df_selected = appliquer_selection_caracteristiques(df_engineered, features_path)

    # --- 7. Standardisation (Doit être fait dans le notebook de modélisation) ---
    # Cette étape est généralement la première du pipeline de MODÉLISATION.
    # Le scaler est appris sur X_train, puis appliqué sur X_test.
    # Il est donc préférable de le gérer dans le notebook 02.
    # Ici, nous supposons qu'un scaler final a été sauvegardé.
    try:
        scaler_path = models_path / f"scaler_final_{methode_imputation}.pkl"
        df_scaled = standardiser_donnees(df_selected, scaler_path)
    except FileNotFoundError:
         logger.warning("⚠️ Fichier scaler non trouvé. Les données ne seront pas standardisées.")
         df_scaled = df_selected.copy()


    # --- 8. Prédiction ---
    try:
        model_path = models_path / "best_model_pipeline.pkl" # Suppose que le meilleur modèle est unique
        seuil_path = models_path / "best_threshold.pkl"
        
        model = joblib.load(model_path)
        seuil_optimal = joblib.load(seuil_path)
        
        # S'assurer que les colonnes sont dans le bon ordre pour le modèle
        model_features = model.feature_names_in_ # Si le modèle est un pipeline scikit-learn
        df_final = df_scaled[model_features]
        
        probas = model.predict_proba(df_final)[:, 1]
        predictions = (probas >= seuil_optimal).astype(int)
        
        logger.info(f"✅ Prédictions générées avec le seuil optimal de {seuil_optimal:.4f}")
    except Exception as e:
        logger.error(f"❌ Erreur lors de la prédiction : {e}")
        raise

    # --- 9. Sauvegarde des résultats ---
    df_predictions = pd.DataFrame({'prediction': predictions}, index=df_test.index)
    output_path = Path(chemin_sortie_predictions)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_predictions.to_csv(output_path, index_label='ID')
    logger.info(f"🎉 Prédictions finales sauvegardées dans : {output_path}")

if __name__ == '__main__':
    # --- À CONFIGURER ---
    METHODE_CHOISIE = 'mice' # Changez pour 'knn' si nécessaire
    
    RAW_DATA_PATH = f"data/raw/data_test.csv"
    MODELS_PATH = "models/notebook1"
    OUTPUT_PATH = f"outputs/predictions/predictions_finales_{METHODE_CHOISIE}.csv"
    
    # Lancer le pipeline complet
    generer_predictions_finales(
        chemin_donnees_test=RAW_DATA_PATH,
        chemin_modeles=MODELS_PATH,
        chemin_sortie_predictions=OUTPUT_PATH,
        methode_imputation=METHODE_CHOISIE
    )