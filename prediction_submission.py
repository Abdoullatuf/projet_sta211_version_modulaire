import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import logging

# Configuration du logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def load_test_data(data_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """
    Charge les données de test avec gestion robuste des erreurs de parsing.
    """
    try:
        # Essayer d'abord avec une lecture standard (avec en-tête)
        df = pd.read_csv(
            data_path,
            sep='\t',
            quotechar='"',
            encoding='utf-8',
            na_values='?',
            engine='python',
            on_bad_lines='skip'
        )
        
        log.info(f"📊 Fichier lu avec {df.shape[1]} colonnes et {df.shape[0]} lignes")
        
        # Si on n'a qu'une colonne, essayer avec virgule
        if df.shape[1] == 1:
            log.warning("⚠️ Lecture avec tabulation a échoué (1 seule colonne détectée). Retraitement avec virgule.")
            df = pd.read_csv(
                data_path,
                sep=',',
                quotechar='"',
                encoding='utf-8',
                na_values='?',
                engine='python',
                on_bad_lines='skip'
            )
            
    except Exception as e:
        log.error(f"❌ Erreur de lecture du fichier : {e}")
        raise

    # Le fichier contient les données avec en-tête, nous devons traiter les colonnes
    log.info(f"📊 Fichier lu avec {df.shape[1]} colonnes et {df.shape[0]} lignes")
    
    # La première colonne est l'ID, les autres sont les features
    ids = df.iloc[:, 0]
    features = df.iloc[:, 1:]  # Toutes les colonnes sauf la première (ID)
    
    # Nettoyer les noms de colonnes (supprimer les guillemets)
    features.columns = [col.replace('"', '') for col in features.columns]
    
    # Vérifier que nous avons les bonnes colonnes
    expected_cols = [f'X{i}' for i in range(1, 1558)]
    missing_cols = [col for col in expected_cols if col not in features.columns]
    if missing_cols:
        log.warning(f"⚠️ Colonnes manquantes : {missing_cols[:10]}...")
        # Créer les colonnes manquantes avec des valeurs par défaut
        for col in missing_cols:
            features[col] = 0
    
    # Réorganiser les colonnes dans l'ordre attendu
    features = features[expected_cols]
    
    # Nettoyer les données - supprimer les guillemets et convertir en numérique
    for col in features.columns:
        features[col] = features[col].astype(str).str.replace('"', '').str.replace("'", "")
        # Convertir en numérique, avec gestion des erreurs
        features[col] = pd.to_numeric(features[col], errors='coerce')
    
    # Remplacer les valeurs NaN par 0 (ou une autre stratégie appropriée)
    features = features.fillna(0)
    
    log.info(f"✅ Données originales chargées et nettoyées : {features.shape}")
    log.info(f"📋 Premières colonnes : {list(features.columns[:5])}")
    return features, ids

def preprocess(df: pd.DataFrame, imputation_method: str, model_dir: Path) -> pd.DataFrame:
    """
    Applique le pipeline de prétraitement complet (Notebook 01) à un DataFrame brut
    pour une méthode d'imputation donnée (mice ou knn).
    """
    log.info(f"--- Démarrage du prétraitement pour '{imputation_method.upper()}' ---")
    df = df.copy()
    
    # Étape 1: Imputation X4 (médiane)
    if 'X4' in df.columns:
        median_path = model_dir / "notebook1" / "median_imputer_X4.pkl"
        median_value = joblib.load(median_path)
        df['X4'] = df['X4'].fillna(median_value)

    # Étape 2: Imputation MICE/KNN (seulement sur les colonnes continues)
    continuous_cols = ['X1', 'X2', 'X3']
    df_continuous = df[continuous_cols].copy()
    
    if imputation_method == "mice":
        imputer_path = model_dir / "notebook1" / imputation_method / "imputer_mice_custom.pkl"
    else:  # knn
        imputer_path = model_dir / "notebook1" / imputation_method / "imputer_knn_k7.pkl"
    
    imputer = joblib.load(imputer_path)
    df_imputed_continuous = pd.DataFrame(
        imputer.transform(df_continuous), 
        columns=continuous_cols, 
        index=df.index
    )
    
    # Remplacer les colonnes originales par les colonnes imputées
    for col in continuous_cols:
        df[col] = df_imputed_continuous[col]
    
    # Étape 3: Transformation (Yeo-Johnson + Box-Cox)
    yj_path = model_dir / "notebook1" / imputation_method / f"{imputation_method}_transformers" / "yeo_johnson_X1_X2.pkl"
    bc_path = model_dir / "notebook1" / imputation_method / f"{imputation_method}_transformers" / "box_cox_X3.pkl"
    transformer_yj = joblib.load(yj_path)
    transformer_bc = joblib.load(bc_path)
    df[['X1', 'X2']] = transformer_yj.transform(df[['X1', 'X2']])
    df_x3 = df[['X3']].copy()
    if (df_x3['X3'] <= 0).any(): df_x3['X3'] += 1e-6
    df['X3'] = transformer_bc.transform(df_x3)
    df.rename(columns={'X1': 'X1_transformed', 'X2': 'X2_transformed', 'X3': 'X3_transformed'}, inplace=True)
    
    # Étape 4: Capping
    capping_path = model_dir / "notebook1" / imputation_method / f"capping_params_{imputation_method}.pkl"
    capping_params = joblib.load(capping_path)
    for col in ['X1_transformed', 'X2_transformed', 'X3_transformed']:
        bounds = capping_params.get(col, {})
        df[col] = np.clip(df[col], bounds.get('lower_bound'), bounds.get('upper_bound'))

    # Étape 5: Suppression de la colinéarité
    corr_path = model_dir / "notebook1" / imputation_method / f"cols_to_drop_corr_{imputation_method}.pkl"
    cols_to_drop = joblib.load(corr_path)
    df = df.drop(columns=cols_to_drop, errors='ignore')
    
    # Étape 6: Ingénierie de caractéristiques
    poly_path = model_dir / "notebook1" / imputation_method / f"poly_transformer_{imputation_method}.pkl"
    poly_transformer = joblib.load(poly_path)
    continuous_cols = ['X1_transformed', 'X2_transformed', 'X3_transformed']
    continuous_features = df[continuous_cols]
    poly_features = poly_transformer.transform(continuous_features)
    poly_feature_names = poly_transformer.get_feature_names_out(continuous_cols)
    df_poly = pd.DataFrame(poly_features, columns=poly_feature_names, index=df.index)
    df = df.drop(columns=continuous_cols).join(df_poly)
    
    log.info(f"--- Prétraitement pour '{imputation_method.upper()}' terminé. Shape final : {df.shape} ---")
    return df

def predict(models_dir: Path, test_data_path: Path, output_dir: Path):
    """
    Génère les prédictions finales en utilisant le modèle champion (Stacking sans refit KNN).
    """
    log.info("🚀 Démarrage de la génération des prédictions finales...")
    
    # Charger les données de test
    log.info("📂 Chargement des données de test...")
    df_test, ids = load_test_data(test_data_path)
    
    # Prétraiter les données avec KNN (méthode du champion)
    log.info("🔄 Prétraitement des données avec KNN...")
    df_processed = preprocess(df_test, "knn", models_dir)
    
    # Charger les colonnes attendues pour KNN
    log.info("📋 Chargement des colonnes attendues...")
    columns_knn = joblib.load(models_dir / "notebook2" / "knn" / "columns_knn.pkl")
    
    # Sélectionner seulement les colonnes utilisées dans l'entraînement
    available_columns = [col for col in columns_knn if col in df_processed.columns]
    missing_columns = [col for col in columns_knn if col not in df_processed.columns]
    
    if missing_columns:
        log.warning(f"⚠️ Colonnes manquantes : {missing_columns}")
        # Ajouter des colonnes avec des valeurs par défaut
        for col in missing_columns:
            df_processed[col] = 0
    
    df_processed = df_processed[columns_knn]
    
    # Charger le seuil optimal
    log.info("📊 Chargement du seuil optimal...")
    threshold_path = models_dir / "notebook3" / "stacking_champion_threshold.json"
    with open(threshold_path, 'r') as f:
        threshold_data = json.load(f)
        threshold = threshold_data["threshold"]
    
    log.info(f"✅ Seuil optimal chargé : {threshold:.4f}")
    
    # Générer les meta-features en utilisant les pipelines individuels
    log.info("🔄 Génération des meta-features...")
    meta_features = {}
    
    model_names = ["SVM", "XGBoost", "RandForest", "GradBoost", "MLP"]
    
    for model_name in model_names:
        try:
            pipeline_path = models_dir / "notebook2" / f"pipeline_{model_name.lower()}_knn.joblib"
            pipeline = joblib.load(pipeline_path)
            
            # Générer les probabilités pour chaque modèle
            proba = pipeline.predict_proba(df_processed)[:, 1]
            meta_features[f"{model_name}_knn"] = proba
            log.info(f"✅ Meta-features générées pour {model_name}")
        except Exception as e:
            log.error(f"❌ Erreur lors de la génération des meta-features pour {model_name}: {e}")
            # Utiliser des probabilités par défaut en cas d'erreur
            meta_features[f"{model_name}_knn"] = np.full(len(df_processed), 0.5)
    
    # Créer le DataFrame des meta-features
    df_meta = pd.DataFrame(meta_features)
    
    # Calculer la moyenne des probabilités (Stacking sans refit)
    log.info("📊 Calcul de la moyenne des probabilités (Stacking sans refit)...")
    proba_final = df_meta.mean(axis=1)
    
    # Appliquer le seuil optimal
    prediction_num = (proba_final >= threshold).astype(int)
    prediction_label = np.where(prediction_num == 1, "ad.", "noad.")
    
    # Debug: vérifier les longueurs
    log.info(f"🔍 Debug - Longueurs des variables:")
    log.info(f"   - ids: {len(ids)} (type: {type(ids)})")
    log.info(f"   - proba_final: {len(proba_final)}")
    log.info(f"   - prediction_label: {len(prediction_label)}")
    
    # Convertir ids en liste si c'est une Series
    if hasattr(ids, 'values'):
        ids_list = ids.values.tolist()
    else:
        ids_list = list(ids)
    
    # Créer le DataFrame détaillé avec toutes les informations
    detailed = pd.DataFrame({
        'ID': ids_list,
        'probabilite_stacking': proba_final.values,
        'prediction_stacking': prediction_label,
        'seuil_optimal': [threshold] * len(ids_list)
    })
    
    # Créer le fichier de soumission (seulement les prédictions)
    submission = pd.DataFrame({
        'prediction': prediction_label
    })
    
    # Sauvegarder les résultats
    output_dir.mkdir(parents=True, exist_ok=True)
    detailed.to_csv(output_dir / "predictions_finales_stacking_knn_detailed.csv", index=False)
    submission.to_csv(output_dir / "predictions_finales_stacking_knn_submission.csv", index=False)
    
    # Afficher les statistiques
    log.info("📊 Statistiques des prédictions :")
    log.info(f"   - Nombre total de prédictions : {len(prediction_label)}")
    log.info(f"   - Prédictions 'ad.' : {np.sum(prediction_num)} ({np.mean(prediction_num)*100:.1f}%)")
    log.info(f"   - Prédictions 'noad.' : {len(prediction_num) - np.sum(prediction_num)} ({(1-np.mean(prediction_num))*100:.1f}%)")
    log.info(f"   - Seuil utilisé : {threshold:.4f}")
    
    log.info("✅ Prédictions générées et fichiers exportés avec succès !")
    log.info(f"   📄 Fichier détaillé : {output_dir / 'predictions_finales_stacking_knn_detailed.csv'}")
    log.info(f"   📄 Fichier de soumission : {output_dir / 'predictions_finales_stacking_knn_submission.csv'}")

if __name__ == "__main__":
    predict(
        models_dir=Path("models"),
        test_data_path=Path("data/raw/data_test.csv"),
        output_dir=Path("outputs/predictions")
    )
