# =============================================================================
# MODULE : CRÉATEUR DE MODÈLES DE STACKING AVEC PARAMÈTRES OPTIMISÉS
# =============================================================================

from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import json
from pathlib import Path
import sys
import os

# Ajouter le chemin du projet pour les imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from modules.config.paths_config import setup_project_paths

def strip_clf_prefix(params):
    """Supprime le préfixe 'clf__' des paramètres"""
    return {k.replace("clf__", ""): v for k, v in params.items()}

def clean_xgb_params(params):
    """Nettoie les paramètres XGBoost en supprimant les paramètres non supportés"""
    params_clean = params.copy()
    
    # Paramètres à supprimer (s'ils existent)
    params_to_remove = ['use_label_encoder', 'eval_metric', 'feature_weights']
    
    for param in params_to_remove:
        if param in params_clean:
            params_clean.pop(param)
    
    return params_clean

def clean_svm_params(params):
    """Nettoie les paramètres SVM et s'assure que probability=True"""
    params_clean = params.copy()
    
    # S'assurer que probability=True est présent
    params_clean['probability'] = True
    
    return params_clean

def load_best_params_from_saved_files(model_name, imputation="knn", models_dir=None):
    """Charge les meilleurs paramètres depuis les fichiers JSON sauvegardés"""
    try:
        # Utiliser le models_dir fourni ou le chemin par défaut
        if models_dir is None:
            # Obtenir les chemins du projet
            paths = setup_project_paths()
            models_dir = paths["MODELS_DIR"] / "notebook2"
        
        # Construire les noms de fichiers possibles (full puis reduced)
        if model_name.lower() in ['gradboost', 'gradientboosting']:
            base_name = f"best_params_gradboost_{imputation}"
        elif model_name.lower() in ['xgboost']:
            base_name = f"best_params_xgboost_{imputation}"
        elif model_name.lower() in ['randforest', 'randomforest']:
            base_name = f"best_params_randforest_{imputation}"
        elif model_name.lower() in ['svm']:
            base_name = f"best_params_svm_{imputation}"
        elif model_name.lower() in ['mlp']:
            base_name = f"best_params_mlp_{imputation}"
        else:
            raise ValueError(f"Modèle {model_name} non reconnu")
        
        # Essayer d'abord _full, puis _reduced, puis sans suffixe
        possible_files = [
            f"{base_name}_full.json",
            f"{base_name}_reduced.json",
            f"{base_name}.json"
        ]
        
        file_path = None
        for file_name in possible_files:
            test_path = Path(models_dir) / file_name
            if test_path.exists():
                file_path = test_path
                break
        
        if file_path is None:
            print(f"⚠️ Aucun fichier trouvé pour {model_name}_{imputation} parmi: {possible_files}")
            return None
            
        # Charger les paramètres
        with open(file_path, 'r') as f:
            params = json.load(f)
        
        # Supprimer le préfixe 'clf__' de tous les paramètres
        params = strip_clf_prefix(params)
        
        # Nettoyer les paramètres selon le type de modèle
        if model_name.lower() in ['gradboost', 'xgboost']:
            params = clean_xgb_params(params)
        elif model_name.lower() in ['svm']:
            params = clean_svm_params(params)
        
        print(f"✅ {model_name}_{imputation}: {len(params)} paramètres chargés depuis {file_name}")
        return params
        
    except Exception as e:
        print(f"❌ Erreur lors du chargement des paramètres pour {model_name}_{imputation}: {e}")
        return None

def create_stacking_models(imputation_method="both", models_dir=None, verbose=True):
    """
    Crée les modèles de stacking avec les paramètres optimisés sauvegardés.
    
    Args:
        imputation_method (str): "knn", "mice", ou "both" (défaut)
        models_dir (str/Path): Chemin vers le dossier contenant les fichiers JSON
        verbose (bool): Afficher les messages de progression
    
    Returns:
        dict: Dictionnaire contenant les modèles créés
    """
    
    if verbose:
        print("=" * 80)
        print(f"🎯 CRÉATION DES MODÈLES DE STACKING - {imputation_method.upper()}")
        print("=" * 80)
    
    result = {}
    
    # Définir les méthodes d'imputation à traiter
    if imputation_method.lower() == "knn":
        imputation_methods = ["knn"]
    elif imputation_method.lower() == "mice":
        imputation_methods = ["mice"]
    else:  # "both"
        imputation_methods = ["knn", "mice"]
    
    for imputation in imputation_methods:
        if verbose:
            print(f"\n🔄 Traitement {imputation.upper()}...")
            print("-" * 50)
        
        # Chargement des paramètres optimisés
        if verbose:
            print(f"📊 Chargement des paramètres optimisés {imputation}...")
        
        best_gradboost_params = load_best_params_from_saved_files("gradboost", imputation, models_dir)
        best_mlp_params = load_best_params_from_saved_files("mlp", imputation, models_dir)
        best_randforest_params = load_best_params_from_saved_files("randforest", imputation, models_dir)
        best_svm_params = load_best_params_from_saved_files("svm", imputation, models_dir)
        best_xgboost_params = load_best_params_from_saved_files("xgboost", imputation, models_dir)
        
        # Vérification que tous les paramètres ont été chargés
        loaded_params = {
            f"gradboost_{imputation}": best_gradboost_params,
            f"mlp_{imputation}": best_mlp_params,
            f"randforest_{imputation}": best_randforest_params,
            f"svm_{imputation}": best_svm_params,
            f"xgboost_{imputation}": best_xgboost_params
        }
        
        missing_params = [name for name, params in loaded_params.items() if params is None]
        
        if missing_params:
            raise Exception(f"Impossible de charger tous les paramètres optimisés {imputation}: {missing_params}")
        
        if verbose:
            print(f"✅ Chargement de TOUS les paramètres optimisés {imputation} !")
        
        # Création des modèles avec les meilleurs paramètres
        if verbose:
            print(f"\n🔄 Création des modèles {imputation} avec les MEILLEURS paramètres...")
        
        try:
            gradboost_model = XGBClassifier(**best_gradboost_params)
            mlp_model = MLPClassifier(**best_mlp_params)
            randforest_model = RandomForestClassifier(**best_randforest_params)
            svm_model = SVC(**best_svm_params)
            xgboost_model = XGBClassifier(**best_xgboost_params)
            
            if verbose:
                print(f"✅ Tous les modèles {imputation} créés avec succès !")
            
            # Affichage des paramètres clés
            if verbose:
                print(f"\n📋 Paramètres clés {imputation.upper()} :")
                print(f"   GradBoost - n_estimators: {best_gradboost_params.get('n_estimators', 'N/A')}")
                print(f"   MLP - hidden_layer_sizes: {best_mlp_params.get('hidden_layer_sizes', 'N/A')}")
                print(f"   RandomForest - n_estimators: {best_randforest_params.get('n_estimators', 'N/A')}")
                print(f"   SVM - C: {best_svm_params.get('C', 'N/A')}, kernel: {best_svm_params.get('kernel', 'N/A')}")
                print(f"   XGBoost - n_estimators: {best_xgboost_params.get('n_estimators', 'N/A')}")
            
        except Exception as e:
            print(f"❌ Erreur lors de la création des modèles {imputation} : {e}")
            raise
        
        # Création du stacking classifier
        if verbose:
            print(f"\n🔄 Création du Stacking Classifier {imputation}...")
        
        try:
            estimators = [
                ('gradboost', gradboost_model),
                ('mlp', mlp_model),
                ('randforest', randforest_model),
                ('svm', svm_model),
                ('xgboost', xgboost_model)
            ]
            
            stacking_classifier = StackingClassifier(
                estimators=estimators,
                final_estimator=LogisticRegression(random_state=42),
                cv=5,
                stack_method='predict_proba',
                n_jobs=-1
            )
            
            if verbose:
                print(f"✅ Stacking classifier {imputation} créé avec succès !")
                print(f"📊 Nombre d'estimateurs de base {imputation} : {len(estimators)}")
            
        except Exception as e:
            print(f"❌ Erreur lors de la création du stacking {imputation} : {e}")
            raise
        
        # Stockage des résultats
        result[f"stacking_classifier_{imputation}"] = stacking_classifier
        result[f"gradboost_{imputation}"] = gradboost_model
        result[f"mlp_{imputation}"] = mlp_model
        result[f"randforest_{imputation}"] = randforest_model
        result[f"svm_{imputation}"] = svm_model
        result[f"xgboost_{imputation}"] = xgboost_model
        result[f"best_gradboost_params_{imputation}"] = best_gradboost_params
        result[f"best_mlp_params_{imputation}"] = best_mlp_params
        result[f"best_randforest_params_{imputation}"] = best_randforest_params
        result[f"best_svm_params_{imputation}"] = best_svm_params
        result[f"best_xgboost_params_{imputation}"] = best_xgboost_params
        
        if verbose:
            print(f"\n" + "=" * 80)
            print(f"🚀 STACKING {imputation.upper()} PRÊT !")
            print("=" * 80)
    
    # Résumé final
    if verbose:
        print("\n" + "=" * 80)
        print("🎯 RÉSUMÉ FINAL - MODÈLES DE STACKING CRÉÉS")
        print("=" * 80)
        
        for imputation in imputation_methods:
            print(f"   ✅ Stacking {imputation.upper()} : stacking_classifier_{imputation}")
        
        print("\n🎯 PROCHAINES ÉTAPES :")
        for imputation in imputation_methods:
            print(f"   1. Entraîner stacking_classifier_{imputation} sur (X_train_{imputation}, y_train_{imputation})")
        print("   2. Optimiser les seuils de décision pour chaque stacking")
        print("   3. Évaluer les performances sur les données de validation")
        print("   4. Comparer les résultats KNN vs MICE")
        
        print("\n✅ Module prêt pour l'utilisation !")
    
    return result

def get_stacking_models_info(models_dict):
    """
    Affiche les informations sur les modèles de stacking créés.
    
    Args:
        models_dict (dict): Dictionnaire retourné par create_stacking_models
    """
    print("\n📋 INFORMATIONS SUR LES MODÈLES CRÉÉS :")
    print("=" * 50)
    
    for key, value in models_dict.items():
        if 'stacking_classifier' in key:
            print(f"   🔹 {key}: StackingClassifier avec {len(value.estimators)} estimateurs")
        elif 'params' in key:
            print(f"   🔹 {key}: {len(value)} paramètres optimisés")
        else:
            print(f"   🔹 {key}: {type(value).__name__}")
    
    print("\n🎯 Variables disponibles :")
    stacking_models = [k for k in models_dict.keys() if 'stacking_classifier' in k]
    base_models = [k for k in models_dict.keys() if 'stacking_classifier' not in k and 'params' not in k]
    params = [k for k in models_dict.keys() if 'params' in k]
    
    print(f"   📊 Modèles de stacking ({len(stacking_models)}): {stacking_models}")
    print(f"   📊 Modèles de base ({len(base_models)}): {base_models}")
    print(f"   📊 Paramètres optimisés ({len(params)}): {params}")

# =============================================================================
# EXEMPLE D'UTILISATION
# =============================================================================

if __name__ == "__main__":
    # Exemple d'utilisation
    print("🔧 Test du module de création de modèles de stacking...")
    
    # Créer les modèles pour KNN et MICE
    models = create_stacking_models(imputation_method="both", verbose=True)
    
    # Afficher les informations
    get_stacking_models_info(models)
    
    print("\n✅ Test terminé avec succès !") 