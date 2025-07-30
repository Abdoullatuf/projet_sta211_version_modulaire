# modules/feature_selection/analyze_model_features_debug.py
"""
Module pour l'analyse approfondie de l'importance des variables (version debug).
Intègre RFECV, Permutation Importance et SHAP avec diagnostic détaillé.
"""

import numpy as np
import pandas as pd
import logging
from sklearn.feature_selection import RFECV
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)

def analyze_feature_importance_debug(model, X_train, y_train, X_eval, y_eval, feature_names,
                               method='all', cv_folds=5, n_repeats_perm=20,
                               output_dir=None, model_name="model", save_results=True,
                               scoring='f1'):
    """
    Analyse l'importance des variables pour un modèle entraîné avec diagnostic détaillé.
    
    Args:
        model: Modèle scikit-learn déjà entraîné.
        X_train, y_train: Données d'entraînement (pour RFECV).
        X_eval, y_eval: Données d'évaluation (pour Permutation Importance et SHAP).
        feature_names (list): Liste des noms de variables.
        method (str ou list): Méthodes à appliquer ('rfecv', 'permutation', 'shap' ou 'all').
        cv_folds (int): Nombre de folds pour RFECV.
        n_repeats_perm (int): Nombre de répétitions pour la permutation.
        output_dir: Répertoire de sauvegarde éventuel (non géré ici).
        model_name (str): Nom du modèle pour les logs.
        save_results (bool): Paramètre conservé pour compatibilité (non utilisé ici).
        scoring (str ou callable): Fonction de scoring pour RFECV et la permutation.

    Returns:
        dict: Dictionnaire contenant les résultats de chaque méthode appliquée.
    """
    
    # === DIAGNOSTIC INITIAL ===
    print(f"\n🔍 DIAGNOSTIC - {model_name}")
    print("="*60)
    print(f"📊 Dimensions des données:")
    print(f"   • X_train: {X_train.shape}")
    print(f"   • X_eval: {X_eval.shape}")
    print(f"   • y_train: {y_train.shape}")
    print(f"   • y_eval: {y_eval.shape}")
    print(f"   • feature_names: {len(feature_names)} features")
    
    if hasattr(model, 'n_features_in_'):
        print(f"   • Modèle attend: {model.n_features_in_} features")
    else:
        print(f"   • Modèle: pas d'attribut n_features_in_")
    
    # Vérifier la cohérence
    if X_train.shape[1] != X_eval.shape[1]:
        print(f"⚠️  ATTENTION: X_train et X_eval ont des nombres de features différents!")
        print(f"   • X_train: {X_train.shape[1]} features")
        print(f"   • X_eval: {X_eval.shape[1]} features")
    
    if hasattr(model, 'n_features_in_') and X_train.shape[1] != model.n_features_in_:
        print(f"⚠️  ATTENTION: X_train et le modèle ont des nombres de features différents!")
        print(f"   • X_train: {X_train.shape[1]} features")
        print(f"   • Modèle: {model.n_features_in_} features")
    
    # On fait une copie de feature_names pour éviter les effets de bord
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
    else:
        feature_names = list(feature_names)

    results = {
        'model_name': model_name,
        'methods_applied': [],
        'rfecv': None,
        'permutation': None,
        'shap': None,
        'debug_info': {
            'X_train_shape': X_train.shape,
            'X_eval_shape': X_eval.shape,
            'model_n_features': getattr(model, 'n_features_in_', 'Unknown'),
            'feature_names_length': len(feature_names)
        }
    }

    # Déterminer les méthodes à exécuter
    methods_to_run = method
    if method == 'all':
        methods_to_run = ['rfecv', 'permutation', 'shap']
    if isinstance(methods_to_run, str):
        methods_to_run = [methods_to_run]

    # === RFECV ===
    if 'rfecv' in methods_to_run:
        print(f"\n🎯 RFECV - {model_name}")
        print("-" * 40)
        try:
            cv_splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

            # Définir un getter d'importance compatible avec les versions récentes de scikit‑learn
            def custom_importance_getter(estimator):
                if hasattr(estimator, 'feature_importances_'):
                    return estimator.feature_importances_
                elif hasattr(estimator, 'coef_'):
                    # Pour les modèles multi‑classes, on moyenne les coefficients absolus
                    coefs = np.array(estimator.coef_)
                    return np.mean(np.abs(coefs), axis=0)
                else:
                    return np.ones(estimator.n_features_in_)

            # Choisir si l'on doit passer un importance_getter explicite
            if hasattr(model, 'feature_importances_') or hasattr(model, 'coef_'):
                rfecv_selector = RFECV(estimator=model, step=1, cv=cv_splitter,
                                       scoring=scoring, n_jobs=-1)
            else:
                rfecv_selector = RFECV(estimator=model, step=1, cv=cv_splitter,
                                       scoring=scoring, n_jobs=-1,
                                       importance_getter=custom_importance_getter)

            print(f"   • Entraînement RFECV avec {X_train.shape[1]} features...")
            rfecv_selector.fit(X_train, y_train)

            # Calcul du meilleur score en fonction des attributs disponibles
            best_score = None
            if hasattr(rfecv_selector, 'best_score_'):
                best_score = float(rfecv_selector.best_score_)
            elif hasattr(rfecv_selector, 'cv_results_'):
                idx = np.argmax(rfecv_selector.cv_results_['mean_test_score'])
                best_score = float(rfecv_selector.cv_results_['mean_test_score'][idx])

            results['rfecv'] = {
                'n_features_optimal': int(rfecv_selector.n_features_),
                'best_score': best_score,
                'support': rfecv_selector.support_.tolist(),
                'ranking': rfecv_selector.ranking_.tolist(),
                'grid_scores_mean': rfecv_selector.cv_results_['mean_test_score'].tolist(),
                'grid_scores_std': rfecv_selector.cv_results_['std_test_score'].tolist(),
                'selected_features': [feature_names[i] for i, s in enumerate(rfecv_selector.support_) if s]
            }
            results['methods_applied'].append('rfecv')
            print(f"   ✅ RFECV réussi: {rfecv_selector.n_features_} features optimales")
        except Exception as e:
            print(f"   ❌ Erreur RFECV: {e}")
            logger.error(f"Erreur lors de RFECV pour {model_name}: {e}")

    # === Importance par permutation ===
    if 'permutation' in methods_to_run:
        print(f"\n🔄 Permutation Importance - {model_name}")
        print("-" * 40)
        try:
            # Diagnostic des dimensions
            print(f"   • X_eval shape: {X_eval.shape}")
            if hasattr(model, 'n_features_in_'):
                print(f"   • Modèle attend: {model.n_features_in_} features")
                
                # Vérifier et ajuster les dimensions
                if X_eval.shape[1] != model.n_features_in_:
                    print(f"   ⚠️  Dimensions incompatibles!")
                    
                    if X_eval.shape[1] > model.n_features_in_:
                        # Prendre les premières features
                        X_eval_adjusted = X_eval[:, :model.n_features_in_]
                        tmp_names = feature_names[:model.n_features_in_]
                        print(f"   • Redimensionnement: {X_eval.shape[1]} → {X_eval_adjusted.shape[1]} features")
                    else:
                        # Ajouter des features factices si nécessaire
                        padding = np.zeros((X_eval.shape[0], model.n_features_in_ - X_eval.shape[1]))
                        X_eval_adjusted = np.hstack([X_eval, padding])
                        tmp_names = feature_names + [f"feature_{i}" for i in range(len(feature_names), model.n_features_in_)]
                        print(f"   • Complétion: {X_eval.shape[1]} → {X_eval_adjusted.shape[1]} features")
                else:
                    X_eval_adjusted = X_eval
                    tmp_names = feature_names
                    print(f"   • Dimensions compatibles")
            else:
                X_eval_adjusted = X_eval
                tmp_names = feature_names
                print(f"   • Modèle sans n_features_in_, utilisation directe")

            print(f"   • Calcul permutation importance...")
            perm_result = permutation_importance(model, X_eval_adjusted, y_eval,
                                                 n_repeats=n_repeats_perm,
                                                 random_state=42, n_jobs=-1,
                                                 scoring=scoring)

            df_perm_importance = pd.DataFrame({
                'feature': tmp_names,
                'importance_mean': perm_result.importances_mean,
                'importance_std': perm_result.importances_std
            }).sort_values(by='importance_mean', ascending=False)

            results['permutation'] = {
                'dataframe': df_perm_importance.to_dict(orient='list'),
                'top_20_features': df_perm_importance.head(20)['feature'].tolist()
            }
            results['methods_applied'].append('permutation')
            print(f"   ✅ Permutation importance calculée")
        except Exception as e:
            print(f"   ❌ Erreur permutation: {e}")
            logger.error(f"Erreur lors de la permutation pour {model_name}: {e}")
            # Ajouter un résultat vide pour éviter les erreurs dans les graphiques
            results['permutation'] = {
                'dataframe': {
                    'feature': feature_names[:10] if len(feature_names) >= 10 else feature_names,
                    'importance_mean': [0.0] * min(10, len(feature_names)),
                    'importance_std': [0.0] * min(10, len(feature_names))
                },
                'top_20_features': feature_names[:10] if len(feature_names) >= 10 else feature_names
            }

    # === SHAP ===
    if 'shap' in methods_to_run:
        print(f"\n📊 SHAP Analysis - {model_name}")
        print("-" * 40)
        try:
            import shap

            # Vérifier la compatibilité des dimensions pour SHAP
            if hasattr(model, 'n_features_in_'):
                if X_eval.shape[1] != model.n_features_in_:
                    if X_eval.shape[1] > model.n_features_in_:
                        X_sample = X_eval[:min(100, X_eval.shape[0]), :model.n_features_in_]
                        print(f"   • Redimensionnement SHAP: {X_eval.shape[1]} → {X_sample.shape[1]} features")
                    else:
                        padding = np.zeros((min(100, X_eval.shape[0]), model.n_features_in_ - X_eval.shape[1]))
                        X_sample = np.hstack([X_eval[:min(100, X_eval.shape[0])], padding])
                        print(f"   • Complétion SHAP: {X_eval.shape[1]} → {X_sample.shape[1]} features")
                else:
                    X_sample = X_eval[:min(100, X_eval.shape[0])]
                    print(f"   • Dimensions SHAP compatibles")
            else:
                X_sample = X_eval[:min(100, X_eval.shape[0])]
                print(f"   • Utilisation directe pour SHAP")

            print(f"   • Échantillon SHAP: {X_sample.shape}")
            
            shap_values_for_plot = None
            # Si modèle d'arbres avec predict_proba
            if hasattr(model, 'predict_proba') and hasattr(model, 'estimators_'):
                try:
                    print(f"   • Essai TreeExplainer avec predict_proba...")
                    explainer = shap.TreeExplainer(model, model_output="predict_proba")
                    shap_values = explainer.shap_values(X_sample)
                    # shap_values est une liste (une par classe) : on prend la classe positive s'il y en a plusieurs
                    if isinstance(shap_values, list):
                        shap_values_for_plot = shap_values[1] if len(shap_values) > 1 else shap_values[0]
                    else:
                        shap_values_for_plot = shap_values
                    print(f"   ✅ TreeExplainer predict_proba réussi")
                except Exception as e:
                    print(f"   ⚠️  Erreur predict_proba: {e}")
                    print(f"   • Essai TreeExplainer avec raw...")
                    # Fallback vers raw
                    explainer = shap.TreeExplainer(model, model_output="raw")
                    shap_values = explainer.shap_values(X_sample)
                    if isinstance(shap_values, list):
                        shap_values_for_plot = shap_values[1] if len(shap_values) > 1 else shap_values[0]
                    else:
                        shap_values_for_plot = shap_values
                    print(f"   ✅ TreeExplainer raw réussi")
            else:
                print(f"   • Utilisation Explainer générique...")
                # Explainer générique pour les autres modèles
                explainer = shap.Explainer(model, X_sample)
                shap_values = explainer(X_sample)
                # shap_values.values : (n_samples, n_features, n_outputs) éventuel
                if hasattr(shap_values, 'values'):
                    sv = shap_values.values
                    if sv.ndim == 3:
                        shap_values_for_plot = sv[:, :, 1] if sv.shape[2] > 1 else sv[:, :, 0]
                    else:
                        shap_values_for_plot = sv
                else:
                    shap_values_for_plot = shap_values
                print(f"   ✅ Explainer générique réussi")

            results['shap'] = {
                'explainer_type': type(explainer).__name__,
                'n_samples_used': X_sample.shape[0],
            }
            results['methods_applied'].append('shap')
            print(f"   ✅ Analyse SHAP réussie")

        except ImportError:
            print(f"   ❌ Module SHAP non installé")
            logger.warning("Le module SHAP n'est pas installé ; l'analyse SHAP est ignorée.")
        except Exception as e:
            print(f"   ❌ Erreur SHAP: {e}")
            logger.error(f"Erreur lors de l'analyse SHAP pour {model_name}: {e}")
            # Ajouter un résultat vide pour éviter les erreurs dans les graphiques
            results['shap'] = {
                'explainer_type': 'Error',
                'n_samples_used': 0,
            }

    print(f"\n✅ Analyse terminée - {model_name}")
    print("="*60)
    return results 