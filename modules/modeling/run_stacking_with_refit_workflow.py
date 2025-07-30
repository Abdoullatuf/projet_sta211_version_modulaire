# modules/modeling/run_stacking_with_refit_workflow.py

import logging
from pathlib import Path
import numpy as np
import joblib
import json
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

# Supposons que ces imports proviennent de vos modules ou sont définis localement
# from modules.modeling.stacking_models_creator import create_stacking_models
# from modules.evaluation.evaluate_predictions import evaluate_from_probabilities

logger = logging.getLogger(__name__)

def run_stacking_with_refit(X_train, y_train, X_val, y_val, X_test, y_test,
                            imputation_method, models_dir, output_dir=None,
                            model_name_suffix="", create_stacking_func=None,
                            threshold_optimization_func=None,
                            stacking_model_key=None, context_name="notebook3"):
    """
    Exécute le processus complet de stacking avec refit : création, entraînement,
    optimisation du seuil, évaluation et sauvegarde.

    Args:
        X_train, y_train, X_val, y_val, X_test, y_test: Données d'entraînement, validation et test.
        imputation_method (str): 'knn' ou 'mice'.
        models_dir (Path): Chemin du répertoire de sauvegarde des modèles.
        output_dir (Path, optional): Chemin du répertoire de sauvegarde des résultats.
        model_name_suffix (str): Suffixe pour le nom du modèle (ex: '_reduced').
        create_stacking_func (callable, optional):
            Fonction pour créer le modèle de stacking.
            Doit prendre `imputation_method` et `verbose` comme arguments.
            Si None, utilise `create_stacking_models` par défaut.
        threshold_optimization_func (callable, optional):
            Fonction personnalisée pour optimiser le seuil.
        stacking_model_key (str, optional):
            Clé pour extraire le modèle de stacking du dictionnaire retourné par create_stacking_func.
            Nécessaire si create_stacking_func retourne un dict.
            Ex: 'stacking_classifier_knn' ou 'stacking_classifier_mice'.
        context_name (str, optional): Nom du contexte (par exemple, 'notebook2') pour organiser les sorties.
                                      La valeur par défaut est 'notebook3'.

    Returns:
        dict: Dictionnaire contenant le modèle, les métriques, le seuil et les chemins de sauvegarde.
              Retourne None en cas d'erreur.
    """
    logger.info(f"🔄 Démarrage du Stacking avec Refit - {imputation_method.upper()} {model_name_suffix}...")

    if output_dir is None:
        stacking_dir = Path(models_dir) / context_name / "stacking"
    else:
        stacking_dir = Path(output_dir)
    stacking_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'model': None,
        'metrics': {},
        'threshold': None,
        'paths': {}
    }

    try:
        # --- 1. Création du modèle ---
        if create_stacking_func is None:
            from modules.modeling.stacking_models_creator import create_stacking_models
            models_output = create_stacking_models(imputation_method=imputation_method, verbose=True)
        else:
            models_output = create_stacking_func(imputation_method=imputation_method, verbose=True)

        # Gestion du retour de la fonction de création
        if isinstance(models_output, dict):
            # Cas où la fonction retourne un dictionnaire
            if stacking_model_key is None:
                 # Essayer de déduire la clé si elle n'est pas fournie
                 # Cela dépend de la convention de nommage de votre fonction create_stacking_models
                 default_key = f"stacking_classifier_{imputation_method}"
                 if default_key in models_output:
                     stacking_model_key = default_key
                 else:
                     # Si impossible de déduire, prendre la première clé ou lever une erreur
                     available_keys = list(models_output.keys())
                     if available_keys:
                         stacking_model_key = available_keys[0] # Prendre la première par défaut
                         logger.warning(f"Clé de modèle non spécifiée. Utilisation de '{stacking_model_key}' parmi {available_keys}.")
                     else:
                         raise ValueError("La fonction create_stacking_func a retourné un dictionnaire vide.")

            if stacking_model_key not in models_output:
                raise KeyError(f"Clé '{stacking_model_key}' non trouvée dans le dictionnaire retourné par create_stacking_func. Clés disponibles: {list(models_output.keys())}")

            stacking_model = models_output[stacking_model_key]
            logger.debug(f"✅ Modèle extrait du dictionnaire avec la clé '{stacking_model_key}'.")
        else:
            # Cas où la fonction retourne directement le modèle
            stacking_model = models_output
            logger.debug("✅ Modèle retourné directement par la fonction de création.")

        if stacking_model is None:
            error_msg = f"Échec de la création du modèle Stacking pour {imputation_method}"
            logger.error(f"❌ {error_msg}")
            raise ValueError(error_msg)

        logger.info(f"✅ Modèle Stacking {imputation_method.upper()} créé.")

        # --- 2. Entraînement du modèle ---
        stacking_model.fit(X_train, y_train)
        logger.info(f"✅ Modèle Stacking {imputation_method.upper()} entraîné.")

        # --- 3. Prédictions sur Validation (pour optimisation du seuil) ---
        y_proba_val = stacking_model.predict_proba(X_val)[:, 1]

        # --- 4. Optimisation du seuil ---
        logger.info(f"🔍 Optimisation du seuil pour {imputation_method.upper()}...")
        if threshold_optimization_func is None:
            best_threshold, best_f1_val = _optimize_threshold_internal(y_val, y_proba_val)
        else:
            best_threshold, best_f1_val = threshold_optimization_func(y_val, y_proba_val)

        logger.info(f"✅ Seuil optimal {imputation_method.upper()}: {best_threshold:.3f} (F1-val: {best_f1_val:.4f})")
        results['threshold'] = best_threshold

        # --- 5. Évaluation sur Test ---
        y_proba_test = stacking_model.predict_proba(X_test)[:, 1]
        y_pred_test = (y_proba_test >= best_threshold).astype(int)

        f1_test = f1_score(y_test, y_pred_test)
        precision_test = precision_score(y_test, y_pred_test)
        recall_test = recall_score(y_test, y_pred_test)
        cm_test = confusion_matrix(y_test, y_pred_test)

        logger.info(f"📊 Résultats sur Test - {imputation_method.upper()}{model_name_suffix}:")
        logger.info(f"   F1-score: {f1_test:.4f}")
        logger.info(f"   Précision: {precision_test:.4f}")
        logger.info(f"   Rappel: {recall_test:.4f}")

        results['model'] = stacking_model
        # Calcul des métriques de validation au seuil optimal
        y_pred_val_opt = (y_proba_val >= best_threshold).astype(int)
        results['metrics'] = {
            'f1_score_val': float(best_f1_val), # Déjà calculé
            'precision_val': float(precision_score(y_val, y_pred_val_opt)),
            'recall_val': float(recall_score(y_val, y_pred_val_opt)),
            'f1_score_test': float(f1_test),
            'precision_test': float(precision_test),
            'recall_test': float(recall_test),
            'confusion_matrix_test': cm_test.tolist()
        }

        # --- 6. Sauvegarde du modèle ---
        model_filename = f"stacking_{imputation_method}_with_refit{model_name_suffix}.joblib"
        model_path = stacking_dir / model_filename
        joblib.dump(stacking_model, model_path)
        logger.info(f"💾 Modèle sauvegardé: {model_path}")
        results['paths']['model'] = model_path

        # --- 7. Sauvegarde des résultats (JSON) ---
        results_filename = f"stacking_with_refit_{imputation_method}{model_name_suffix}.json"
        results_path = stacking_dir / results_filename
        results_to_save = {
            "method": f"stacking_with_refit_{imputation_method}{model_name_suffix}",
            "threshold": float(best_threshold),
            "performance": results['metrics'],
            "predictions": {
                "test_proba": y_proba_test.tolist(),
                "test_pred": y_pred_test.tolist()
            }
        }
        with open(results_path, 'w') as f:
            json.dump(results_to_save, f, indent=4)
        logger.info(f"💾 Résultats sauvegardés: {results_path}")
        results['paths']['results'] = results_path

    except Exception as e:
        logger.error(f"❌ Erreur lors de l'exécution du Stacking avec Refit - {imputation_method.upper()}: {e}", exc_info=True)
        return None

    logger.info(f"🎉 Stacking avec Refit - {imputation_method.upper()} {model_name_suffix} terminé.")
    return results

def _optimize_threshold_internal(y_true, y_proba, metric='f1', thresholds=None):
    """Optimise un seuil de classification pour une métrique donnée (fonction interne)."""
    if thresholds is None:
        thresholds = np.linspace(0.2, 0.8, 61)

    best_score = -np.inf
    best_threshold = 0.5

    metric_funcs = {
        'f1': f1_score,
        'precision': precision_score,
        'recall': recall_score
    }

    if metric not in metric_funcs:
        raise ValueError(f"Métrique '{metric}' non supportée.")

    metric_func = metric_funcs[metric]

    for thr in thresholds:
        y_pred = (y_proba >= thr).astype(int)
        try:
            score = metric_func(y_true, y_pred)
            if np.isfinite(score) and score > best_score:
                best_score = score
                best_threshold = thr
        except Exception:
            pass # Ignore les erreurs de calcul

    return best_threshold, best_score

# --- Exemple d'utilisation ---
# 1. Si create_stacking_models retourne un dict et que vous connaissez la clé:
# results_knn = run_stacking_with_refit(
#     X_train_knn, y_train_knn, X_val_knn, y_val_knn, X_test_knn, y_test_knn,
#     imputation_method='knn',
#     models_dir=MODELS_DIR,
#     output_dir=Path("outputs/modeling/results"),
#     stacking_model_key='stacking_classifier_knn' # <-- Clé spécifiée
# )
