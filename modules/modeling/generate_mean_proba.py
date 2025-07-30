#modules/modeling/generate_mean_proba.py

import numpy as np

def generate_mean_proba(pipeline_list, X):
    """
    Génère la moyenne des probabilités pour une liste de pipelines.

    Args:
        pipeline_list (list): Liste d'objets pipeline scikit-learn.
        X (array-like): Données d'entrée.

    Returns:
        np.ndarray: Vecteur de probabilités moyennes (classe 1).
    """
    if not pipeline_list:
        raise ValueError("La liste des pipelines est vide.")

    probas = []
    for pipeline in pipeline_list:
        if pipeline is not None:
             try:
                 proba = pipeline.predict_proba(X)[:, 1]
                 probas.append(proba)
             except Exception as e:
                 print(f"⚠️ Erreur lors de la prédiction avec un pipeline: {e}")
                 # Gérer le cas où un pipeline échoue?
                 # Par exemple, ignorer ou lever une erreur
                 pass # Ignore ici

    if not probas:
        raise RuntimeError("Aucune probabilité valide n'a pu être générée.")

    mean_proba = np.mean(probas, axis=0)
    return mean_proba

# Exemple d'utilisation (dans la section Stacking sans refit KNN)
# pipelines_knn_full = [
#     load_pipeline('randforest', 'knn', 'full'),
#     load_pipeline('xgboost', 'knn', 'full'),
#     load_pipeline('svm', 'knn', 'full'),
#     load_pipeline('mlp', 'knn', 'full'),
#     load_pipeline('gradboost', 'knn', 'full')
# ]
# proba_mean_val_knn_no_refit = generate_mean_proba(pipelines_knn_full, X_val_knn)
# proba_mean_test_knn_no_refit = generate_mean_proba(pipelines_knn_full, X_test_knn)