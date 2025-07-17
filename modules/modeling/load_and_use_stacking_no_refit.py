## Chargement et utilisation du stacking sans refit

import numpy as np
import joblib
import json
from pathlib import Path

def load_stacking_no_refit_knn(stacking_dir):
    """
    Charge le modèle de stacking sans refit pour les données KNN
    """
    # Charger le modèle de stacking
    stack_model = joblib.load(stacking_dir / "stack_no_refit_knn.joblib")
    
    # Charger le seuil optimal
    with open(stacking_dir / "best_thr_stack_no_refit_knn.json", "r") as f:
        threshold_data = json.load(f)
    
    return stack_model, threshold_data["best_thr_stack_no_refit_knn"]

def predict_stacking_no_refit(X, stack_model, threshold):
    """
    Fait des prédictions avec le stacking sans refit
    
    Parameters:
    -----------
    X : array-like
        Données à prédire
    stack_model : dict
        Modèle de stacking chargé
    threshold : float
        Seuil optimal pour la classification
    
    Returns:
    --------
    y_pred : array
        Prédictions binaires
    proba_mean : array
        Probabilités moyennes
    """
    pipelines = stack_model["pipelines"]
    
    # Prédictions de probabilité pour chaque modèle
    proba_rf = pipelines["rf"].predict_proba(X)[:, 1]
    proba_xgb = pipelines["xgb"].predict_proba(X)[:, 1]
    proba_logreg = pipelines["logreg"].predict_proba(X)[:, 1]
    proba_svm = pipelines["svm"].predict_proba(X)[:, 1]
    proba_mlp = pipelines["mlp"].predict_proba(X)[:, 1]
    
    # Moyenne des probabilités
    proba_mean = np.mean([proba_rf, proba_xgb, proba_logreg, proba_svm, proba_mlp], axis=0)
    
    # Prédictions binaires au seuil optimal
    y_pred = (proba_mean >= threshold).astype(int)
    
    return y_pred, proba_mean

# Exemple d'utilisation
if __name__ == "__main__":
    # Charger le modèle et le seuil
    stack_model, threshold = load_stacking_no_refit_knn(stacking_dir)
    
    print(f"Seuil optimal chargé : {threshold:.3f}")
    print(f"Modèle de stacking chargé avec {len(stack_model['pipelines'])} pipelines")
    
    # Faire des prédictions sur de nouvelles données
    # y_pred, proba_mean = predict_stacking_no_refit(X_new, stack_model, threshold)
    
    # Afficher les métriques de performance sauvegardées
    if "performance" in stack_model:
        perf = stack_model["performance"]
        print(f"\nMétriques de performance sauvegardées :")
        print(f"F1-score : {perf['f1_score']:.4f}")
        print(f"Précision : {perf['precision']:.4f}")
        print(f"Rappel : {perf['recall']:.4f}") 