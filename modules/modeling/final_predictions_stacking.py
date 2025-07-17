## Prédictions finales avec le meilleur modèle de stacking sans refit

import numpy as np
import joblib
import json
import pandas as pd
from pathlib import Path

def load_best_stacking_model(stacking_dir):
    """
    Charge le meilleur modèle de stacking (KNN ou MICE) basé sur les performances
    """
    # Charger les performances KNN
    with open(stacking_dir / "stack_no_refit_knn_performance.json", "r") as f:
        perf_knn = json.load(f)
    
    # Charger les performances MICE
    with open(stacking_dir / "stack_no_refit_mice_performance.json", "r") as f:
        perf_mice = json.load(f)
    
    # Déterminer le meilleur modèle
    if perf_knn["f1_score"] >= perf_mice["f1_score"]:
        best_method = "knn"
        best_f1 = perf_knn["f1_score"]
        print(f"🏆 Meilleur modèle : KNN (F1 = {best_f1:.4f})")
    else:
        best_method = "mice"
        best_f1 = perf_mice["f1_score"]
        print(f"🏆 Meilleur modèle : MICE (F1 = {best_f1:.4f})")
    
    # Charger le modèle de stacking
    stack_model = joblib.load(stacking_dir / f"stack_no_refit_{best_method}.joblib")
    
    # Charger le seuil optimal
    with open(stacking_dir / f"best_thr_stack_no_refit_{best_method}.json", "r") as f:
        threshold_data = json.load(f)
    
    threshold = threshold_data[f"best_thr_stack_no_refit_{best_method}"]
    
    return stack_model, threshold, best_method, best_f1

def predict_with_stacking(X, stack_model, threshold):
    """
    Fait des prédictions avec le modèle de stacking
    
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

def make_final_predictions(X_test, stacking_dir, output_dir):
    """
    Fait les prédictions finales avec le meilleur modèle de stacking
    """
    print("Chargement du meilleur modèle de stacking...")
    stack_model, threshold, best_method, best_f1 = load_best_stacking_model(stacking_dir)
    
    print(f"Seuil optimal : {threshold:.3f}")
    print(f"Modèle utilisé : {best_method.upper()}")
    
    # Faire les prédictions
    print("Génération des prédictions...")
    y_pred, proba_mean = predict_with_stacking(X_test, stack_model, threshold)
    
    # Créer un DataFrame avec les résultats
    results_df = pd.DataFrame({
        'prediction': y_pred,
        'probability': proba_mean
    })
    
    # Sauvegarder les prédictions
    output_file = output_dir / f"predictions_stacking_{best_method}.csv"
    results_df.to_csv(output_file, index=False)
    
    print(f"✅ Prédictions sauvegardées dans : {output_file}")
    print(f"📊 Nombre de prédictions positives : {y_pred.sum()}")
    print(f"📊 Nombre de prédictions négatives : {(y_pred == 0).sum()}")
    print(f"📊 Taux de prédiction positive : {y_pred.mean():.3f}")
    
    # Sauvegarder les informations du modèle utilisé
    model_info = {
        "best_method": best_method,
        "threshold": float(threshold),
        "f1_score": float(best_f1),
        "n_predictions": len(y_pred),
        "n_positive": int(y_pred.sum()),
        "n_negative": int((y_pred == 0).sum()),
        "positive_rate": float(y_pred.mean())
    }
    
    info_file = output_dir / f"model_info_stacking_{best_method}.json"
    with open(info_file, "w") as f:
        json.dump(model_info, f, indent=2)
    
    print(f"📋 Informations du modèle sauvegardées dans : {info_file}")
    
    return results_df, model_info

# Exemple d'utilisation
if __name__ == "__main__":
    # Charger les données de test (à adapter selon ton projet)
    # X_test = load_test_data()
    
    # Faire les prédictions finales
    # results_df, model_info = make_final_predictions(X_test, stacking_dir, output_dir)
    
    print("Script de prédictions finales avec stacking sans refit")
    print("Utilisez make_final_predictions() pour faire des prédictions sur de nouvelles données") 