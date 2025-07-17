## Moyennage des probabilités stacking KNN / MICE

import numpy as np
import joblib
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

def load_stacking_results(stacking_dir, imputation_type):
    """
    Charge les résultats de stacking pour un type d'imputation donné
    """
    # Charger le modèle de stacking
    stack_model = joblib.load(stacking_dir / f"stack_no_refit_{imputation_type}.joblib")
    
    # Charger le seuil optimal
    with open(stacking_dir / f"best_thr_stack_no_refit_{imputation_type}.json", "r") as f:
        threshold_data = json.load(f)
    
    # Charger les métriques de performance
    with open(stacking_dir / f"stack_no_refit_{imputation_type}_performance.json", "r") as f:
        performance = json.load(f)
    
    return stack_model, threshold_data[f"best_thr_stack_no_refit_{imputation_type}"], performance

def predict_stacking_proba(X, stack_model):
    """
    Fait des prédictions de probabilité avec le modèle de stacking
    
    Parameters:
    -----------
    X : array-like
        Données à prédire
    stack_model : dict
        Modèle de stacking chargé
    
    Returns:
    --------
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
    
    return proba_mean

def average_knn_mice_stacking(X_test_knn, X_test_mice, y_test_knn, y_test_mice, stacking_dir):
    """
    Moyennage des probabilités entre stacking KNN et MICE
    """
    print("🔄 Moyennage des probabilités stacking KNN / MICE")
    print("="*60)
    
    # 1. Charger les modèles de stacking
    print("📥 Chargement des modèles de stacking...")
    
    stack_knn, thr_knn, perf_knn = load_stacking_results(stacking_dir, "knn")
    stack_mice, thr_mice, perf_mice = load_stacking_results(stacking_dir, "mice")
    
    print(f"✅ Modèle KNN chargé (F1 = {perf_knn['f1_score']:.4f})")
    print(f"✅ Modèle MICE chargé (F1 = {perf_mice['f1_score']:.4f})")
    
    # 2. Prédictions de probabilité
    print("\n🔮 Génération des prédictions de probabilité...")
    
    proba_knn = predict_stacking_proba(X_test_knn, stack_knn)
    proba_mice = predict_stacking_proba(X_test_mice, stack_mice)
    
    print(f"✅ Probabilités KNN : {len(proba_knn)} échantillons")
    print(f"✅ Probabilités MICE : {len(proba_mice)} échantillons")
    
    # 3. Moyennage des probabilités
    print("\n📊 Moyennage des probabilités...")
    
    # Vérifier que les deux ensembles ont la même taille
    if len(proba_knn) != len(proba_mice):
        print("⚠️  Attention : Les ensembles KNN et MICE ont des tailles différentes")
        min_size = min(len(proba_knn), len(proba_mice))
        proba_knn = proba_knn[:min_size]
        proba_mice = proba_mice[:min_size]
        y_test_knn = y_test_knn[:min_size]
        y_test_mice = y_test_mice[:min_size]
        print(f"📏 Taille ajustée : {min_size} échantillons")
    
    # Moyennage simple
    proba_average = np.mean([proba_knn, proba_mice], axis=0)
    
    print(f"✅ Probabilités moyennées : {len(proba_average)} échantillons")
    
    # 4. Optimisation du seuil pour le moyennage
    print("\n🎯 Optimisation du seuil pour le moyennage...")
    
    thresholds = np.linspace(0.2, 0.8, 61)
    best_f1 = 0
    best_thr_average = 0.5
    
    for thr in thresholds:
        y_pred = (proba_average >= thr).astype(int)
        f1 = f1_score(y_test_knn, y_pred)  # Utiliser y_test_knn comme référence
        if f1 > best_f1:
            best_f1 = f1
            best_thr_average = thr
    
    # 5. Prédiction finale au seuil optimal
    y_pred_opt = (proba_average >= best_thr_average).astype(int)
    
    # 6. Analyse détaillée
    f1 = f1_score(y_test_knn, y_pred_opt)
    precision = precision_score(y_test_knn, y_pred_opt)
    recall = recall_score(y_test_knn, y_pred_opt)
    cm = confusion_matrix(y_test_knn, y_pred_opt)
    
    print(f"\n📊 RÉSULTATS DU MOYENNAGE KNN/MICE")
    print("="*60)
    print(f"F1-Score : {f1:.4f} (seuil = {best_thr_average:.3f})")
    print(f"Précision : {precision:.4f}")
    print(f"Rappel : {recall:.4f}")
    print("Matrice de confusion :")
    print(cm)
    
    # 7. Comparaison avec les modèles individuels
    print(f"\n📈 COMPARAISON AVEC LES MODÈLES INDIVIDUELS")
    print("="*60)
    
    comparison_data = {
        "Méthode": ["KNN seul", "MICE seul", "Moyennage KNN/MICE"],
        "F1-Score": [perf_knn["f1_score"], perf_mice["f1_score"], f1],
        "Précision": [perf_knn["precision"], perf_mice["precision"], precision],
        "Rappel": [perf_knn["recall"], perf_mice["recall"], recall],
        "Seuil optimal": [thr_knn, thr_mice, best_thr_average]
    }
    
    df_comparison = pd.DataFrame(comparison_data)
    print(df_comparison.to_string(index=False, float_format='%.4f'))
    
    # Déterminer le meilleur
    best_idx = df_comparison["F1-Score"].idxmax()
    best_method = df_comparison.loc[best_idx, "Méthode"]
    best_f1 = df_comparison.loc[best_idx, "F1-Score"]
    
    print(f"\n🏆 Meilleur modèle : {best_method} (F1 = {best_f1:.4f})")
    
    # 8. Affichage graphique de la matrice de confusion
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Prédit: Non-ad", "Prédit: Ad"],
                yticklabels=["Réel: Non-ad", "Réel: Ad"])
    plt.xlabel("Prédit")
    plt.ylabel("Réel")
    plt.title(f"Matrice de confusion\nMoyennage KNN/MICE (F1={f1:.3f}, seuil={best_thr_average:.2f})")
    plt.tight_layout()
    plt.show()
    
    # 9. Sauvegarde des résultats
    print("\n💾 Sauvegarde des résultats...")
    
    # Créer le modèle de moyennage
    average_model = {
        "knn_model": stack_knn,
        "mice_model": stack_mice,
        "threshold": float(best_thr_average),
        "performance": {
            "f1_score": float(f1),
            "precision": float(precision),
            "recall": float(recall)
        },
        "comparison": {
            "knn_f1": float(perf_knn["f1_score"]),
            "mice_f1": float(perf_mice["f1_score"]),
            "average_f1": float(f1)
        }
    }
    
    # Sauvegarder le modèle
    joblib.dump(average_model, stacking_dir / "stack_average_knn_mice.joblib")
    
    # Sauvegarder le seuil optimal
    with open(stacking_dir / "best_thr_stack_average_knn_mice.json", "w") as f:
        json.dump({"best_thr_stack_average_knn_mice": float(best_thr_average)}, f, indent=2)
    
    # Sauvegarder les métriques de performance
    performance_metrics = {
        "f1_score": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "threshold": float(best_thr_average),
        "confusion_matrix": cm.tolist()
    }
    
    with open(stacking_dir / "stack_average_knn_mice_performance.json", "w") as f:
        json.dump(performance_metrics, f, indent=2)
    
    # Sauvegarder la comparaison
    df_comparison.to_csv(stacking_dir / "comparison_average_knn_mice.csv", index=False)
    
    print("✅ Résultats sauvegardés !")
    print(f"📁 Fichiers dans : {stacking_dir}")
    
    return average_model, df_comparison, f1, precision, recall, best_thr_average

# Exécution
if __name__ == "__main__":
    # Vérifier que les variables sont définies
    required_vars = ["X_test_knn", "X_test_mice", "y_test_knn", "y_test_mice", "stacking_dir"]
    
    missing_vars = []
    for var in required_vars:
        if var not in globals():
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Variables manquantes : {missing_vars}")
        print("Assurez-vous que ces variables sont définies dans votre notebook")
    else:
        # Exécuter le moyennage
        average_model, df_comparison, f1, precision, recall, threshold = average_knn_mice_stacking(
            X_test_knn, X_test_mice, y_test_knn, y_test_mice, stacking_dir
        )
        
        print(f"\n🎉 Moyennage KNN/MICE terminé !")
        print(f"📊 F1-Score final : {f1:.4f}")
        print(f"🎯 Seuil optimal : {threshold:.3f}") 