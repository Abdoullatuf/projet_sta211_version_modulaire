## Comparaison des résultats de stacking sans refit (KNN vs MICE)

import numpy as np
import joblib
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

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

def compare_stacking_results(stacking_dir):
    """
    Compare les résultats de stacking entre KNN et MICE
    """
    print("Chargement des résultats de stacking...")
    
    # Charger les résultats KNN
    stack_knn, thr_knn, perf_knn = load_stacking_results(stacking_dir, "knn")
    
    # Charger les résultats MICE
    stack_mice, thr_mice, perf_mice = load_stacking_results(stacking_dir, "mice")
    
    # Créer un DataFrame de comparaison
    comparison_data = {
        "Méthode": ["KNN", "MICE"],
        "F1-Score": [perf_knn["f1_score"], perf_mice["f1_score"]],
        "Précision": [perf_knn["precision"], perf_mice["precision"]],
        "Rappel": [perf_knn["recall"], perf_mice["recall"]],
        "Seuil optimal": [thr_knn, thr_mice]
    }
    
    df_comparison = pd.DataFrame(comparison_data)
    
    print("\n" + "="*60)
    print("COMPARAISON STACKING SANS REFIT (KNN vs MICE)")
    print("="*60)
    print(df_comparison.to_string(index=False, float_format='%.4f'))
    
    # Déterminer le meilleur modèle
    best_idx = df_comparison["F1-Score"].idxmax()
    best_method = df_comparison.loc[best_idx, "Méthode"]
    best_f1 = df_comparison.loc[best_idx, "F1-Score"]
    
    print(f"\n🏆 Meilleur modèle : {best_method} (F1 = {best_f1:.4f})")
    
    # Différences
    f1_diff = abs(perf_knn["f1_score"] - perf_mice["f1_score"])
    print(f"📊 Différence de F1-score : {f1_diff:.4f}")
    
    return df_comparison, stack_knn, stack_mice, thr_knn, thr_mice

def plot_comparison(df_comparison, stacking_dir):
    """
    Crée des graphiques de comparaison
    """
    # Graphique des métriques
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # F1-Score
    axes[0, 0].bar(df_comparison["Méthode"], df_comparison["F1-Score"], 
                   color=['skyblue', 'lightcoral'])
    axes[0, 0].set_title("F1-Score")
    axes[0, 0].set_ylim(0, 1)
    for i, v in enumerate(df_comparison["F1-Score"]):
        axes[0, 0].text(i, v + 0.01, f'{v:.4f}', ha='center')
    
    # Précision
    axes[0, 1].bar(df_comparison["Méthode"], df_comparison["Précision"], 
                   color=['skyblue', 'lightcoral'])
    axes[0, 1].set_title("Précision")
    axes[0, 1].set_ylim(0, 1)
    for i, v in enumerate(df_comparison["Précision"]):
        axes[0, 1].text(i, v + 0.01, f'{v:.4f}', ha='center')
    
    # Rappel
    axes[1, 0].bar(df_comparison["Méthode"], df_comparison["Rappel"], 
                   color=['skyblue', 'lightcoral'])
    axes[1, 0].set_title("Rappel")
    axes[1, 0].set_ylim(0, 1)
    for i, v in enumerate(df_comparison["Rappel"]):
        axes[1, 0].text(i, v + 0.01, f'{v:.4f}', ha='center')
    
    # Seuil optimal
    axes[1, 1].bar(df_comparison["Méthode"], df_comparison["Seuil optimal"], 
                   color=['skyblue', 'lightcoral'])
    axes[1, 1].set_title("Seuil optimal")
    axes[1, 1].set_ylim(0, 1)
    for i, v in enumerate(df_comparison["Seuil optimal"]):
        axes[1, 1].text(i, v + 0.01, f'{v:.3f}', ha='center')
    
    plt.tight_layout()
    plt.savefig(stacking_dir / "comparison_stacking_no_refit.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Sauvegarder le DataFrame de comparaison
    df_comparison.to_csv(stacking_dir / "comparison_stacking_no_refit.csv", index=False)
    print(f"\n📁 Résultats sauvegardés dans : {stacking_dir}")

# Exécution de la comparaison
if __name__ == "__main__":
    df_comp, stack_knn, stack_mice, thr_knn, thr_mice = compare_stacking_results(stacking_dir)
    plot_comparison(df_comp, stacking_dir) 