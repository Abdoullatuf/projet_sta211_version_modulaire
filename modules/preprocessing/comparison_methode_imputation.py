# modules/preprocessing/comparison_methode_imputation.py


## 6.3 Comparaison des méthodes d'imputation - PARTIE 1/4

print("📊 COMPARAISON DES MÉTHODES D'IMPUTATION")
print("="*60)

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Créer les répertoires nécessaires
Path("outputs/figures").mkdir(parents=True, exist_ok=True)

# ============================================================================
# FONCTIONS DE COMPARAISON
# ============================================================================

def compare_imputation_methods(datasets_dict):
    """
    Compare les différentes méthodes d'imputation générées.
    """
    print("\n🔍 ANALYSE COMPARATIVE DES DATASETS")
    print("-" * 50)
    
    comparison_data = []
    
    for dataset_name, dataset_info in datasets_dict.items():
        if dataset_info['status'] == 'success':
            df = dataset_info['dataframe']
            
            # Statistiques de base
            stats = {
                'Dataset': dataset_name,
                'Dimensions': f"{df.shape[0]}x{df.shape[1]}",
                'Lignes': df.shape[0],
                'Colonnes': df.shape[1],
                'Valeurs_manquantes': df.isnull().sum().sum(),
                'Taille_MB': dataset_info.get('file_size_mb', 'N/A')
            }
            
            # Statistiques des variables continues
            continuous_vars = ['X1_trans', 'X2_trans', 'X3_trans']
            if all(col in df.columns for col in continuous_vars):
                for var in continuous_vars:
                    stats[f'{var}_mean'] = round(df[var].mean(), 4)
                    stats[f'{var}_std'] = round(df[var].std(), 4)
                    stats[f'{var}_min'] = round(df[var].min(), 4)
                    stats[f'{var}_max'] = round(df[var].max(), 4)
            
            # Distribution de la target
            if 'outcome' in df.columns:
                target_dist = df['outcome'].value_counts()
                stats['Target_0'] = target_dist.get(0, 0)
                stats['Target_1'] = target_dist.get(1, 0)
                stats['Ratio_desequilibre'] = round(target_dist.get(0, 0) / target_dist.get(1, 1), 2)
            
            comparison_data.append(stats)
    
    return comparison_data

def plot_distributions_comparison(datasets_dict, save_path=None):
    """
    Compare les distributions des variables continues entre méthodes.
    """
    print("\n📊 Génération des graphiques de distribution...")
    
    # Préparer les données
    methods = []
    datasets = {}
    
    for name, info in datasets_dict.items():
        if info['status'] == 'success':
            df = info['dataframe']
            method = 'KNN' if 'knn' in name else 'MICE'
            outliers = 'Avec outliers' if 'with_outliers' in name else 'Sans outliers'
            key = f"{method} - {outliers}"
            methods.append(key)
            datasets[key] = df
    
    if len(datasets) < 2:
        print("⚠️ Pas assez de datasets pour la comparaison")
        return
    
    # Graphiques des distributions
    continuous_vars = ['X1_trans', 'X2_trans', 'X3_trans']
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Comparaison des Distributions après Imputation', fontsize=16, fontweight='bold')
    
    for i, var in enumerate(continuous_vars):
        ax = axes[i]
        
        for j, (method, df) in enumerate(datasets.items()):
            if var in df.columns:
                ax.hist(df[var], bins=30, alpha=0.6, label=method, 
                       color=colors[j % len(colors)], density=True)
        
        ax.set_title(f'Distribution de {var}')
        ax.set_xlabel(var)
        ax.set_ylabel('Densité')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Graphiques sauvegardés: {save_path}")
    else:
        plt.show()

print("✅ Fonctions de comparaison définies")


## 6.3 Comparaison des méthodes d'imputation - PARTIE 2/4

def plot_statistics_comparison(datasets_dict, save_path=None):
    """
    Compare les statistiques (moyennes, écarts-types) entre méthodes.
    """
    print("\n📈 Génération des graphiques statistiques...")
    
    # Préparer les données
    methods = []
    datasets = {}
    
    for name, info in datasets_dict.items():
        if info['status'] == 'success':
            df = info['dataframe']
            method = 'KNN' if 'knn' in name else 'MICE'
            outliers = 'Avec outliers' if 'with_outliers' in name else 'Sans outliers'
            key = f"{method}\n{outliers}"
            methods.append(key)
            datasets[key] = df
    
    if len(datasets) < 2:
        return
    
    # Configuration
    continuous_vars = ['X1_trans', 'X2_trans', 'X3_trans']
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Analyse Statistique Comparative', fontsize=16, fontweight='bold')
    
    method_names = list(datasets.keys())
    
    # 1. Comparaison des moyennes
    ax = axes[0, 0]
    means_data = {}
    for var in continuous_vars:
        means_data[var] = [datasets[method][var].mean() 
                          for method in method_names 
                          if var in datasets[method].columns]
    
    x = np.arange(len(method_names))
    width = 0.25
    
    for i, var in enumerate(continuous_vars):
        ax.bar(x + i*width, means_data[var], width, 
              label=var, color=colors[i], alpha=0.8)
    
    ax.set_xlabel('Méthodes d\'imputation')
    ax.set_ylabel('Moyenne')
    ax.set_title('Comparaison des Moyennes')
    ax.set_xticks(x + width)
    ax.set_xticklabels(method_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Comparaison des écarts-types
    ax = axes[0, 1]
    stds_data = {}
    for var in continuous_vars:
        stds_data[var] = [datasets[method][var].std() 
                         for method in method_names 
                         if var in datasets[method].columns]
    
    for i, var in enumerate(continuous_vars):
        ax.bar(x + i*width, stds_data[var], width, 
              label=var, color=colors[i], alpha=0.8)
    
    ax.set_xlabel('Méthodes d\'imputation')
    ax.set_ylabel('Écart-type')
    ax.set_title('Comparaison des Écarts-types')
    ax.set_xticks(x + width)
    ax.set_xticklabels(method_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Impact sur la distribution de la target
    ax = axes[1, 0]
    target_ratios = []
    data_retention = []
    
    for method in method_names:
        df = datasets[method]
        
        # Ratio de déséquilibre
        if 'outcome' in df.columns:
            dist = df['outcome'].value_counts()
            ratio = dist.get(0, 0) / dist.get(1, 1) if dist.get(1, 1) > 0 else 0
            target_ratios.append(ratio)
        else:
            target_ratios.append(0)
        
        # Rétention des données
        retention = df.shape[0] / 2459  # Taille originale supposée
        data_retention.append(retention)
    
    bars = ax.bar(method_names, target_ratios, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'][:len(method_names)], alpha=0.8)
    ax.set_xlabel('Méthodes d\'imputation')
    ax.set_ylabel('Ratio de déséquilibre')
    ax.set_title('Impact sur le Déséquilibre des Classes')
    ax.set_xticklabels(method_names, rotation=45, ha='right')
    
    # Ajouter les valeurs sur les barres
    for bar, ratio in zip(bars, target_ratios):
        height = bar.get_height()
        ax.annotate(f'{ratio:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    ax.grid(True, alpha=0.3)
    
    # 4. Rétention des données
    ax = axes[1, 1]
    bars = ax.bar(method_names, [r*100 for r in data_retention], 
                  color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'][:len(method_names)], alpha=0.8)
    ax.set_xlabel('Méthodes d\'imputation')
    ax.set_ylabel('Rétention des données (%)')
    ax.set_title('Pourcentage de Données Conservées')
    ax.set_xticklabels(method_names, rotation=45, ha='right')
    
    # Ajouter les valeurs sur les barres
    for bar, retention in zip(bars, data_retention):
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Graphiques sauvegardés: {save_path}")
    else:
        plt.show()

print("✅ Fonctions de visualisation définies")


## 6.3 Comparaison des méthodes d'imputation - PARTIE 3/4

def statistical_tests_imputation(datasets_dict):
    """
    Effectue des tests statistiques pour comparer les méthodes d'imputation.
    """
    print("\n📈 TESTS STATISTIQUES COMPARATIFS")
    print("-" * 50)
    
    continuous_vars = ['X1_trans', 'X2_trans', 'X3_trans']
    methods = {}
    
    # Organiser les données par méthode
    for name, info in datasets_dict.items():
        if info['status'] == 'success':
            df = info['dataframe']
            method_base = 'KNN' if 'knn' in name else 'MICE'
            outlier_status = 'with_outliers' if 'with_outliers' in name else 'no_outliers'
            
            if method_base not in methods:
                methods[method_base] = {}
            methods[method_base][outlier_status] = df
    
    # Tests principaux
    results_summary = []
    
    print("🔬 Tests de Kolmogorov-Smirnov (différences de distribution):")
    print("-" * 55)
    
    for var in continuous_vars:
        print(f"\n📊 Variable: {var}")
        
        # 1. KNN vs MICE (avec outliers)
        if ('KNN' in methods and 'MICE' in methods and 
            'with_outliers' in methods['KNN'] and 'with_outliers' in methods['MICE']):
            
            data_knn = methods['KNN']['with_outliers'][var]
            data_mice = methods['MICE']['with_outliers'][var]
            
            ks_stat, ks_p = stats.ks_2samp(data_knn, data_mice)
            
            print(f"   KNN vs MICE (avec outliers):")
            print(f"     • KS statistic: {ks_stat:.4f}")
            print(f"     • p-value: {ks_p:.4f}")
            print(f"     • Différence significative: {'Oui' if ks_p < 0.05 else 'Non'}")
            
            results_summary.append({
                'Variable': var,
                'Comparison': 'KNN vs MICE (with outliers)',
                'KS_stat': ks_stat,
                'KS_p': ks_p,
                'Significant': ks_p < 0.05
            })
        
        # 2. Avec vs Sans outliers pour KNN
        if ('KNN' in methods and 
            'with_outliers' in methods['KNN'] and 'no_outliers' in methods['KNN']):
            
            data_with = methods['KNN']['with_outliers'][var]
            data_without = methods['KNN']['no_outliers'][var]
            
            ks_stat, ks_p = stats.ks_2samp(data_with, data_without)
            
            print(f"   KNN: Avec vs Sans outliers:")
            print(f"     • KS statistic: {ks_stat:.4f}")
            print(f"     • p-value: {ks_p:.4f}")
            print(f"     • Impact des outliers: {'Significatif' if ks_p < 0.05 else 'Non significatif'}")
            
            results_summary.append({
                'Variable': var,
                'Comparison': 'KNN: with vs no outliers',
                'KS_stat': ks_stat,
                'KS_p': ks_p,
                'Significant': ks_p < 0.05
            })
    
    return results_summary

def calculate_imputation_quality_scores(datasets_dict):
    """
    Calcule des scores de qualité pour chaque méthode d'imputation.
    """
    print("\n🏆 CALCUL DES SCORES DE QUALITÉ")
    print("-" * 40)
    
    scores = {}
    
    for name, info in datasets_dict.items():
        if info['status'] == 'success':
            df = info['dataframe']
            score = 0
            details = {}
            
            # 1. Complétude (20 points max)
            completeness = 1 - (df.isnull().sum().sum() / df.size)
            completeness_score = completeness * 20
            score += completeness_score
            details['Complétude'] = f"{completeness*100:.1f}% ({completeness_score:.1f}/20)"
            
            # 2. Rétention des données (25 points max)
            retention = df.shape[0] / 2459  # Taille originale
            retention_score = retention * 25
            score += retention_score
            details['Rétention'] = f"{retention*100:.1f}% ({retention_score:.1f}/25)"
            
            # 3. Préservation du déséquilibre (25 points max)
            if 'outcome' in df.columns:
                target_dist = df['outcome'].value_counts()
                current_ratio = target_dist.get(0, 0) / target_dist.get(1, 1)
                original_ratio = 6.15  # Ratio original
                
                # Score basé sur la proximité au ratio original
                ratio_diff = abs(current_ratio - original_ratio) / original_ratio
                balance_score = max(0, 25 * (1 - ratio_diff))
                score += balance_score
                details['Déséquilibre'] = f"{current_ratio:.2f}:1 ({balance_score:.1f}/25)"
            
            # 4. Stabilité des distributions (30 points max)
            if all(col in df.columns for col in ['X1_trans', 'X2_trans', 'X3_trans']):
                # Mesure de stabilité basée sur les coefficients de variation
                cvs = []
                for var in ['X1_trans', 'X2_trans', 'X3_trans']:
                    cv = df[var].std() / abs(df[var].mean()) if df[var].mean() != 0 else float('inf')
                    cvs.append(cv)
                
                avg_cv = np.mean(cvs)
                # Score inversement proportionnel à la variabilité
                stability_score = max(0, 30 * (1 - min(avg_cv, 1)))
                score += stability_score
                details['Stabilité'] = f"CV={avg_cv:.3f} ({stability_score:.1f}/30)"
            
            scores[name] = {
                'score_total': score,
                'details': details,
                'rank': 0  # Sera calculé après
            }
    
    # Calcul des rangs
    sorted_scores = sorted(scores.items(), key=lambda x: x[1]['score_total'], reverse=True)
    for rank, (name, score_info) in enumerate(sorted_scores, 1):
        scores[name]['rank'] = rank
    
    # Affichage des résultats
    print("📊 Scores de qualité par méthode:")
    print()
    
    for rank, (name, score_info) in enumerate(sorted_scores, 1):
        print(f"{rank}. {name.upper()}")
        print(f"   Score total: {score_info['score_total']:.1f}/100")
        for criterion, detail in score_info['details'].items():
            print(f"   • {criterion}: {detail}")
        print()
    
    return scores

print("✅ Fonctions d'analyse statistique définies")


## 6.3 Comparaison des méthodes d'imputation - PARTIE 4/4

def generate_final_recommendations(quality_scores, statistical_results):
    """
    Génère les recommandations finales basées sur toutes les analyses.
    """
    print("💡 RECOMMANDATIONS FINALES")
    print("=" * 50)
    
    # Trouver la meilleure méthode
    best_method = max(quality_scores.items(), key=lambda x: x[1]['score_total'])
    best_name, best_scores = best_method
    
    print(f"🏆 MÉTHODE RECOMMANDÉE: {best_name.upper()}")
    print(f"   Score de qualité: {best_scores['score_total']:.1f}/100")
    print()
    
    # Justification détaillée
    print("📋 JUSTIFICATION DU CHOIX:")
    print("-" * 30)
    
    if 'mice' in best_name.lower():
        print("🔬 MICE (Multiple Imputation by Chained Equations)")
        print("   ✅ Avantages:")
        print("     • Modélise les relations complexes entre variables")
        print("     • Idéal pour données MAR (Missing At Random)")
        print("     • Prend en compte l'incertitude de l'imputation")
        print("     • Robuste aux patterns de données manquantes")
    else:
        print("🔢 KNN (K-Nearest Neighbors)")
        print("   ✅ Avantages:")
        print("     • Préserve les relations de similarité locales")
        print("     • Simple et interprétable")
        print("     • Efficace computationnellement")
        print("     • Moins d'assumptions sur la distribution des données")
    
    if 'with_outliers' in best_name:
        print("\n📊 AVEC OUTLIERS:")
        print("   ✅ Avantages:")
        print("     • Préserve toute l'information disponible")
        print("     • Maximum de données pour l'entraînement")
        print("     • Conserve le déséquilibre original des classes")
        print("   ⚠️ Considérations:")
        print("     • Utiliser des modèles robustes aux outliers")
        print("     • Surveiller les performances avec validation croisée")
    else:
        print("\n🎯 SANS OUTLIERS:")
        print("   ✅ Avantages:")
        print("     • Données plus stables et cohérentes")
        print("     • Réduit le risque d'overfitting")
        print("     • Meilleur pour modèles sensibles aux outliers")
        print("   ⚠️ Considérations:")
        print("     • Perte d'informations (moins de données)")
        print("     • Déséquilibre accru des classes")
    
    # Recommandations pour la modélisation
    print(f"\n🚀 RECOMMANDATIONS POUR LA MODÉLISATION:")
    print("-" * 45)
    
    print("1. 📊 DONNÉES À UTILISER:")
    print(f"   • Dataset: {best_name}")
    print(f"   • Dimensions: Vérifiez dans GENERATED_DATASETS['{best_name}']['dataframe'].shape")
    
    print("\n2. 🤖 MODÈLES RECOMMANDÉS:")
    if 'with_outliers' in best_name:
        print("   • Random Forest (robuste aux outliers)")
        print("   • XGBoost avec régularisation")
        print("   • Gradient Boosting")
        print("   • SVM avec RBF kernel")
    else:
        print("   • Logistic Regression")
        print("   • Random Forest")
        print("   • Gradient Boosting")
        print("   • Réseaux de neurones")
    
    print("\n3. ⚖️ GESTION DU DÉSÉQUILIBRE:")
    target_ratio = None
    for detail in best_scores['details'].values():
        if ':1' in str(detail):
            target_ratio = detail.split('(')[0].strip()
            break
    
    if target_ratio:
        print(f"   • Ratio actuel: {target_ratio}")
        if '9.' in target_ratio or '8.' in target_ratio:
            print("   • FORTEMENT déséquilibré → Utiliser SMOTE ou ADASYN")
            print("   • class_weight='balanced' obligatoire")
            print("   • Optimiser pour F1-score, pas accuracy")
        else:
            print("   • Modérément déséquilibré → class_weight='balanced'")
            print("   • Validation croisée stratifiée")
    
    print("\n4. 📈 MÉTRIQUES DE VALIDATION:")
    print("   • F1-score (métrique principale)")
    print("   • Precision-Recall AUC")
    print("   • Matrice de confusion détaillée")
    print("   • Optimisation du seuil de classification")
    
    print("\n5. 🔄 VALIDATION:")
    print("   • StratifiedKFold avec 5 folds minimum")
    print("   • Hold-out test set de 20%")
    print("   • Validation croisée répétée si possible")
    
    return best_name

# ============================================================================
# EXÉCUTION COMPLÈTE DE L'ANALYSE
# ============================================================================

if 'GENERATED_DATASETS' in locals():
    print("🔄 LANCEMENT DE L'ANALYSE COMPARATIVE COMPLÈTE")
    print("="*60)
    
    # 1. Analyse comparative de base
    comparison_data = compare_imputation_methods(GENERATED_DATASETS)
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        print("\n📊 TABLEAU COMPARATIF RÉSUMÉ:")
        # Afficher seulement les colonnes principales
        key_columns = ['Dataset', 'Dimensions', 'Valeurs_manquantes', 'Target_0', 'Target_1', 'Ratio_desequilibre']
        print(comparison_df[key_columns].to_string(index=False))
    
    # 2. Visualisations
    print("\n📊 Génération des visualisations...")
    plot_distributions_comparison(
        GENERATED_DATASETS, 
        save_path="outputs/figures/distributions_comparison.png"
    )
    
    plot_statistics_comparison(
        GENERATED_DATASETS,
        save_path="outputs/figures/statistics_comparison.png"
    )
    
    # 3. Tests statistiques
    statistical_results = statistical_tests_imputation(GENERATED_DATASETS)
    
    # 4. Scores de qualité
    quality_scores = calculate_imputation_quality_scores(GENERATED_DATASETS)
    
    # 5. Recommandations finales
    recommended_method = generate_final_recommendations(quality_scores, statistical_results)
    
    print(f"\n✅ ANALYSE COMPARATIVE TERMINÉE!")
    print(f"📁 Graphiques sauvegardés dans: outputs/figures/")
    print(f"🏆 Méthode recommandée: {recommended_method}")
    print(f"📊 Utilisez: GENERATED_DATASETS['{recommended_method}']['dataframe']")

else:
    print("⚠️ Variable GENERATED_DATASETS non trouvée.")
    print("Assurez-vous d'avoir exécuté la section de génération des datasets.")


# Nouvelle fonction d'exécution complète
def run_imputation_comparison(datasets_dict, output_dir="outputs/figures"):
    """
    Exécute l'analyse comparative complète des méthodes d'imputation :
    - Analyse descriptive des datasets
    - Visualisations comparatives
    - Tests statistiques
    - Scores de qualité
    - Recommandations pour la modélisation
    
    Args:
        datasets_dict (dict): Dictionnaire contenant les datasets générés.
        output_dir (str or Path): Dossier de sauvegarde des figures.
    
    Returns:
        str: Nom de la méthode recommandée.
    """
    import pandas as pd
    from pathlib import Path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Analyse comparative
    comparison = compare_imputation_methods(datasets_dict)
    comparison_df = pd.DataFrame(comparison)
    print("\\n📊 TABLEAU COMPARATIF :")
    display(comparison_df[["Dataset", "Valeurs_manquantes", "Target_0", "Target_1", "Ratio_desequilibre"]])

    # Visualisations
    plot_distributions_comparison(datasets_dict, save_path=output_dir / "distributions_comparison.png")
    plot_statistics_comparison(datasets_dict, save_path=output_dir / "statistics_comparison.png")

    # Tests statistiques
    stats = statistical_tests_imputation(datasets_dict)

    # Scores et recommandations
    scores = calculate_imputation_quality_scores(datasets_dict)
    best_method = generate_final_recommendations(scores, stats)

    return best_method