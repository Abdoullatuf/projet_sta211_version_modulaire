"""
Extensions du module d'optimisation du seuil pour supporter plusieurs modèles.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import logging

# Import des fonctions de base du module original
from .optimize_threshold import optimize_threshold, _convert_labels_to_numeric

log = logging.getLogger(__name__)

def optimize_multiple_models(models_dict, datasets_info, optimization_method="f1", cv=5, verbose=True):
    """
    Optimise le seuil pour plusieurs modèles sur plusieurs datasets.
    
    Parameters:
    -----------
    models_dict : dict
        Dictionnaire des modèles {nom: model_instance}
    datasets_info : list
        Liste des infos datasets [(X_train, X_test, y_train, y_test, dataset_name), ...]
    optimization_method : str
        Méthode d'optimisation ('f1', 'precision', 'recall')
    cv : int
        Nombre de folds pour validation croisée
    verbose : bool
        Affichage détaillé
        
    Returns:
    --------
    tuple : (optimization_results, comparison_df)
    """
    
    if verbose:
        log.info("🎯 Début optimisation du seuil pour TOUS les modèles")
        log.info(f"   Méthode: {optimization_method.upper()}")
        log.info(f"   Validation croisée: {cv} folds")
    
    optimization_results = {}
    performance_comparison = []
    
    for X_train, X_test, y_train, y_test, dataset_name in datasets_info:
        
        if verbose:
            log.info(f"\n📊 Optimisation pour dataset: {dataset_name}")
            print(f"\n{'='*60}")
            print(f"🎯 OPTIMISATION SEUIL - {dataset_name.upper()}")
            print(f"{'='*60}")
        
        dataset_results = {}
        
        for model_name, model in models_dict.items():
            if verbose:
                print(f"\n🔍 Modèle: {model_name}")
                print("-" * 40)
            
            try:
                # Entraînement du modèle
                model.fit(X_train, y_train)
                
                # Vérifier si le modèle supporte predict_proba
                if not hasattr(model, 'predict_proba'):
                    if verbose:
                        print(f"⚠️  {model_name} ne supporte pas predict_proba - seuil par défaut 0.5")
                    
                    # Évaluation avec seuil par défaut
                    y_pred_default = model.predict(X_test)
                    f1_default = f1_score(y_test, y_pred_default)
                    
                    dataset_results[model_name] = {
                        'optimal_threshold': 0.5,
                        'default_f1': f1_default,
                        'optimized_f1': f1_default,
                        'improvement': 0.0,
                        'supports_proba': False,
                        'predictions_default': y_pred_default,
                        'predictions_optimal': y_pred_default
                    }
                    
                    performance_comparison.append({
                        'model': model_name,
                        'dataset': dataset_name,
                        'default_f1': f1_default,
                        'optimized_f1': f1_default,
                        'improvement': 0.0,
                        'improvement_pct': 0.0,
                        'optimal_threshold': 0.5
                    })
                    
                    continue
                
                # Optimisation du seuil (version silencieuse)
                if verbose:
                    print("   Optimisation du seuil en cours...")
                
                # Créer une version temporaire sans print pour éviter le bruit
                temp_model = type(model)(**model.get_params())
                temp_model.fit(X_train, y_train)
                
                # Obtenir les probabilités et optimiser
                y_proba_train = temp_model.predict_proba(X_train)[:, 1]
                y_numeric = _convert_labels_to_numeric(y_train)
                
                # Optimisation silencieuse
                thresholds = np.linspace(0.01, 0.99, 99)
                scores = []
                
                for threshold in thresholds:
                    y_pred = (y_proba_train >= threshold).astype(int)
                    if optimization_method == "f1":
                        score = f1_score(y_numeric, y_pred, zero_division=0)
                    elif optimization_method == "precision":
                        from sklearn.metrics import precision_score
                        score = precision_score(y_numeric, y_pred, zero_division=0)
                    elif optimization_method == "recall":
                        from sklearn.metrics import recall_score
                        score = recall_score(y_numeric, y_pred, zero_division=0)
                    else:
                        score = f1_score(y_numeric, y_pred, zero_division=0)
                    scores.append(score)
                
                best_idx = np.argmax(scores)
                optimal_threshold = thresholds[best_idx]
                
                # Évaluation avec seuil par défaut (0.5)
                y_proba_test = model.predict_proba(X_test)[:, 1]
                y_pred_default = (y_proba_test >= 0.5).astype(int)
                f1_default = f1_score(y_test, y_pred_default)
                
                # Évaluation avec seuil optimal
                y_pred_optimal = (y_proba_test >= optimal_threshold).astype(int)
                f1_optimal = f1_score(y_test, y_pred_optimal)
                
                # Calcul de l'amélioration
                improvement = f1_optimal - f1_default
                improvement_pct = (improvement / f1_default * 100) if f1_default > 0 else 0
                
                # Stockage des résultats
                dataset_results[model_name] = {
                    'optimal_threshold': optimal_threshold,
                    'default_f1': f1_default,
                    'optimized_f1': f1_optimal,
                    'improvement': improvement,
                    'improvement_pct': improvement_pct,
                    'supports_proba': True,
                    'predictions_default': y_pred_default,
                    'predictions_optimal': y_pred_optimal,
                    'probabilities': y_proba_test
                }
                
                performance_comparison.append({
                    'model': model_name,
                    'dataset': dataset_name,
                    'default_f1': f1_default,
                    'optimized_f1': f1_optimal,
                    'improvement': improvement,
                    'improvement_pct': improvement_pct,
                    'optimal_threshold': optimal_threshold
                })
                
                # Affichage des résultats
                if verbose:
                    print(f"   Seuil par défaut (0.5): F1 = {f1_default:.4f}")
                    print(f"   Seuil optimal ({optimal_threshold:.3f}): F1 = {f1_optimal:.4f}")
                    if improvement > 0:
                        print(f"   🚀 Amélioration: +{improvement:.4f} ({improvement_pct:+.1f}%)")
                    else:
                        print(f"   📊 Variation: {improvement:+.4f} ({improvement_pct:+.1f}%)")
                
            except Exception as e:
                if verbose:
                    print(f"❌ Erreur avec {model_name}: {e}")
                dataset_results[model_name] = {
                    'error': str(e),
                    'optimal_threshold': 0.5,
                    'default_f1': 0.0,
                    'optimized_f1': 0.0,
                    'improvement': 0.0
                }
        
        optimization_results[dataset_name] = dataset_results
    
    return optimization_results, pd.DataFrame(performance_comparison)

def analyze_optimization_results(comparison_df, verbose=True):
    """
    Analyse les résultats d'optimisation de multiple modèles.
    
    Parameters:
    -----------
    comparison_df : pd.DataFrame
        DataFrame avec les résultats de comparaison
    verbose : bool
        Affichage détaillé
        
    Returns:
    --------
    pd.DataFrame : DataFrame analysé avec statistiques
    """
    
    if verbose:
        print(f"\n📊 ANALYSE DES RÉSULTATS D'OPTIMISATION")
        print(f"{'='*50}")
        
        # 1. Résumé global
        total_models = len(comparison_df)
        improved_models = len(comparison_df[comparison_df['improvement'] > 0])
        degraded_models = len(comparison_df[comparison_df['improvement'] < 0])
        unchanged_models = len(comparison_df[comparison_df['improvement'] == 0])
        
        print(f"\n🎯 Résumé global:")
        print(f"   Total évaluations: {total_models}")
        print(f"   Améliorés: {improved_models} ({improved_models/total_models*100:.1f}%)")
        print(f"   Dégradés: {degraded_models} ({degraded_models/total_models*100:.1f}%)")
        print(f"   Inchangés: {unchanged_models} ({unchanged_models/total_models*100:.1f}%)")
        
        # 2. Top améliorations
        print(f"\n🚀 Top 5 améliorations:")
        top_improvements = comparison_df.nlargest(5, 'improvement')
        for _, row in top_improvements.iterrows():
            print(f"   {row['model']} ({row['dataset']}): +{row['improvement']:.4f} ({row['improvement_pct']:+.1f}%)")
        
        # 3. Moyennes par dataset
        print(f"\n📈 Amélioration moyenne par dataset:")
        dataset_stats = comparison_df.groupby('dataset').agg({
            'improvement': ['mean', 'std', 'count'],
            'improvement_pct': 'mean'
        }).round(4)
        print(dataset_stats)
        
        # 4. Moyennes par modèle
        print(f"\n🤖 Amélioration moyenne par modèle:")
        model_stats = comparison_df.groupby('model').agg({
            'improvement': ['mean', 'std'],
            'improvement_pct': 'mean',
            'optimal_threshold': 'mean'
        }).round(4)
        print(model_stats)
    
    return comparison_df

def plot_optimization_comparison(comparison_df, save_path=None):
    """
    Visualise les résultats de comparaison d'optimisation pour tous les modèles.
    
    Parameters:
    -----------
    comparison_df : pd.DataFrame
        DataFrame avec les résultats de comparaison
    save_path : str or Path, optional
        Chemin pour sauvegarder le graphique
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('🎯 Analyse Optimisation Seuil - Tous Modèles', fontsize=16, fontweight='bold')
    
    # 1. Amélioration par modèle
    ax1 = axes[0, 0]
    model_improvements = comparison_df.groupby('model')['improvement'].mean().sort_values(ascending=True)
    colors = ['red' if x < 0 else 'green' if x > 0 else 'gray' for x in model_improvements]
    model_improvements.plot(kind='barh', ax=ax1, color=colors)
    ax1.set_title('📊 Amélioration F1 moyenne par modèle')
    ax1.set_xlabel('Amélioration F1-score')
    ax1.axvline(x=0, color='black', linestyle='--', alpha=0.7)
    
    # 2. Distribution des seuils optimaux
    ax2 = axes[0, 1]
    thresholds = comparison_df['optimal_threshold']
    ax2.hist(thresholds, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Seuil par défaut (0.5)')
    ax2.set_title('📈 Distribution des seuils optimaux')
    ax2.set_xlabel('Seuil optimal')
    ax2.set_ylabel('Fréquence')
    ax2.legend()
    
    # 3. Comparaison avant/après par dataset
    ax3 = axes[1, 0]
    datasets = comparison_df['dataset'].unique()
    x = np.arange(len(datasets))
    width = 0.35
    
    default_means = [comparison_df[comparison_df['dataset'] == d]['default_f1'].mean() for d in datasets]
    optimized_means = [comparison_df[comparison_df['dataset'] == d]['optimized_f1'].mean() for d in datasets]
    
    ax3.bar(x - width/2, default_means, width, label='Seuil 0.5', alpha=0.7, color='lightcoral')
    ax3.bar(x + width/2, optimized_means, width, label='Seuil optimal', alpha=0.7, color='lightgreen')
    
    ax3.set_title('📊 F1-score moyen: Avant vs Après optimisation')
    ax3.set_xlabel('Dataset')
    ax3.set_ylabel('F1-score moyen')
    ax3.set_xticks(x)
    ax3.set_xticklabels(datasets, rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Heatmap améliorations
    ax4 = axes[1, 1]
    pivot_data = comparison_df.pivot(index='model', columns='dataset', values='improvement_pct')
    sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='RdYlGn', center=0, 
                ax=ax4, cbar_kws={'label': 'Amélioration (%)'})
    ax4.set_title('🔥 Heatmap améliorations (%)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Graphique sauvegardé: {save_path}")
    
    plt.show()

def save_optimization_results(optimization_results, comparison_df, save_dir):
    """
    Sauvegarde les résultats d'optimisation.
    
    Parameters:
    -----------
    optimization_results : dict
        Résultats détaillés d'optimisation
    comparison_df : pd.DataFrame
        DataFrame de comparaison
    save_dir : str or Path
        Répertoire de sauvegarde
        
    Returns:
    --------
    tuple : (json_file, csv_file) chemins des fichiers sauvegardés
    """
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Format pour sauvegarde
    thresholds_summary = {}
    
    for dataset_name, dataset_results in optimization_results.items():
        thresholds_summary[dataset_name] = {}
        
        for model_name, model_result in dataset_results.items():
            if 'optimal_threshold' in model_result:
                thresholds_summary[dataset_name][model_name] = {
                    'optimal_threshold': float(model_result['optimal_threshold']),
                    'default_f1': float(model_result.get('default_f1', 0)),
                    'optimized_f1': float(model_result.get('optimized_f1', 0)),
                    'improvement': float(model_result.get('improvement', 0)),
                    'supports_proba': model_result.get('supports_proba', False)
                }
    
    # Sauvegarde JSON
    json_file = save_dir / "optimal_thresholds_all_models.json"
    with open(json_file, 'w') as f:
        json.dump(thresholds_summary, f, indent=2)
    
    # Sauvegarde CSV du DataFrame de comparaison
    csv_file = save_dir / "threshold_optimization_comparison.csv"
    comparison_df.to_csv(csv_file, index=False)
    
    print(f"\n💾 Seuils optimaux sauvegardés:")
    print(f"   JSON: {json_file}")
    print(f"   CSV:  {csv_file}")
    
    return json_file, csv_file

def identify_best_model(comparison_df, verbose=True):
    """
    Identifie le meilleur modèle après optimisation.
    
    Parameters:
    -----------
    comparison_df : pd.DataFrame
        DataFrame avec les résultats de comparaison
    verbose : bool
        Affichage détaillé
        
    Returns:
    --------
    pd.Series : Informations du meilleur modèle global
    """
    
    if verbose:
        print(f"\n🏆 IDENTIFICATION DU MEILLEUR MODÈLE")
        print(f"{'='*45}")
        
        # Meilleur par dataset
        for dataset in comparison_df['dataset'].unique():
            dataset_df = comparison_df[comparison_df['dataset'] == dataset]
            best_model = dataset_df.loc[dataset_df['optimized_f1'].idxmax()]
            
            print(f"\n📊 Dataset: {dataset}")
            print(f"   🥇 Meilleur modèle: {best_model['model']}")
            print(f"   📈 F1-score optimisé: {best_model['optimized_f1']:.4f}")
            print(f"   🎯 Seuil optimal: {best_model['optimal_threshold']:.3f}")
            print(f"   🚀 Amélioration: +{best_model['improvement']:.4f}")
        
        # Meilleur global
        best_overall = comparison_df.loc[comparison_df['optimized_f1'].idxmax()]
        print(f"\n🏆 CHAMPION GLOBAL:")
        print(f"   🥇 Modèle: {best_overall['model']}")
        print(f"   📊 Dataset: {best_overall['dataset']}")
        print(f"   📈 F1-score: {best_overall['optimized_f1']:.4f}")
        print(f"   🎯 Seuil: {best_overall['optimal_threshold']:.3f}")
    else:
        best_overall = comparison_df.loc[comparison_df['optimized_f1'].idxmax()]
    
    return best_overall 

