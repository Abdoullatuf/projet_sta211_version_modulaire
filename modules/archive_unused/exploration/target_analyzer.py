# modules/exploration/target_analyzer.py

"""
Module d'analyse de la variable cible pour STA211.
Analyse la distribution, le déséquilibre et génère des recommandations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Optional

def analyze_target_distribution(df: pd.DataFrame, target_col: str = 'y') -> Dict[str, Any]:
    """
    Analyse complète de la distribution de la variable cible.
    
    Args:
        df: DataFrame contenant la variable cible
        target_col: Nom de la variable cible
        
    Returns:
        Dict: Statistiques complètes de la distribution
    """
    
    if target_col not in df.columns:
        raise ValueError(f"Colonne '{target_col}' introuvable dans le DataFrame")
    
    # Statistiques de base
    target_counts = df[target_col].value_counts().sort_index()
    target_pct = df[target_col].value_counts(normalize=True).sort_index() * 100
    total_samples = len(df)
    missing_values = df[target_col].isnull().sum()
    
    # Calculs de déséquilibre
    majority_class = target_counts.idxmax()
    minority_class = target_counts.idxmin()
    imbalance_ratio = target_counts[majority_class] / target_counts[minority_class]
    
    # Métriques pour la modélisation
    minority_prop = target_counts[minority_class] / total_samples
    majority_prop = target_counts[majority_class] / total_samples
    
    # F1-score baseline (stratégie naive)
    baseline_f1 = 2 * minority_prop / (1 + minority_prop)
    
    return {
        'counts': dict(target_counts),
        'percentages': dict(target_pct),
        'total_samples': total_samples,
        'missing_values': missing_values,
        'majority_class': majority_class,
        'minority_class': minority_class,
        'imbalance_ratio': imbalance_ratio,
        'minority_proportion': minority_prop,
        'majority_proportion': majority_prop,
        'baseline_f1': baseline_f1
    }

def create_target_visualizations(stats: Dict[str, Any], save_path: Optional[Path] = None, 
                               show: bool = True) -> plt.Figure:
    """
    Crée des visualisations de la distribution de la variable cible.
    
    Args:
        stats: Statistiques de distribution
        save_path: Chemin de sauvegarde (optionnel)
        show: Afficher le graphique
        
    Returns:
        Figure matplotlib
    """
    
    # Configuration des couleurs
    colors = ['#3498db', '#e74c3c']  # Bleu pour 0, Rouge pour 1
    
    # Création de la figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # 1. Bar plot avec annotations
    counts = pd.Series(stats['counts'])
    bars = counts.plot(kind='bar', ax=ax1, color=colors, width=0.6)
    ax1.set_title('Distribution des classes', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Classe')
    ax1.set_ylabel('Nombre d\'échantillons')
    ax1.set_xticklabels(['Non-publicité (0)', 'Publicité (1)'], rotation=0)
    
    # Annotations sur les barres
    for i, (idx, value) in enumerate(counts.items()):
        ax1.annotate(f'{value:,}', (i, value), ha='center', va='bottom', fontweight='bold')
    
    # 2. Pie chart
    percentages = pd.Series(stats['percentages'])
    wedges, texts, autotexts = ax2.pie(
        percentages.values, 
        labels=['Non-publicité', 'Publicité'],
        colors=colors,
        autopct='%1.1f%%', 
        startangle=90,
        explode=(0.05, 0.05)
    )
    ax2.set_title('Proportion des classes', fontsize=14, fontweight='bold')
    
    # Style des textes
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    plt.tight_layout()
    
    # Sauvegarde
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    if show:
        plt.show()
    
    return fig

def print_distribution_analysis(stats: Dict[str, Any], target_col: str = 'y') -> None:
    """
    Affiche l'analyse de distribution de manière formatée.
    
    Args:
        stats: Statistiques de distribution
        target_col: Nom de la variable cible
    """
    
    print(f"📊 Distribution de la variable cible '{target_col}' :")
    
    for label in sorted(stats['counts'].keys()):
        class_name = "ad." if label == 1 else "noad."
        count = stats['counts'][label]
        pct = stats['percentages'][label]
        status = "🔴 Minoritaire" if label == stats['minority_class'] else "🔵 Majoritaire"
        print(f"  • Classe {label} ({class_name:<6}): {count:>4,} ({pct:>5.1f}%) {status}")
    
    print(f"\n📈 Métriques de déséquilibre :")
    print(f"  • Ratio de déséquilibre  : {stats['imbalance_ratio']:.2f}:1")
    print(f"  • Échantillons totaux    : {stats['total_samples']:,}")
    print(f"  • F1-score baseline      : {stats['baseline_f1']:.3f}")
    
    if stats['missing_values'] > 0:
        print(f"  ⚠️ {stats['missing_values']} valeurs manquantes détectées")

def generate_modeling_recommendations(stats: Dict[str, Any]) -> None:
    """
    Génère des recommandations pour la modélisation.
    
    Args:
        stats: Statistiques de distribution
    """
    
    ratio = stats['imbalance_ratio']
    baseline_f1 = stats['baseline_f1']
    
    print(f"\n💡 RECOMMANDATIONS POUR LA MODÉLISATION")
    print("=" * 45)
    
    # Classification du niveau de déséquilibre
    if ratio < 2:
        severity = "🟢 Léger"
    elif ratio < 5:
        severity = "🟡 Modéré"
    elif ratio < 10:
        severity = "🟠 Sévère"
    else:
        severity = "🔴 Critique"
    
    print(f"📊 Niveau de déséquilibre : {severity} ({ratio:.1f}:1)")
    
    # Stratégies recommandées
    print(f"\n🎯 Stratégies recommandées :")
    
    strategies = [
        "✅ Stratify=True pour train/validation split",
        "✅ Optimiser pour F1-score",
        "✅ Validation croisée stratifiée"
    ]
    
    if ratio >= 3:
        strategies.extend([
            "🔄 SMOTE ou class_weight='balanced'",
            "📊 Métriques robustes (Precision-Recall AUC)",
            "🎛️ Ajustement du seuil de classification"
        ])
    
    if ratio >= 6:
        strategies.extend([
            "🏗️ Modèles robustes (XGBoost, Random Forest)",
            "🎭 Techniques d'ensemble"
        ])
    
    for strategy in strategies:
        print(f"  {strategy}")
    
    # Seuils de performance
    print(f"\n📈 Seuils de performance :")
    print(f"  • F1-score baseline      : {baseline_f1:.3f}")
    print(f"  • F1-score cible minimum : {baseline_f1 + 0.1:.3f}")
    print(f"  • F1-score excellent     : 0.700+")

def analyze_target_complete(df: pd.DataFrame, target_col: str = 'y', 
                          figures_dir: Optional[Path] = None, 
                          verbose: bool = True) -> Dict[str, Any]:
    """
    Analyse complète de la variable cible avec visualisations.
    
    Args:
        df: DataFrame contenant la variable cible
        target_col: Nom de la variable cible
        figures_dir: Répertoire de sauvegarde des figures
        verbose: Affichage détaillé
        
    Returns:
        Dict: Statistiques complètes
    """
    
    if verbose:
        print("🎯 Analyse complète de la variable cible")
        print("=" * 45)
    
    # Analyse statistique
    stats = analyze_target_distribution(df, target_col)
    
    if verbose:
        print_distribution_analysis(stats, target_col)
    
    # Visualisations
    if figures_dir:
        eda_dir = Path(figures_dir) / 'eda'
        eda_dir.mkdir(parents=True, exist_ok=True)
        save_path = eda_dir / 'target_distribution.png'
    else:
        save_path = None
    
    fig = create_target_visualizations(stats, save_path=save_path, show=verbose)
    
    if save_path and verbose:
        print(f"\n💾 Graphique sauvegardé : {save_path}")
    
    # Recommandations
    if verbose:
        generate_modeling_recommendations(stats)
    
    return stats

def update_config_with_target_stats(config, stats: Dict[str, Any]) -> bool:
    """
    Met à jour la configuration avec les statistiques de la cible.
    
    Args:
        config: Objet de configuration
        stats: Statistiques de distribution
        
    Returns:
        bool: Succès de la mise à jour
    """
    
    try:
        config.update("PROJECT_CONFIG.CLASS_DISTRIBUTION", stats['counts'])
        config.update("PROJECT_CONFIG.IMBALANCE_RATIO_CALCULATED", stats['imbalance_ratio'])
        config.update("PROJECT_CONFIG.BASELINE_F1_SCORE", stats['baseline_f1'])
        config.update("PROJECT_CONFIG.MINORITY_CLASS_PROPORTION", stats['minority_proportion'])
        
        return True
        
    except Exception as e:
        print(f"⚠️ Erreur mise à jour configuration : {e}")
        return False