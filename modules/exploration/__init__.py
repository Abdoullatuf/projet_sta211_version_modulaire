# ============================================================================
# 4. modules/exploration/__init__.py
# ============================================================================

"""
Module d'exploration et analyse des données pour STA211.

Contient les outils pour :
- Analyse exploratoire des données (EDA)
- Visualisations statistiques
- Analyse des corrélations
- Statistiques descriptives
"""

from .target_analyzer import analyze_target_complete, update_config_with_target_stats

# Imports conditionnels selon ce que vous avez déjà
try:
    from .eda_analysis import (
        full_correlation_analysis
    )
except ImportError:
    pass

try:
    from .statistics import (
        analyze_continuous_variables,
        optimized_feature_importance
    )
except ImportError:
    pass

try:
    from .visualization import (
        visualize_distributions_and_boxplots,
        compare_visualization_methods,
        plot_continuous_by_class,
        plot_binary_sparsity,
        plot_continuous_target_corr,
        plot_eda_summary,
        save_fig
    )
except ImportError:
    pass

# Exports dynamiques selon les modules disponibles
__all__ = []

# Vérification des modules disponibles
import importlib.util

if importlib.util.find_spec('.eda_analysis', package=__name__):
    __all__.extend(['full_correlation_analysis'])

if importlib.util.find_spec('.statistics', package=__name__):
    __all__.extend(['analyze_continuous_variables', 'optimized_feature_importance'])

if importlib.util.find_spec('.visualization', package=__name__):
    __all__.extend([
        'visualize_distributions_and_boxplots',
        'compare_visualization_methods',
        'plot_continuous_by_class',
        'plot_binary_sparsity', 
        'plot_continuous_target_corr',
        'plot_eda_summary',
        'save_fig',
        analyze_target_complete,
        update_config_with_target_stats
    ])