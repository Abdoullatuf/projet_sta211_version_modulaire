# modules/config/display_config.py
"""
Configuration complète de l'environnement STA211.
RÔLE : Configuration affichage, warnings, random seed, etc.

Utilisation :
    from modules.config import set_display_options
    set_display_options()
"""

from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
import logging
from sklearn.exceptions import ConvergenceWarning, FitFailedWarning

__all__ = ["set_display_options", "configure_warnings", "set_random_seed", "setup_environment"]

# ============================================================================
# CONSTANTES DE CONFIGURATION
# ============================================================================
RANDOM_STATE = 42

# ============================================================================
# CONFIGURATION PANDAS
# ============================================================================
def configure_pandas() -> None:
    """Configure les options d'affichage pandas."""
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", 120)
    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.precision", 4)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", 50)

# ============================================================================
# CONFIGURATION MATPLOTLIB
# ============================================================================
def configure_matplotlib() -> None:
    """Configure le thème matplotlib pour le projet."""
    plt.rcParams.update({
        # Taille des figures
        "figure.figsize": (10, 6),
        "figure.dpi": 100,
        
        # Polices et tailles
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "font.size": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        
        # Grille et style
        "grid.linestyle": "--",
        "grid.alpha": 0.4,
        "axes.grid": True,
        
        # Couleurs
        "axes.prop_cycle": plt.cycler(
            'color', ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        ),
        
        # Qualité
        "savefig.dpi": 300,
        "savefig.bbox": "tight"
    })

# ============================================================================
# CONFIGURATION WARNINGS
# ============================================================================
def configure_warnings() -> None:
    """Configure la suppression des warnings non critiques."""
    # Warnings généraux
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    
    # Warnings scikit-learn spécifiques
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=FitFailedWarning)
    
    # Suppression warnings pandas
    warnings.filterwarnings("ignore", message=".*DataFrame.iloc.*")
    warnings.filterwarnings("ignore", message=".*DataFrame.loc.*")
    
    # Logging level pour réduire le bruit
    logging.getLogger("sklearn").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

# ============================================================================
# CONFIGURATION RANDOM SEED
# ============================================================================
def set_random_seed(seed: int = RANDOM_STATE) -> None:
    """Configure les seeds pour la reproductibilité."""
    import random
    
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # Pour scikit-learn (via environnement)
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # TensorFlow/Keras (si disponible)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
    
    # PyTorch (si disponible)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

# ============================================================================
# FONCTION PRINCIPALE
# ============================================================================
def set_display_options(seed: int = RANDOM_STATE, verbose: bool = True) -> None:
    """
    Configure complètement l'environnement STA211.
    
    Args:
        seed: Graine pour la reproductibilité
        verbose: Affichage des messages de configuration
    """
    if verbose:
        print("🔧 Configuration environnement STA211...")
    
    # Configuration pandas
    configure_pandas()
    if verbose:
        print("   ✅ Pandas configuré")
    
    # Configuration matplotlib
    configure_matplotlib()
    if verbose:
        print("   ✅ Matplotlib configuré")
    
    # Configuration warnings
    configure_warnings()
    if verbose:
        print("   ✅ Warnings configurés")
    
    # Configuration random seed
    set_random_seed(seed)
    if verbose:
        print(f"   ✅ Random seed défini : {seed}")
    
    if verbose:
        print("🎯 Environnement prêt !")

# ============================================================================
# ALIAS POUR COMPATIBILITÉ
# ============================================================================
def setup_environment(seed: int = RANDOM_STATE, verbose: bool = True) -> None:
    """Alias pour set_display_options."""
    set_display_options(seed=seed, verbose=verbose)

# ============================================================================
# CONFIGURATION SPÉCIALISÉE NOTEBOOK
# ============================================================================
def setup_notebook_environment() -> None:
    """Configuration spécifique pour environnement Jupyter/Colab."""
    # Configuration standard
    set_display_options(verbose=False)
    
    # Configuration spécifique notebook
    try:
        from IPython.display import display, HTML
        # Améliorer l'affichage des DataFrames
        pd.set_option('display.notebook_repr_html', True)
        print("📱 Configuration notebook activée")
    except ImportError:
        print("⚠️ Environnement non-notebook détecté")

# ============================================================================
# UTILITAIRES DE DIAGNOSTIC
# ============================================================================
def print_environment_info() -> None:
    """Affiche les informations sur la configuration actuelle."""
    print("🔍 INFORMATIONS ENVIRONNEMENT")
    print("=" * 35)
    print(f"Pandas version: {pd.__version__}")
    print(f"Matplotlib version: {plt.matplotlib.__version__}")
    print(f"NumPy version: {np.__version__}")
    print(f"Random seed: {RANDOM_STATE}")
    print(f"Pandas max_columns: {pd.get_option('display.max_columns')}")
    print(f"Matplotlib figure size: {plt.rcParams['figure.figsize']}")

# Configuration automatique à l'import (peut être désactivée)
if __name__ != "__main__":
    # Auto-configuration silencieuse lors de l'import
    set_display_options(verbose=False) 