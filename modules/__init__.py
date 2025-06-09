# ============================================================================
# 1. modules/__init__.py (racine des modules)
# ============================================================================

"""
Modules du projet STA211 - Internet Advertisements Classification

Ce package contient tous les modules personnalisés du projet :
- preprocessing : Modules de prétraitement des données
- validation : Modules de diagnostic et validation
- exploration : Modules d'analyse exploratoire
- config : Modules de configuration
"""

__version__ = "1.0"
__author__ = "Abdoullatuf"
__project__ = "STA211 - Internet Advertisements Classification"

# Imports principaux pour faciliter l'utilisation
from .config.project_config import ProjectConfig, create_config
from .config.paths_config import setup_project_paths

# Métadonnées du package
__all__ = [
    'preprocessing',
    'validation', 
    'exploration',
    'config'
]