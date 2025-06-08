# ============================================================================
# 5. modules/config/__init__.py
# ============================================================================

"""
Module de configuration pour STA211.

Contient :
- Configuration des chemins du projet
- Configuration des paramètres de preprocessing
- Configuration des modèles et pipelines
"""

from .paths_config import setup_project_paths, is_colab
from .project_config import ProjectConfig, create_config
from .setup import quick_setup, silent_setup
from .config_setup import setup_project_configuration, quick_project_config

__all__ = [
    'setup_project_paths',
    'is_colab',
    'ProjectConfig',
    'create_config',
    'quick_setup',
    'silent_setup',
    'setup_project_configuration', 
    'quick_project_config'          
]

