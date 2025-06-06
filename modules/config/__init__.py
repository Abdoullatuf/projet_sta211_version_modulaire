# config/__init__.py
"""
Module de configuration pour le projet STA211.
"""
from .paths_config import setup_project_paths, is_colab
from .project_config import ProjectConfig, create_config

__all__ = [
    'setup_project_paths',
    'is_colab',
    'ProjectConfig',
    'create_config'
]