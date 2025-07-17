"""
Package de configuration du projet STA211.

Importe les composants clés afin qu'ils soient accessibles directement :
    from modules.config import init_project, setup_project_paths, ProjectConfig
"""

from .env_setup import init_project
from .paths_config import setup_project_paths
from .project_config import ProjectConfig, create_config
from .display_config import set_display_options

__all__ = [
    "init_project",
    "setup_project_paths", 
    "ProjectConfig",
    "create_config",
    "set_display_options",
]