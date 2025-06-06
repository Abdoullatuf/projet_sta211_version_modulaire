# utils/__init__.py
"""
Module utilitaire pour le projet STA211.
"""

# Import de tous les éléments depuis imports.py
from .imports import *

# Import des helpers
from .helpers import (
    print_shape_change,
    check_gpu_availability,
    create_directories,
    timer,
    memory_usage,
    set_random_seed,
    cache_result,
    format_number,
    get_memory_usage_df
)

# Import du logger
from .logger import setup_logging, get_logger, logger

__all__ = [
    # Helpers
    'print_shape_change',
    'check_gpu_availability',
    'create_directories',
    'timer',
    'memory_usage',
    'set_random_seed',
    'cache_result',
    'format_number',
    'get_memory_usage_df',
    
    # Logger
    'setup_logging',
    'get_logger',
    'logger',
    
    # Constantes importantes
    'RANDOM_STATE',
    'XGB_AVAILABLE',
    'LGB_AVAILABLE',
    'PLOTLY_AVAILABLE'
]