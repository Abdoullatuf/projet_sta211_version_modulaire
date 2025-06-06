# utils/helpers.py
"""
Fonctions utilitaires pour le projet STA211.
"""
import time
import psutil
import functools
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Callable, Any
from contextlib import contextmanager

def print_shape_change(df_before: pd.DataFrame, df_after: pd.DataFrame, operation: str) -> None:
    """
    Affiche le changement de dimensions après une opération.
    
    Parameters:
    -----------
    df_before : pd.DataFrame
        DataFrame avant l'opération
    df_after : pd.DataFrame
        DataFrame après l'opération
    operation : str
        Description de l'opération
    """
    print(f"\n{operation}:")
    print(f"  Avant : {df_before.shape}")
    print(f"  Après : {df_after.shape}")
    print(f"  Lignes supprimées : {df_before.shape[0] - df_after.shape[0]}")
    print(f"  Colonnes supprimées : {df_before.shape[1] - df_after.shape[1]}")

def check_gpu_availability() -> None:
    """Vérifie la disponibilité du GPU pour XGBoost et affiche les informations."""
    try:
        import xgboost as xgb
        # Vérifier si XGBoost peut utiliser le GPU
        try:
            # Test simple pour voir si GPU est disponible
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                print(f"✅ GPU détecté : {gpus[0].name}")
                print(f"   Mémoire : {gpus[0].memoryTotal}MB")
                print(f"   Utilisation : {gpus[0].load*100:.1f}%")
            else:
                print("⚠️ Aucun GPU détecté")
        except ImportError:
            print("ℹ️ GPUtil non installé - impossible de vérifier le GPU")
            print("   Installer avec : pip install gputil")
    except ImportError:
        print("⚠️ XGBoost non installé")

def create_directories(paths: dict) -> None:
    """
    Crée les répertoires s'ils n'existent pas.
    
    Parameters:
    -----------
    paths : dict
        Dictionnaire des chemins à créer
    """
    for name, path in paths.items():
        path = Path(path)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            print(f"✅ Créé : {name} -> {path}")
        else:
            print(f"✓ Existe : {name} -> {path}")

@contextmanager
def timer(description: str = "Opération"):
    """
    Context manager pour mesurer le temps d'exécution.
    
    Usage:
    ------
    with timer("Entraînement du modèle"):
        model.fit(X, y)
    """
    start = time.time()
    print(f"\n⏱️ Début : {description}")
    try:
        yield
    finally:
        elapsed = time.time() - start
        print(f"✅ Terminé : {description} - Durée : {elapsed:.2f}s")

def memory_usage():
    """Affiche l'utilisation actuelle de la mémoire."""
    process = psutil.Process()
    mem_info = process.memory_info()
    print(f"\n💾 Utilisation mémoire :")
    print(f"   RSS : {mem_info.rss / 1024 / 1024:.1f} MB")
    print(f"   VMS : {mem_info.vms / 1024 / 1024:.1f} MB")
    print(f"   Pourcentage : {process.memory_percent():.1f}%")

def set_random_seed(seed: int = 42) -> None:
    """
    Définit la seed pour tous les générateurs aléatoires.
    
    Parameters:
    -----------
    seed : int
        Valeur de la seed
    """
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
    
    print(f"🎲 Random seed définie à : {seed}")

def cache_result(func: Callable) -> Callable:
    """
    Décorateur pour mettre en cache les résultats d'une fonction.
    
    Usage:
    ------
    @cache_result
    def expensive_computation(x):
        return x ** 2
    """
    cache = {}
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]
    
    return wrapper

def format_number(num: float, precision: int = 2) -> str:
    """
    Formate un nombre pour l'affichage.
    
    Parameters:
    -----------
    num : float
        Nombre à formater
    precision : int
        Nombre de décimales
        
    Returns:
    --------
    str : Nombre formaté
    """
    if abs(num) >= 1e6:
        return f"{num/1e6:.{precision}f}M"
    elif abs(num) >= 1e3:
        return f"{num/1e3:.{precision}f}K"
    else:
        return f"{num:.{precision}f}"

def get_memory_usage_df(df: pd.DataFrame) -> None:
    """
    Affiche l'utilisation mémoire d'un DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame à analyser
    """
    memory_usage = df.memory_usage(deep=True)
    total_memory = memory_usage.sum() / 1024 / 1024  # En MB
    
    print(f"\n💾 Utilisation mémoire du DataFrame :")
    print(f"   Shape : {df.shape}")
    print(f"   Total : {total_memory:.2f} MB")
    print(f"   Par colonne :")
    
    for col, mem in memory_usage.items():
        if col != 'Index':
            print(f"     - {col}: {mem/1024/1024:.2f} MB")