# modules/config/utils.py
"""
Fonctions utilitaires pour le projet STA211.
RÔLE : Helper functions et outils génériques.

Utilisation :
    from modules.config.utils import print_shape_change
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import time
from pathlib import Path

__all__ = [
    "print_shape_change", 
    "compare_dataframes", 
    "memory_usage", 
    "timer",
    "save_results",
    "load_results"
]

# ============================================================================
# UTILITAIRES D'AFFICHAGE
# ============================================================================

def print_shape_change(df_before: pd.DataFrame, df_after: pd.DataFrame, operation: str) -> None:
    """Affiche le changement de dimensions après une opération de pré-traitement."""
    rows_removed = df_before.shape[0] - df_after.shape[0]
    cols_removed = df_before.shape[1] - df_after.shape[1]
    
    print(f"\n📊 {operation}:")
    print(f"   Avant : {df_before.shape}")
    print(f"   Après : {df_after.shape}")
    print(f"   Δ Lignes : {rows_removed:+d} ({rows_removed/df_before.shape[0]*100:+.1f}%)")
    if cols_removed != 0:
        print(f"   Δ Colonnes : {cols_removed:+d}")

def compare_dataframes(df1: pd.DataFrame, df2: pd.DataFrame, 
                      name1: str = "DataFrame 1", name2: str = "DataFrame 2") -> None:
    """Compare deux DataFrames et affiche les différences."""
    print(f"\n🔍 COMPARAISON : {name1} vs {name2}")
    print("=" * 50)
    
    # Dimensions
    print(f"📏 Dimensions:")
    print(f"   {name1}: {df1.shape}")
    print(f"   {name2}: {df2.shape}")
    
    # Colonnes
    cols1, cols2 = set(df1.columns), set(df2.columns)
    common_cols = cols1.intersection(cols2)
    only_df1 = cols1 - cols2
    only_df2 = cols2 - cols1
    
    print(f"\n📋 Colonnes:")
    print(f"   Communes: {len(common_cols)}")
    if only_df1:
        print(f"   Uniquement {name1}: {len(only_df1)} - {list(only_df1)[:5]}{'...' if len(only_df1) > 5 else ''}")
    if only_df2:
        print(f"   Uniquement {name2}: {len(only_df2)} - {list(only_df2)[:5]}{'...' if len(only_df2) > 5 else ''}")
    
    # Valeurs manquantes
    if not df1.empty and not df2.empty:
        na1, na2 = df1.isnull().sum().sum(), df2.isnull().sum().sum()
        print(f"\n❓ Valeurs manquantes:")
        print(f"   {name1}: {na1}")
        print(f"   {name2}: {na2}")

# ============================================================================
# UTILITAIRES DE PERFORMANCE
# ============================================================================

def memory_usage(df: pd.DataFrame, detailed: bool = False) -> Dict[str, Any]:
    """Analyse l'usage mémoire d'un DataFrame."""
    memory_mb = df.memory_usage(deep=True).sum() / 1024**2
    
    result = {
        'memory_mb': round(memory_mb, 2),
        'memory_per_row_kb': round(memory_mb * 1024 / len(df), 2) if len(df) > 0 else 0,
        'shape': df.shape
    }
    
    if detailed:
        result['dtypes'] = df.dtypes.value_counts().to_dict()
        result['null_counts'] = df.isnull().sum().sum()
    
    return result

class timer:
    """Context manager pour mesurer le temps d'exécution."""
    
    def __init__(self, operation: str = "Opération"):
        self.operation = operation
        self.start_time = None
    
    def __enter__(self):
        print(f"⏱️ Début : {self.operation}...")
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        duration = time.time() - self.start_time
        print(f"✅ Terminé : {self.operation} en {duration:.2f}s")

# ============================================================================
# UTILITAIRES DE SAUVEGARDE
# ============================================================================

def save_results(data: Any, filename: str, output_dir: str = "outputs") -> Path:
    """Sauvegarde des résultats avec gestion automatique du format."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    file_path = output_path / filename
    
    if isinstance(data, pd.DataFrame):
        if filename.endswith('.csv'):
            data.to_csv(file_path, index=False)
        elif filename.endswith('.parquet'):
            data.to_parquet(file_path)
        else:
            data.to_csv(file_path.with_suffix('.csv'), index=False)
            file_path = file_path.with_suffix('.csv')
    
    elif isinstance(data, dict):
        import json
        if not filename.endswith('.json'):
            file_path = file_path.with_suffix('.json')
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    else:
        import joblib
        if not filename.endswith('.pkl'):
            file_path = file_path.with_suffix('.pkl')
        joblib.dump(data, file_path)
    
    print(f"💾 Sauvegardé : {file_path}")
    return file_path

def load_results(filename: str, input_dir: str = "outputs") -> Any:
    """Charge des résultats avec détection automatique du format."""
    file_path = Path(input_dir) / filename
    
    if not file_path.exists():
        raise FileNotFoundError(f"Fichier non trouvé : {file_path}")
    
    if filename.endswith('.csv'):
        return pd.read_csv(file_path)
    elif filename.endswith('.parquet'):
        return pd.read_parquet(file_path)
    elif filename.endswith('.json'):
        import json
        with open(file_path, 'r') as f:
            return json.load(f)
    elif filename.endswith('.pkl'):
        import joblib
        return joblib.load(file_path)
    else:
        raise ValueError(f"Format non supporté : {filename}")

# ============================================================================
# UTILITAIRES D'ANALYSE
# ============================================================================

def summarize_dataset(df: pd.DataFrame, name: str = "Dataset") -> Dict[str, Any]:
    """Résumé complet d'un dataset."""
    summary = {
        'name': name,
        'shape': df.shape,
        'memory_mb': round(df.memory_usage(deep=True).sum() / 1024**2, 2),
        'dtypes': df.dtypes.value_counts().to_dict(),
        'null_counts': df.isnull().sum().sum(),
        'null_percentage': round(df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100, 2)
    }
    
    # Colonnes numériques
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        summary['numeric_stats'] = {
            'count': len(numeric_cols),
            'mean_values': df[numeric_cols].mean().mean(),
            'std_values': df[numeric_cols].std().mean()
        }
    
    # Colonnes catégorielles
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        summary['categorical_stats'] = {
            'count': len(categorical_cols),
            'unique_values_mean': df[categorical_cols].nunique().mean()
        }
    
    return summary

def print_dataset_summary(df: pd.DataFrame, name: str = "Dataset") -> None:
    """Affiche un résumé formaté d'un dataset."""
    summary = summarize_dataset(df, name)
    
    print(f"\n📊 RÉSUMÉ : {summary['name']}")
    print("=" * 40)
    print(f"📏 Dimensions: {summary['shape']}")
    print(f"💾 Mémoire: {summary['memory_mb']} MB")
    print(f"❓ Valeurs manquantes: {summary['null_counts']} ({summary['null_percentage']}%)")
    print(f"🏷️ Types: {summary['dtypes']}")
    
    if 'numeric_stats' in summary:
        print(f"🔢 Colonnes numériques: {summary['numeric_stats']['count']}")
    
    if 'categorical_stats' in summary:
        print(f"📝 Colonnes catégorielles: {summary['categorical_stats']['count']}")

# ============================================================================
# UTILITAIRES DE VALIDATION
# ============================================================================

def validate_splits(X_train: pd.DataFrame, X_test: pd.DataFrame, 
                   y_train: pd.Series, y_test: pd.Series) -> bool:
    """Valide la cohérence des splits train/test."""
    errors = []
    
    # Vérification des dimensions
    if len(X_train) != len(y_train):
        errors.append(f"X_train ({len(X_train)}) et y_train ({len(y_train)}) ont des tailles différentes")
    
    if len(X_test) != len(y_test):
        errors.append(f"X_test ({len(X_test)}) et y_test ({len(y_test)}) ont des tailles différentes")
    
    # Vérification des colonnes
    if not X_train.columns.equals(X_test.columns):
        errors.append("X_train et X_test ont des colonnes différentes")
    
    # Vérification des index
    common_indices = set(X_train.index).intersection(set(X_test.index))
    if common_indices:
        errors.append(f"Indices communs entre train et test: {len(common_indices)}")
    
    if errors:
        print("❌ ERREURS DE VALIDATION:")
        for error in errors:
            print(f"   - {error}")
        return False
    else:
        print("✅ Validation des splits réussie")
        return True 