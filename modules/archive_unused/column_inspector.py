# modules/preprocessing/column_inspector.py

"""
Module d'inspection et classification des colonnes pour STA211.
Analyse les types de données, valide la structure et met à jour la configuration.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any

def classify_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Classifie les colonnes d'un DataFrame selon leur type logique.
    
    Args:
        df: DataFrame à analyser
        
    Returns:
        Dict: Classification des colonnes par type
    """
    
    # Classification par type pandas
    continuous_cols = df.select_dtypes(include=['float64']).columns.tolist()
    int_cols = df.select_dtypes(include=['int64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Vérification binaire réelle pour les colonnes int
    binary_cols = []
    non_binary_cols = []
    
    for col in int_cols:
        unique_vals = df[col].dropna().unique()
        if len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1}):
            binary_cols.append(col)
        else:
            non_binary_cols.append(col)
    
    return {
        'continuous': continuous_cols,
        'binary': binary_cols,
        'non_binary_int': non_binary_cols,
        'categorical': categorical_cols
    }

def validate_target_variable(df: pd.DataFrame, target_col: str = 'y') -> Dict[str, Any]:
    """
    Valide la variable cible du dataset.
    
    Args:
        df: DataFrame contenant la variable cible
        target_col: Nom de la variable cible
        
    Returns:
        Dict: Résultats de validation
    """
    
    if target_col not in df.columns:
        return {'valid': False, 'error': f"Variable '{target_col}' manquante"}
    
    # Informations de base
    target_dtype = df[target_col].dtype
    unique_vals = sorted(df[target_col].dropna().unique())
    missing_count = df[target_col].isnull().sum()
    
    # Validation binaire
    is_binary_encoded = (
        len(unique_vals) == 2 and 
        set(unique_vals) == {0, 1} and 
        target_dtype in ['int64', 'int32']
    )
    
    result = {
        'valid': is_binary_encoded,
        'dtype': target_dtype,
        'unique_values': unique_vals,
        'missing_count': missing_count,
        'is_binary_encoded': is_binary_encoded
    }
    
    # Distribution des classes si valide
    if is_binary_encoded:
        class_counts = df[target_col].value_counts().sort_index()
        class_props = (class_counts / class_counts.sum() * 100).round(2)
        
        result['distribution'] = {
            'counts': dict(class_counts),
            'proportions': dict(class_props)
        }
    
    return result

def compare_dataset_structures(df1: pd.DataFrame, df2: pd.DataFrame) -> Dict[str, Any]:
    """
    Compare la structure de deux datasets.
    
    Args:
        df1: Premier dataset (train)
        df2: Deuxième dataset (test)
        
    Returns:
        Dict: Résultats de la comparaison
    """
    
    # Colonnes communes
    common_cols = set(df1.columns) & set(df2.columns)
    
    # Détection des incohérences de types
    type_mismatches = []
    for col in common_cols:
        if df1[col].dtype != df2[col].dtype:
            type_mismatches.append((col, df1[col].dtype, df2[col].dtype))
    
    # Classification des deux datasets
    struct1 = classify_columns(df1)
    struct2 = classify_columns(df2)
    
    return {
        'common_columns': len(common_cols),
        'type_mismatches': type_mismatches,
        'structures': {'train': struct1, 'test': struct2},
        'coherent': len(type_mismatches) == 0
    }

def inspect_datasets(df_train: pd.DataFrame, df_test: pd.DataFrame, 
                    target_col: str = 'y', verbose: bool = True) -> Dict[str, Any]:
    """
    Inspection complète de deux datasets (train/test).
    
    Args:
        df_train: Dataset d'entraînement
        df_test: Dataset de test
        target_col: Nom de la variable cible
        verbose: Affichage détaillé
        
    Returns:
        Dict: Résultats complets de l'inspection
    """
    
    if verbose:
        print("🔍 Inspection automatisée des datasets")
        print("=" * 50)
    
    # Classification des colonnes
    train_structure = classify_columns(df_train)
    test_structure = classify_columns(df_test)
    
    if verbose:
        print(f"\n📊 Dataset d'entraînement {df_train.shape}:")
        print(f"  • Continues    : {len(train_structure['continuous']):>4}")
        print(f"  • Binaires     : {len(train_structure['binary']):>4}")
        print(f"  • Catégorielles: {len(train_structure['categorical']):>4}")
        
        print(f"\n📊 Dataset de test {df_test.shape}:")
        print(f"  • Continues    : {len(test_structure['continuous']):>4}")
        print(f"  • Binaires     : {len(test_structure['binary']):>4}")
        print(f"  • Catégorielles: {len(test_structure['categorical']):>4}")
    
    # Validation de la variable cible
    target_validation = validate_target_variable(df_train, target_col)
    
    if verbose:
        print(f"\n🎯 Variable cible '{target_col}':")
        if target_validation['valid']:
            print(f"  ✅ Encodage binaire valide")
            dist = target_validation['distribution']
            for val, (count, prop) in zip([0, 1], 
                                        zip(dist['counts'].values(), dist['proportions'].values())):
                class_name = "noad." if val == 0 else "ad."
                print(f"  • {val} ({class_name}): {count} ({prop:.1f}%)")
        else:
            print(f"  ❌ Problème détecté: {target_validation.get('error', 'Encodage invalide')}")
    
    # Comparaison des structures
    comparison = compare_dataset_structures(df_train, df_test)
    
    if verbose:
        print(f"\n🔄 Cohérence train/test:")
        if comparison['coherent']:
            print(f"  ✅ Types parfaitement cohérents")
        else:
            print(f"  ⚠️ {len(comparison['type_mismatches'])} incohérences détectées")
    
    # Résultats compilés
    results = {
        'train_structure': train_structure,
        'test_structure': test_structure,
        'target_validation': target_validation,
        'comparison': comparison,
        'summary': {
            'total_features': df_train.shape[1] - (1 if target_col in df_train.columns else 0),
            'target_ready': target_validation['valid'],
            'types_coherent': comparison['coherent']
        }
    }
    
    return results

def update_column_config(config, inspection_results: Dict[str, Any]) -> bool:
    """
    Met à jour la configuration avec les résultats d'inspection.
    
    Args:
        config: Objet de configuration
        inspection_results: Résultats de l'inspection
        
    Returns:
        bool: Succès de la mise à jour
    """
    
    try:
        train_struct = inspection_results['train_structure']
        summary = inspection_results['summary']
        
        # Mise à jour des listes de colonnes
        config.update("COLUMN_CONFIG.CONTINUOUS_COLS", train_struct['continuous'])
        config.update("COLUMN_CONFIG.BINARY_COLS", train_struct['binary'])
        config.update("COLUMN_CONFIG.CATEGORICAL_COLS", train_struct['categorical'])
        
        # Métadonnées
        config.update("COLUMN_CONFIG.TOTAL_FEATURES", summary['total_features'])
        config.update("COLUMN_CONFIG.TARGET_READY", summary['target_ready'])
        config.update("COLUMN_CONFIG.TYPES_COHERENT", summary['types_coherent'])
        
        return True
        
    except Exception as e:
        print(f"⚠️ Erreur mise à jour configuration : {e}")
        return False

def print_inspection_summary(inspection_results: Dict[str, Any], 
                           expected_features: int = None) -> None:
    """
    Affiche un résumé de l'inspection des colonnes.
    
    Args:
        inspection_results: Résultats de l'inspection
        expected_features: Nombre de features attendu
    """
    
    print(f"\n📋 RÉSUMÉ DE L'INSPECTION")
    print("=" * 40)
    
    train_struct = inspection_results['train_structure']
    summary = inspection_results['summary']
    target_val = inspection_results['target_validation']
    
    print(f"📊 Structure des données :")
    print(f"  • Variables continues    : {len(train_struct['continuous']):>4}")
    print(f"  • Variables binaires     : {len(train_struct['binary']):>4}")
    print(f"  • Variables catégorielles: {len(train_struct['categorical']):>4}")
    
    total_features = summary['total_features']
    if expected_features:
        print(f"  • Total features         : {total_features:>4} (attendu: {expected_features})")
    else:
        print(f"  • Total features         : {total_features:>4}")
    
    print(f"\n🎯 Validation :")
    print(f"  • Variable cible         : {'✅' if summary['target_ready'] else '❌'}")
    print(f"  • Cohérence train/test   : {'✅' if summary['types_coherent'] else '❌'}")
    
    print(f"\n🌍 Variables exportées :")
    print(f"  • continuous_cols        : {len(train_struct['continuous'])} variables")
    print(f"  • binary_cols            : {len(train_struct['binary'])} variables")
    print(f"  • categorical_cols       : {len(train_struct['categorical'])} variables")