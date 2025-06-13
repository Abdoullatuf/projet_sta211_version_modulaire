"""
Module de prétraitement des données pour STA211.

Contient les outils pour :
- Génération automatisée de datasets avec protection X4
- Pipeline de prétraitement complet et robuste
- Imputation des valeurs manquantes (MICE, KNN) 
- Détection et suppression des outliers
- Transformation des variables (Yeo-Johnson)
- Filtrage de la colinéarité avec protection
- Réorganisation optimisée des colonnes
- Validation et diagnostic automatisés
- Inspection et classification des colonnes
- Comparaison des méthodes d'imputation
- Export multi-format et rapports détaillés

Version: 2.1 (Corrigée)
Auteur: Abdoullatuf
Date: 2025
"""

# ============================================================================
# IMPORTS PRINCIPAUX - PIPELINE ET GÉNÉRATION
# ============================================================================

# Import sécurisé du générateur de datasets
try:
    from .dataset_generator import (
        DatasetGenerator,
        quick_generate_datasets,
        print_generation_summary
    )
    _GENERATOR_AVAILABLE = True
except ImportError:
    _GENERATOR_AVAILABLE = False

# Import sécurisé des fonctions principales
try:
    from .final_preprocessing import (
        # 🔧 Utilitaires de base
        convert_X4_to_int,
        apply_yeojohnson,
        
        # 🔗 Gestion de la corrélation  
        find_highly_correlated_groups,
        drop_correlated_duplicates,
        #apply_collinearity_filter,
        
        # 🛡️ Validation et protection X4
        validate_x4_presence,
        quick_x4_check,
        
        # 📌 Gestion de l'ordre des colonnes
        reorder_columns_priority,
        #check_column_order,
        reorganize_existing_datasets,
        
        # 🚀 Pipeline principal
        prepare_final_dataset,
        
        # 🔧 Utilitaires pour datasets existants
        apply_full_preprocessing_to_existing,
        batch_process_datasets,
        validate_all_datasets,
        print_validation_summary,
        
        # 🔍 Diagnostic et debug
        diagnose_pipeline_issue,
        
        # 💾 Export et sauvegarde
        export_datasets_multiple_formats,
        create_preprocessing_report
    )
    _FINAL_PREPROCESSING_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Erreur import final_preprocessing: {e}")
    _FINAL_PREPROCESSING_AVAILABLE = False

# Import sécurisé des nouvelles fonctions (version corrigée)
try:
    from .final_preprocessing import (
        # 🆕 Nouvelles fonctions de la version corrigée
        prepare_dataset_safe,
        run_comprehensive_test,
        quick_pipeline_test,
        get_preprocessing_summary,
        print_preprocessing_summary,
        compare_preprocessing_results,
        migrate_old_results,
        check_compatibility,
        print_compatibility_report
    )
    _NEW_FUNCTIONS_AVAILABLE = True
except ImportError:
    _NEW_FUNCTIONS_AVAILABLE = False

# ============================================================================
# IMPORTS MODULES SPÉCIALISÉS
# ============================================================================

try:
    from .missing_values import (
        analyze_missing_values,
        handle_missing_values,
        find_optimal_k
    )
    _MISSING_VALUES_AVAILABLE = True
except ImportError:
    _MISSING_VALUES_AVAILABLE = False

try:
    from .outliers import (
        detect_and_remove_outliers
    )
    _OUTLIERS_AVAILABLE = True
except ImportError:
    _OUTLIERS_AVAILABLE = False

try:
    from .column_inspector import (
        inspect_datasets, 
        update_column_config,
        print_inspection_summary
    )
    _COLUMN_INSPECTOR_AVAILABLE = True
except ImportError:
    _COLUMN_INSPECTOR_AVAILABLE = False

try:
    from .data_loader import (
        load_data
    )
    _DATA_LOADER_AVAILABLE = True
except ImportError:
    _DATA_LOADER_AVAILABLE = False

# ============================================================================
# IMPORTS COMPARAISON D'IMPUTATION
# ============================================================================

try:
    from .comparison_methode_imputation import (
        compare_imputation_methods,
        plot_distributions_comparison,
        plot_statistics_comparison,
        statistical_tests_imputation,
        calculate_imputation_quality_scores,
        generate_final_recommendations,
        run_imputation_comparison
    )
    _COMPARISON_AVAILABLE = True
except ImportError:
    _COMPARISON_AVAILABLE = False

# ============================================================================
# IMPORT PIPELINE PRINCIPAL (RÉTROCOMPATIBILITÉ)
# ============================================================================

try:
    from .main_pipeline import prepare_final_dataset as prepare_final_dataset_legacy
    _LEGACY_PIPELINE_AVAILABLE = True
except ImportError:
    _LEGACY_PIPELINE_AVAILABLE = False

# ============================================================================
# EXPORTS PUBLICS ORGANISÉS
# ============================================================================

# Exports principaux (conditionnels selon disponibilité)
_CORE_EXPORTS = []

# Génération de datasets
if _GENERATOR_AVAILABLE:
    _CORE_EXPORTS.extend([
        'DatasetGenerator',
        'quick_generate_datasets', 
        'print_generation_summary'
    ])

# Pipeline principal
if _FINAL_PREPROCESSING_AVAILABLE:
    _CORE_EXPORTS.extend([
        # 🔧 Utilitaires de base
        'convert_X4_to_int',
        'apply_yeojohnson',

        # 🔗 Gestion de la corrélation
        'find_highly_correlated_groups',
        'drop_correlated_duplicates',
        'apply_collinearity_filter',

        # 🛡️ Validation et protection X4
        'validate_x4_presence',
        'quick_x4_check',

        # 📌 Gestion de l'ordre des colonnes
        'reorder_columns_priority',
        'check_column_order',
        'reorganize_existing_datasets',

        # 🚀 Pipeline principal
        'prepare_final_dataset',

        # 🔧 Utilitaires pour datasets existants
        'apply_full_preprocessing_to_existing',
        'batch_process_datasets',
        'validate_all_datasets',
        'print_validation_summary',

        # 🔍 Diagnostic et debug
        'diagnose_pipeline_issue',

        # 💾 Export et sauvegarde
        'export_datasets_multiple_formats',
        'create_preprocessing_report'
    ])

# Nouvelles fonctions version corrigée
if _NEW_FUNCTIONS_AVAILABLE:
    _CORE_EXPORTS.extend([
        'prepare_dataset_safe',
        'run_comprehensive_test',
        'quick_pipeline_test',
        'get_preprocessing_summary',
        'print_preprocessing_summary',
        'compare_preprocessing_results',
        'migrate_old_results',
        'check_compatibility',
        'print_compatibility_report'
    ])

# Modules spécialisés
if _MISSING_VALUES_AVAILABLE:
    _CORE_EXPORTS.extend([
        'analyze_missing_values',
        'handle_missing_values',
        'find_optimal_k'
    ])

if _OUTLIERS_AVAILABLE:
    _CORE_EXPORTS.append('detect_and_remove_outliers')

if _COLUMN_INSPECTOR_AVAILABLE:
    _CORE_EXPORTS.extend([
        'inspect_datasets',
        'update_column_config',
        'print_inspection_summary'
    ])

if _DATA_LOADER_AVAILABLE:
    _CORE_EXPORTS.append('load_data')

# Exports conditionnels
_COMPARISON_EXPORTS = []
if _COMPARISON_AVAILABLE:
    _COMPARISON_EXPORTS.extend([
        'compare_imputation_methods',
        'plot_distributions_comparison',
        'plot_statistics_comparison',
        'statistical_tests_imputation',
        'calculate_imputation_quality_scores',
        'generate_final_recommendations',
        'run_imputation_comparison'
    ])

_LEGACY_EXPORTS = []
if _LEGACY_PIPELINE_AVAILABLE:
    _LEGACY_EXPORTS.append('prepare_final_dataset_legacy')

# Export final
__all__ = _CORE_EXPORTS + _COMPARISON_EXPORTS + _LEGACY_EXPORTS

# ============================================================================
# INFORMATIONS SUR LA DISPONIBILITÉ DES MODULES
# ============================================================================

def get_module_info():
    """
    Retourne les informations sur les modules disponibles.
    
    Returns:
        Dict avec le statut de chaque module
    """
    return {
        'final_preprocessing': _FINAL_PREPROCESSING_AVAILABLE,
        'new_functions': _NEW_FUNCTIONS_AVAILABLE,
        'generator': _GENERATOR_AVAILABLE,
        'missing_values': _MISSING_VALUES_AVAILABLE,
        'outliers': _OUTLIERS_AVAILABLE,
        'column_inspector': _COLUMN_INSPECTOR_AVAILABLE,
        'data_loader': _DATA_LOADER_AVAILABLE,
        'comparison_methods': _COMPARISON_AVAILABLE,
        'legacy_pipeline': _LEGACY_PIPELINE_AVAILABLE,
        'version': '2.1',
        'total_functions': len(__all__)
    }

def print_module_status():
    """
    Affiche le statut des modules de prétraitement.
    """
    info = get_module_info()
    
    print("📦 MODULES DE PRÉTRAITEMENT STA211")
    print("=" * 50)
    print(f"🔧 Version: {info['version']}")
    print(f"📊 Fonctions disponibles: {info['total_functions']}")
    print()
    print("📋 Statut des modules:")
    
    modules_status = [
        ('final_preprocessing', 'Pipeline principal'),
        ('new_functions', 'Nouvelles fonctions (v2.1)'),
        ('data_loader', 'Chargement de données'),
        ('missing_values', 'Valeurs manquantes'),
        ('outliers', 'Détection outliers'),
        ('generator', 'Génération datasets'),
        ('column_inspector', 'Inspection colonnes'),
        ('comparison_methods', 'Comparaison méthodes'),
        ('legacy_pipeline', 'Pipeline legacy')
    ]
    
    for module_key, module_name in modules_status:
        status = info[module_key]
        icon = "✅" if status else "❌"
        status_text = "Disponible" if status else "Indisponible"
        print(f"  {icon} {module_name}: {status_text}")
    
    # Recommandations si des modules manquent
    missing_modules = [name for key, name in modules_status if not info[key]]
    if missing_modules:
        print(f"\n⚠️ Modules manquants: {', '.join(missing_modules)}")
        
        if not info['new_functions']:
            print("💡 Pour les nouvelles fonctions, mettez à jour final_preprocessing.py")
        if not info['data_loader']:
            print("💡 Assurez-vous que data_loader.py existe")

def check_essential_functions():
    """Vérifie que les fonctions essentielles sont disponibles."""
    essential_functions = [
        'prepare_final_dataset',
        'load_data',
        'handle_missing_values'
    ]
    
    available_functions = []
    missing_functions = []
    
    for func_name in essential_functions:
        if func_name in globals():
            available_functions.append(func_name)
        else:
            missing_functions.append(func_name)
    
    print("🔍 VÉRIFICATION DES FONCTIONS ESSENTIELLES")
    print("=" * 50)
    
    for func in available_functions:
        print(f"✅ {func}")
    
    for func in missing_functions:
        print(f"❌ {func}")
    
    if missing_functions:
        print(f"\n⚠️ {len(missing_functions)} fonctions essentielles manquantes")
        return False
    else:
        print(f"\n✅ Toutes les fonctions essentielles ({len(available_functions)}) sont disponibles")
        return True

# ============================================================================
# FONCTIONS DE CONVENANCE
# ============================================================================

def quick_start_preprocessing(file_path, **kwargs):
    """
    Démarrage rapide du prétraitement avec paramètres par défaut.
    
    Args:
        file_path: Chemin vers le fichier de données
        **kwargs: Arguments supplémentaires pour prepare_final_dataset
        
    Returns:
        DataFrame prétraité
        
    Example:
        >>> df_clean = quick_start_preprocessing("data_train.csv")
        >>> quick_x4_check(df_clean)
    """
    if not _FINAL_PREPROCESSING_AVAILABLE:
        raise ImportError("Module final_preprocessing non disponible")
    
    return prepare_final_dataset(
        file_path=file_path,
        protect_x4=True,
        display_info=True,
        **kwargs
    )

def safe_preprocessing(file_path, **kwargs):
    """
    Version sécurisée du prétraitement (utilise prepare_dataset_safe si disponible).
    
    Args:
        file_path: Chemin vers le fichier
        **kwargs: Arguments supplémentaires
        
    Returns:
        DataFrame prétraité
    """
    if _NEW_FUNCTIONS_AVAILABLE and 'prepare_dataset_safe' in globals():
        return prepare_dataset_safe(file_path, **kwargs)
    elif _FINAL_PREPROCESSING_AVAILABLE:
        print("⚠️ prepare_dataset_safe non disponible, utilisation de prepare_final_dataset")
        return prepare_final_dataset(file_path, **kwargs)
    else:
        raise ImportError("Aucune fonction de prétraitement disponible")

def validate_preprocessing_result(df, dataset_name="Dataset"):
    """
    Validation rapide d'un résultat de prétraitement.
    
    Args:
        df: DataFrame à valider
        dataset_name: Nom du dataset pour l'affichage
        
    Returns:
        Bool: True si la validation est réussie
    """
    if not _FINAL_PREPROCESSING_AVAILABLE:
        print("❌ Module de validation non disponible")
        return False
    
    print(f"🔍 Validation de {dataset_name}...")
    
    # Vérifications de base
    x4_ok = quick_x4_check(df, dataset_name)
    order_ok = check_column_order(df, display_info=False)
    
    # Résumé
    if x4_ok and order_ok:
        print(f"✅ {dataset_name} validé avec succès")
        return True
    else:
        print(f"❌ {dataset_name} a des problèmes")
        if not x4_ok:
            print("  - X4 manquante ou incorrecte")
        if not order_ok:
            print("  - Ordre des colonnes incorrect")
        return False

def get_recommended_pipeline_config():
    """
    Retourne une configuration recommandée pour le pipeline.
    
    Returns:
        Dict avec la configuration recommandée
    """
    return {
        'strategy': 'mixed_mar_mcar',
        'mar_method': 'knn',  # ou 'mice'
        'correlation_threshold': 0.95,
        'protect_x4': True,
        'priority_cols': ['X1_trans', 'X2_trans', 'X3_trans', 'X4'],
        'display_info': True
    }

# ============================================================================
# DOCUMENTATION RAPIDE
# ============================================================================

_QUICK_DOC = """
🚀 UTILISATION RAPIDE DU MODULE PREPROCESSING (Version 2.1)

1. Vérification des modules disponibles:
   >>> from preprocessing import print_module_status, check_essential_functions
   >>> print_module_status()
   >>> check_essential_functions()

2. Pipeline simple (version sécurisée):
   >>> from preprocessing import safe_preprocessing
   >>> df = safe_preprocessing("data_train.csv")

3. Pipeline complet standard:
   >>> from preprocessing import quick_start_preprocessing
   >>> df = quick_start_preprocessing("data_train.csv")

4. Pipeline avec options avancées:
   >>> from preprocessing import prepare_final_dataset
   >>> df = prepare_final_dataset(
   ...     file_path="data_train.csv",
   ...     mar_method="mice",
   ...     correlation_threshold=0.95,
   ...     protect_x4=True
   ... )

5. Validation et diagnostic:
   >>> from preprocessing import validate_preprocessing_result
   >>> validate_preprocessing_result(df, "Mon Dataset")

6. Test complet (si disponible):
   >>> try:
   ...     from preprocessing import run_comprehensive_test
   ...     results = run_comprehensive_test("data_train.csv")
   ... except ImportError:
   ...     print("Fonction de test avancé non disponible")

Pour plus d'informations: print_module_status()
"""

def help_preprocessing():
    """Affiche l'aide rapide du module."""
    print(_QUICK_DOC)

# ============================================================================
# AJOUT D'EXPORTS POUR LA DOCUMENTATION
# ============================================================================

__all__.extend([
    'get_module_info',
    'print_module_status',
    'check_essential_functions',
    'quick_start_preprocessing',
    'safe_preprocessing',
    'validate_preprocessing_result',
    'get_recommended_pipeline_config',
    'help_preprocessing'
])

# ============================================================================
# MESSAGE D'INITIALISATION
# ============================================================================

def _print_init_message():
    """Message d'initialisation du module."""
    import os
    if os.getenv('PREPROCESSING_VERBOSE', '').lower() in ('1', 'true', 'yes'):
        print("📦 Module preprocessing STA211 v2.1 chargé")
        print(f"   ✅ {len(__all__)} fonctions disponibles")
        info = get_module_info()
        if not info['new_functions']:
            print("   ⚠️ Nouvelles fonctions non disponibles - mettez à jour final_preprocessing.py")
        print("   💡 Utilisez help_preprocessing() pour l'aide rapide")

# Uncomment the next line if you want initialization message
# _print_init_message()