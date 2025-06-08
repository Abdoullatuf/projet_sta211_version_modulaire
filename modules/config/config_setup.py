# modules/config/config_setup.py

"""
Module de configuration simplifiée pour le projet STA211.
Gère la création, validation et export de la configuration projet.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

def setup_project_configuration(project_name: str, version: str, author: str, 
                               paths: Dict, random_state: int = 42, 
                               verbose: bool = True) -> Dict[str, Any]:
    """
    Configure automatiquement le projet STA211 avec optimisation F1-score.
    
    Args:
        project_name: Nom du projet
        version: Version du projet
        author: Auteur du projet
        paths: Dictionnaire des chemins
        random_state: Graine aléatoire
        verbose: Affichage détaillé
        
    Returns:
        Dict: Configuration complète et variables exportées
    """
    
    if verbose:
        print("⚙️ Configuration automatisée du projet STA211...")
    
    try:
        from config.project_config import ProjectConfig, create_config
        if verbose:
            print("✅ Module project_config importé")
    except ImportError as e:
        if verbose:
            print(f"❌ Erreur import project_config : {e}")
        raise
    
    # ========================================================================
    # CRÉATION DE LA CONFIGURATION DE BASE
    # ========================================================================
    
    config = create_config(
        project_name=project_name,
        version=version,
        author=author,
        paths=paths
    )
    
    if verbose:
        print("✅ Configuration de base créée")
    
    # ========================================================================
    # OPTIMISATIONS F1-SCORE ET DATASET
    # ========================================================================
    
    # Configuration F1-score
    f1_config = {
        "PROJECT_CONFIG.SCORING": "f1",
        "PROJECT_CONFIG.SCORING_METRICS": ["f1", "roc_auc", "precision", "recall"],
        "PROJECT_CONFIG.PRIMARY_METRIC": "f1",
        "PROJECT_CONFIG.CROSS_VALIDATION": True,
        "PROJECT_CONFIG.CV_FOLDS": 5,
        "MODEL_CONFIG.class_weight": "balanced",
        "MODEL_CONFIG.stratify": True
    }
    
    # Configuration dataset Internet Advertisements
    dataset_config = {
        "PROJECT_CONFIG.EXPECTED_TRAIN_SIZE": 2459,
        "PROJECT_CONFIG.EXPECTED_TEST_SIZE": 820,
        "PROJECT_CONFIG.EXPECTED_FEATURES": 1558,
        "PROJECT_CONFIG.CLASS_IMBALANCE_RATIO": 6.15,
        "PIPELINE_CONFIG.handle_imbalance": True,
        "PIPELINE_CONFIG.correlation_threshold": 0.95
    }
    
    # Application des configurations
    all_updates = {**f1_config, **dataset_config}
    
    for key, value in all_updates.items():
        config.update(key, value)
    
    if verbose:
        print("🎯 Optimisation F1-score appliquée")
        print("📊 Configuration dataset appliquée")
    
    # ========================================================================
    # VALIDATION DE LA CONFIGURATION
    # ========================================================================
    
    critical_configs = [
        ("Métrique primaire", "PROJECT_CONFIG.PRIMARY_METRIC"),
        ("Seuil corrélation", "PIPELINE_CONFIG.correlation_threshold"),
        ("Class weight", "MODEL_CONFIG.class_weight"),
        ("CV folds", "PROJECT_CONFIG.CV_FOLDS")
    ]
    
    config_valid = True
    
    for description, config_path in critical_configs:
        try:
            value = config.get(config_path)
            if verbose:
                print(f"  ✅ {description}: {value}")
        except Exception as e:
            if verbose:
                print(f"  ❌ {description}: Erreur - {e}")
            config_valid = False
    
    if not config_valid:
        raise ValueError("Configuration invalide détectée")
    
    # ========================================================================
    # SAUVEGARDE ROBUSTE
    # ========================================================================
    
    if verbose:
        print("\n💾 Sauvegarde de la configuration...")
    
    config_saved = False
    config_file = None
    
    try:
        # Tentative sauvegarde normale
        config_dir = Path(paths["ROOT_DIR"]) / "config"
        config_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        config_file = config_dir / f"project_config_v{version}_{timestamp}.json"
        
        config.save_config(str(config_file))
        config_saved = True
        
        if verbose:
            print(f"✅ Configuration sauvegardée : {config_file}")
            
    except Exception as e:
        if verbose:
            print(f"⚠️ Sauvegarde normale échouée : {e}")
            print("💡 Sauvegarde manuelle...")
        
        # Sauvegarde manuelle
        try:
            config_data = {
                "project_info": {
                    "name": project_name,
                    "version": version,
                    "author": author,
                    "timestamp": datetime.now().isoformat()
                },
                "f1_optimization": {
                    "primary_metric": config.get("PROJECT_CONFIG.PRIMARY_METRIC"),
                    "scoring_metrics": config.get("PROJECT_CONFIG.SCORING_METRICS"),
                    "class_weight": config.get("MODEL_CONFIG.class_weight"),
                    "cv_folds": config.get("PROJECT_CONFIG.CV_FOLDS")
                },
                "dataset_config": {
                    "correlation_threshold": config.get("PIPELINE_CONFIG.correlation_threshold"),
                    "handle_imbalance": config.get("PIPELINE_CONFIG.handle_imbalance"),
                    "expected_train_size": config.get("PROJECT_CONFIG.EXPECTED_TRAIN_SIZE")
                }
            }
            
            config_file_manual = config_dir / f"project_config_manual_v{version}.json"
            with open(config_file_manual, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            config_saved = True
            config_file = config_file_manual
            
            if verbose:
                print(f"✅ Sauvegarde manuelle réussie : {config_file}")
                
        except Exception as e2:
            if verbose:
                print(f"❌ Sauvegarde manuelle échouée : {e2}")
    
    # ========================================================================
    # EXPORT DES VARIABLES
    # ========================================================================
    
    config_exports = {
        'PROJECT_CONFIG': config.PROJECT_CONFIG,
        'COLUMN_CONFIG': config.COLUMN_CONFIG,
        'VIZ_CONFIG': config.VIZ_CONFIG,
        'MODEL_CONFIG': config.MODEL_CONFIG,
        'PIPELINE_CONFIG': config.PIPELINE_CONFIG,
        'SAVE_PATHS': config.SAVE_PATHS,
        'F1_OPTIMIZATION': config.F1_OPTIMIZATION,
        'config': config
    }
    
    # ========================================================================
    # RÉSUMÉ
    # ========================================================================
    
    if verbose:
        print(f"\n📋 Résumé de la configuration :")
        print(f"  🎯 Métrique cible     : {config.get('PROJECT_CONFIG.PRIMARY_METRIC')}")
        print(f"  🔢 Random State      : {random_state}")
        print(f"  📊 Seuil corrélation : {config.get('PIPELINE_CONFIG.correlation_threshold')}")
        print(f"  ⚖️ Gestion déséquilibre : {config.get('PIPELINE_CONFIG.handle_imbalance')}")
        print(f"  💾 Config sauvegardée : {config_saved}")
        print(f"  🌍 Variables exportées : {len(config_exports)}")
        
        print(f"\n✅ Configuration complète et prête")
    
    return config_exports


def quick_project_config(project_name: str, version: str, author: str, paths: Dict,
                        random_state: int = 42) -> Dict[str, Any]:
    """Configuration rapide et silencieuse."""
    return setup_project_configuration(
        project_name=project_name,
        version=version, 
        author=author,
        paths=paths,
        random_state=random_state,
        verbose=True
    )


def validate_required_variables(**kwargs) -> bool:
    """Valide que toutes les variables requises sont présentes."""
    
    required = ['project_name', 'version', 'author', 'paths']
    missing = [var for var in required if var not in kwargs or kwargs[var] is None]
    
    if missing:
        raise NameError(f"Variables requises manquantes : {missing}")
    
    return True