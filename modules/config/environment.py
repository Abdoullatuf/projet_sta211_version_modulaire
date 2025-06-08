#modules/config/environement.py

"""
Module de configuration de l'environnement pour le projet STA211

Ce module centralise toute la configuration de l'environnement :
- Détection d'environnement (Colab/Local)
- Installation des packages
- Configuration des chemins
- Import des bibliothèques
- Configuration globale du projet
"""

import sys
import os
import warnings
import subprocess
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, Tuple, Optional, Any

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def detect_environment() -> str:
    """
    Détecte l'environnement d'exécution
    
    Returns:
    --------
    str
        'colab' ou 'local'
    """
    try:
        import google.colab
        return "colab"
    except ImportError:
        return "local"

def install_required_packages(packages: Optional[list] = None, quiet: bool = True) -> bool:
    """
    Installe les packages requis si nécessaire
    
    Parameters:
    -----------
    packages : Optional[list]
        Liste des packages à installer. Par défaut: packages du projet
    quiet : bool
        Installation silencieuse
        
    Returns:
    --------
    bool
        True si l'installation réussit
    """
    if packages is None:
        packages = [
            "scikit-learn", 
            "imbalanced-learn", 
            "umap-learn", 
            "prince",
            "seaborn>=0.11.0",
            "matplotlib>=3.5.0",
            "pandas>=1.3.0",
            "numpy>=1.21.0"
        ]
    
    logger.info(f"Installation de {len(packages)} packages...")
    
    try:
        for package in packages:
            cmd = [sys.executable, "-m", "pip", "install"]
            if quiet:
                cmd.append("-q")
            cmd.append(package)
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.warning(f"Problème avec {package}: {result.stderr}")
            else:
                logger.debug(f"Package installé: {package}")
        
        logger.info("Installation des packages terminée")
        return True
        
    except Exception as e:
        logger.error(f"Erreur lors de l'installation: {e}")
        return False

def setup_google_drive() -> bool:
    """
    Monte Google Drive si on est dans Colab
    
    Returns:
    --------
    bool
        True si le montage réussit
    """
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=True)
        logger.info("Google Drive monté avec succès")
        return True
    except ImportError:
        logger.debug("Pas dans Colab, montage Drive ignoré")
        return False
    except Exception as e:
        logger.error(f"Erreur montage Google Drive: {e}")
        return False

def import_core_libraries() -> Dict[str, Any]:
    """
    Importe et configure les bibliothèques principales
    
    Returns:
    --------
    Dict[str, Any]
        Dictionnaire des bibliothèques importées
    """
    libraries = {}
    
    try:
        # Manipulation des données
        import numpy as np
        import pandas as pd
        libraries['numpy'] = np
        libraries['pandas'] = pd
        logger.debug("Bibliothèques de données importées")
        
        # Visualisation
        import matplotlib.pyplot as plt
        import seaborn as sns
        libraries['matplotlib'] = plt
        libraries['seaborn'] = sns
        logger.debug("Bibliothèques de visualisation importées")
        
        # Machine Learning
        import sklearn
        libraries['sklearn'] = sklearn
        logger.debug("Scikit-learn importé")
        
        # Statistiques
        from scipy import stats
        libraries['scipy_stats'] = stats
        logger.debug("SciPy stats importé")
        
        # IPython/Jupyter
        try:
            from IPython.display import display, Markdown, HTML
            libraries['display'] = display
            libraries['Markdown'] = Markdown
            libraries['HTML'] = HTML
            logger.debug("IPython display importé")
        except ImportError:
            logger.debug("IPython non disponible")
        
        logger.info(f"Bibliothèques principales importées: {len(libraries)}")
        
    except ImportError as e:
        logger.error(f"Erreur import bibliothèques: {e}")
        
    return libraries

def import_optional_libraries() -> Dict[str, Any]:
    """
    Importe les bibliothèques optionnelles avec gestion d'erreur
    
    Returns:
    --------
    Dict[str, Any]
        Dictionnaire des bibliothèques optionnelles disponibles
    """
    optional = {}
    
    # UMAP pour la réduction de dimension
    try:
        import umap
        optional['umap'] = umap
        optional['UMAP_AVAILABLE'] = True
        logger.debug("UMAP disponible")
    except ImportError:
        optional['UMAP_AVAILABLE'] = False
        logger.debug("UMAP non disponible")
    
    # Prince pour l'analyse factorielle
    try:
        import prince
        optional['prince'] = prince
        optional['PRINCE_AVAILABLE'] = True
        logger.debug("Prince disponible")
    except ImportError:
        optional['PRINCE_AVAILABLE'] = False
        logger.debug("Prince non disponible")
    
    # Imbalanced-learn
    try:
        from imblearn.over_sampling import SMOTE
        from imblearn.under_sampling import RandomUnderSampler
        optional['SMOTE'] = SMOTE
        optional['RandomUnderSampler'] = RandomUnderSampler
        optional['IMBALANCED_AVAILABLE'] = True
        logger.debug("Imbalanced-learn disponible")
    except ImportError:
        optional['IMBALANCED_AVAILABLE'] = False
        logger.debug("Imbalanced-learn non disponible")
    
    # XGBoost
    try:
        import xgboost as xgb
        optional['xgboost'] = xgb
        optional['XGBOOST_AVAILABLE'] = True
        logger.debug("XGBoost disponible")
    except ImportError:
        optional['XGBOOST_AVAILABLE'] = False
        logger.debug("XGBoost non disponible")
    
    # LightGBM
    try:
        import lightgbm as lgb
        optional['lightgbm'] = lgb
        optional['LIGHTGBM_AVAILABLE'] = True
        logger.debug("LightGBM disponible")
    except ImportError:
        optional['LIGHTGBM_AVAILABLE'] = False
        logger.debug("LightGBM non disponible")
    
    available_count = sum(1 for key in optional.keys() if key.endswith('_AVAILABLE') and optional[key])
    logger.info(f"Bibliothèques optionnelles disponibles: {available_count}")
    
    return optional

def setup_random_seeds(seed: int = 42) -> int:
    """
    Configure la reproductibilité avec les graines aléatoires
    
    Parameters:
    -----------
    seed : int
        Graine aléatoire principale
        
    Returns:
    --------
    int
        Graine configurée
    """
    try:
        import numpy as np
        import random
        
        # Configuration des graines
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        
        # Configuration pour TensorFlow si disponible
        try:
            import tensorflow as tf
            tf.random.set_seed(seed)
        except ImportError:
            pass
        
        # Configuration pour PyTorch si disponible
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass
        
        logger.info(f"Graines aléatoires configurées: {seed}")
        return seed
        
    except Exception as e:
        logger.error(f"Erreur configuration graines: {e}")
        return seed

def verify_environment() -> Dict[str, Any]:
    """
    Vérifie l'environnement et retourne les informations système
    
    Returns:
    --------
    Dict[str, Any]
        Informations sur l'environnement
    """
    env_info = {
        'python_version': sys.version.split()[0],
        'platform': sys.platform,
        'working_directory': os.getcwd(),
        'environment_type': detect_environment(),
        'available_memory': 'N/A',
        'cpu_count': os.cpu_count(),
        'timestamp': datetime.now().isoformat()
    }
    
    # Mémoire disponible (si psutil est disponible)
    try:
        import psutil
        memory = psutil.virtual_memory()
        env_info['available_memory'] = f"{memory.available // (1024**3):.1f} GB"
        env_info['total_memory'] = f"{memory.total // (1024**3):.1f} GB"
    except ImportError:
        pass
    
    # Informations GPU (si disponible)
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        if gpus:
            env_info['gpu_count'] = len(gpus)
            env_info['gpu_names'] = [gpu.name for gpu in gpus]
    except ImportError:
        env_info['gpu_count'] = 0
    
    return env_info

def display_environment_summary(env_info: Dict[str, Any], libraries: Dict[str, Any]):
    """
    Affiche un résumé de l'environnement configuré
    
    Parameters:
    -----------
    env_info : Dict[str, Any]
        Informations sur l'environnement
    libraries : Dict[str, Any]
        Bibliothèques disponibles
    """
    print("\n🔧 Résumé de l'environnement")
    print("=" * 50)
    
    # Informations système
    print("📱 Système:")
    print(f"  • Environnement: {env_info['environment_type']}")
    print(f"  • Python: {env_info['python_version']}")
    print(f"  • Plateforme: {env_info['platform']}")
    print(f"  • CPU: {env_info['cpu_count']} cores")
    if env_info['available_memory'] != 'N/A':
        print(f"  • Mémoire: {env_info['available_memory']}")
    
    # Bibliothèques principales
    print("\n📦 Bibliothèques principales:")
    core_libs = ['numpy', 'pandas', 'sklearn', 'matplotlib', 'seaborn']
    for lib in core_libs:
        if lib in libraries:
            try:
                version = getattr(libraries[lib], '__version__', 'N/A')
                print(f"  ✅ {lib}: {version}")
            except:
                print(f"  ✅ {lib}: disponible")
        else:
            print(f"  ❌ {lib}: non disponible")
    
    # Bibliothèques optionnelles
    print("\n📦 Bibliothèques optionnelles:")
    optional_status = {
        'UMAP': libraries.get('UMAP_AVAILABLE', False),
        'Prince': libraries.get('PRINCE_AVAILABLE', False),
        'Imbalanced-learn': libraries.get('IMBALANCED_AVAILABLE', False),
        'XGBoost': libraries.get('XGBOOST_AVAILABLE', False),
        'LightGBM': libraries.get('LIGHTGBM_AVAILABLE', False)
    }
    
    for name, available in optional_status.items():
        status = "✅" if available else "❌"
        print(f"  {status} {name}")
    
    print(f"\n📍 Répertoire de travail: {env_info['working_directory']}")
    print(f"⏰ Configuré le: {env_info['timestamp'][:19]}")

def setup_environment(
    project_info: Dict[str, str],
    random_state: int = 42,
    install_packages: bool = False,
    quiet: bool = True
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Configuration complète de l'environnement pour le projet STA211
    
    Parameters:
    -----------
    project_info : Dict[str, str]
        Informations du projet (name, author, version, target_metric)
    random_state : int
        Graine aléatoire pour la reproductibilité
    install_packages : bool
        Installer automatiquement les packages requis
    quiet : bool
        Mode silencieux pour l'installation
        
    Returns:
    --------
    Tuple[Dict, Dict, Dict]
        (config, paths, metadata) - Configuration complète
    """
    
    logger.info(f"🚀 Configuration de l'environnement pour {project_info.get('name', 'STA211')}")
    
    # 1. Détection de l'environnement
    env = detect_environment()
    logger.info(f"Environnement détecté: {env}")
    
    # 2. Configuration Google Drive si nécessaire
    if env == "colab":
        setup_google_drive()
    
    # 3. Installation des packages si demandée
    if install_packages:
        install_success = install_required_packages(quiet=quiet)
        if not install_success:
            logger.warning("Certains packages n'ont pas pu être installés")
    
    # 4. Import des bibliothèques
    core_libraries = import_core_libraries()
    optional_libraries = import_optional_libraries()
    all_libraries = {**core_libraries, **optional_libraries}
    
    # 5. Configuration des graines aléatoires
    setup_random_seeds(random_state)
    
    # 6. Configuration des chemins - Import absolu corrigé
    try:
        from paths_config import setup_project_paths
    except ImportError:
        # Fallback si import relatif nécessaire
        try:
            import importlib.util
            current_dir = Path(__file__).parent
            spec = importlib.util.spec_from_file_location("paths_config", current_dir / "paths_config.py")
            paths_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(paths_module)
            setup_project_paths = paths_module.setup_project_paths
        except Exception as e:
            logger.error(f"Impossible d'importer setup_project_paths: {e}")
            raise ImportError("Module paths_config non disponible")
    
    paths = setup_project_paths(env)
    
    # 7. Configuration d'affichage
    try:
        from display_config import setup_display_config
    except ImportError:
        # Fallback
        try:
            import importlib.util
            current_dir = Path(__file__).parent
            spec = importlib.util.spec_from_file_location("display_config", current_dir / "display_config.py")
            display_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(display_module)
            setup_display_config = display_module.setup_display_config
        except Exception as e:
            logger.warning(f"Configuration d'affichage non disponible: {e}")
            setup_display_config = lambda: {}
    
    display_config = setup_display_config()
    
    # 8. Configuration du projet
    try:
        from project_config import create_config
    except ImportError:
        # Fallback
        try:
            import importlib.util
            current_dir = Path(__file__).parent
            spec = importlib.util.spec_from_file_location("project_config", current_dir / "project_config.py")
            project_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(project_module)
            create_config = project_module.create_config
        except Exception as e:
            logger.error(f"Impossible d'importer create_config: {e}")
            raise ImportError("Module project_config non disponible")
    
    config = create_config(
        project_name=project_info.get('name', 'STA211 Project'),
        version=project_info.get('version', '1.0'),
        author=project_info.get('author', 'Unknown'),
        paths=paths
    )
    
    # Configuration spécifique F1 si c'est la métrique cible
    if project_info.get('target_metric') == 'F1-Score':
        config.update("PROJECT_CONFIG.SCORING", "f1")
        config.update("PROJECT_CONFIG.PRIMARY_METRIC", "f1")
        config.update("PROJECT_CONFIG.OPTIMIZATION_DIRECTION", "maximize")
    
    # 9. Vérification de l'environnement
    env_info = verify_environment()
    
    # 10. Métadonnées de session
    metadata = {
        'project_info': project_info,
        'environment': env_info,
        'libraries': {
            'core_available': len(core_libraries),
            'optional_available': sum(1 for k, v in optional_libraries.items() 
                                    if k.endswith('_AVAILABLE') and v),
            'display_config': display_config
        },
        'configuration': {
            'random_state': random_state,
            'packages_installed': install_packages,
            'paths_configured': len(paths),
            'config_created': True
        },
        'session': {
            'started_at': datetime.now().isoformat(),
            'python_path': sys.path[:3],  # Premiers éléments du path
            'working_dir': os.getcwd()
        }
    }
    
    # 11. Ajout des bibliothèques aux globals pour compatibilité
    globals().update(all_libraries)
    
    # 12. Affichage du résumé
    display_environment_summary(env_info, all_libraries)
    
    # 13. Affichage du résumé de configuration
    print(f"\n📋 Configuration du projet")
    print("-" * 30)
    print(f"  🎯 Projet: {project_info.get('name', 'N/A')}")
    print(f"  👤 Auteur: {project_info.get('author', 'N/A')}")
    print(f"  🏷️  Version: {project_info.get('version', 'N/A')}")
    print(f"  📊 Métrique cible: {project_info.get('target_metric', 'N/A')}")
    print(f"  🎲 Random state: {random_state}")
    print(f"  📁 Chemins configurés: {len(paths)}")
    print(f"  ⚙️  Configuration: ✅ Terminée")
    
    logger.info("Configuration de l'environnement terminée avec succès")
    
    return config, paths, metadata

def quick_setup(project_name: str = "STA211 Project", random_state: int = 42) -> Tuple[Dict, Dict, Dict]:
    """
    Configuration rapide avec les paramètres par défaut
    
    Parameters:
    -----------
    project_name : str
        Nom du projet
    random_state : int
        Graine aléatoire
        
    Returns:
    --------
    Tuple[Dict, Dict, Dict]
        (config, paths, metadata)
    """
    project_info = {
        'name': project_name,
        'author': 'User',
        'version': '1.0',
        'target_metric': 'F1-Score'
    }
    
    return setup_environment(
        project_info=project_info,
        random_state=random_state,
        install_packages=True,
        quiet=True
    )

def validate_environment_setup(config: Dict, paths: Dict, metadata: Dict) -> bool:
    """
    Valide que l'environnement est correctement configuré
    
    Parameters:
    -----------
    config : Dict
        Configuration du projet
    paths : Dict
        Chemins configurés
    metadata : Dict
        Métadonnées de session
        
    Returns:
    --------
    bool
        True si la validation réussit
    """
    validation_results = {
        'config_valid': False,
        'paths_valid': False,
        'libraries_valid': False,
        'random_state_valid': False
    }
    
    try:
        # Validation de la configuration
        if hasattr(config, 'PROJECT_CONFIG') and config.PROJECT_CONFIG:
            validation_results['config_valid'] = True
        
        # Validation des chemins
        critical_paths = ['ROOT_DIR', 'RAW_DATA_DIR', 'DATA_PROCESSED']
        if all(path_name in paths for path_name in critical_paths):
            validation_results['paths_valid'] = True
        
        # Validation des bibliothèques
        core_libs = ['numpy', 'pandas', 'sklearn']
        if all(lib in sys.modules for lib in core_libs):
            validation_results['libraries_valid'] = True
        
        # Validation du random state
        import numpy as np
        if 'random_state' in metadata.get('configuration', {}):
            validation_results['random_state_valid'] = True
        
        # Résumé de validation
        all_valid = all(validation_results.values())
        
        print(f"\n🔍 Validation de l'environnement")
        print("-" * 35)
        for check, status in validation_results.items():
            icon = "✅" if status else "❌"
            print(f"  {icon} {check.replace('_', ' ').title()}")
        
        print(f"\n🎯 Statut global: {'✅ Prêt' if all_valid else '⚠️  Problèmes détectés'}")
        
        return all_valid
        
    except Exception as e:
        logger.error(f"Erreur lors de la validation: {e}")
        return False

def cleanup_environment():
    """Nettoie l'environnement et libère les ressources"""
    try:
        import matplotlib.pyplot as plt
        import gc
        
        # Fermeture des figures matplotlib
        plt.close('all')
        
        # Garbage collection
        gc.collect()
        
        logger.info("Environnement nettoyé")
        
    except Exception as e:
        logger.warning(f"Erreur lors du nettoyage: {e}")

# Test de la configuration si exécuté directement
if __name__ == "__main__":
    # Configuration pour test
    test_project_info = {
        'name': 'Test STA211',
        'author': 'Test User',
        'version': '1.0',
        'target_metric': 'F1-Score'
    }
    
    print("🧪 Test de la configuration d'environnement")
    
    try:
        config, paths, metadata = setup_environment(
            project_info=test_project_info,
            random_state=42,
            install_packages=False
        )
        
        # Validation
        is_valid = validate_environment_setup(config, paths, metadata)
        
        print(f"\n✅ Test terminé - Environnement {'valide' if is_valid else 'invalide'}")
        
    except Exception as e:
        print(f"❌ Erreur lors du test: {e}")
        raise