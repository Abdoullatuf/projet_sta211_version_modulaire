

#!/usr/bin/env python3

# modules/config/test_config.py

"""
Script de test pour valider les modules de configuration du projet STA211

Ce script teste tous les modules de configuration et valide que
l'environnement peut être configuré correctement.

Usage:
    python test_config.py
"""

import sys
import logging
from pathlib import Path
import traceback

# Configuration du logging pour les tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_paths_config():
    """Test du module paths_config"""
    print("\n🧪 Test: modules.config.paths_config")
    print("-" * 40)
    
    try:
        from paths_config import (
            detect_environment, 
            setup_project_paths,
            get_data_files_paths,
            validate_paths
        )
        
        # Test détection environnement
        env = detect_environment()
        print(f"  ✅ Environnement détecté: {env}")
        
        # Test configuration chemins (sans création)
        paths = setup_project_paths(env, create_structure=False)
        print(f"  ✅ Chemins configurés: {len(paths)} chemins")
        
        # Test validation
        validation = validate_paths(paths)
        print(f"  ✅ Validation chemins: {len(validation)} chemins validés")
        
        # Test chemins fichiers
        data_files = get_data_files_paths(paths)
        print(f"  ✅ Fichiers de données: {len(data_files)} fichiers")
        
        return True, paths
        
    except Exception as e:
        print(f"  ❌ Erreur: {e}")
        traceback.print_exc()
        return False, None

def test_display_config():
    """Test du module display_config"""
    print("\n🧪 Test: modules.config.display_config")
    print("-" * 40)
    
    try:
        from display_config import (
            setup_display_config,
            get_color_palette,
            create_custom_style
        )
        
        # Test configuration affichage
        display_config = setup_display_config()
        print(f"  ✅ Configuration affichage: {len(display_config)} modules")
        
        # Test palette couleurs
        colors = get_color_palette("project", 5)
        print(f"  ✅ Palette couleurs: {len(colors)} couleurs")
        
        # Test style personnalisé
        style = create_custom_style()
        print(f"  ✅ Style personnalisé: {len(style)} paramètres")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Erreur: {e}")
        traceback.print_exc()
        return False

def test_project_config():
    """Test du module project_config"""
    print("\n🧪 Test: modules.config.project_config")
    print("-" * 40)
    
    try:
        from project_config import (
            ProjectConfig,
            create_config
        )
        
        # Test création config
        test_paths = {
            'ROOT_DIR': Path('.'),
            'DATA_PROCESSED': Path('./data/processed'),
            'FIGURES_DIR': Path('./figures')
        }
        
        config = create_config(
            project_name="Test STA211",
            version="1.0", 
            author="Test",
            paths=test_paths
        )
        print(f"  ✅ Configuration créée: {type(config).__name__}")
        
        # Test accès valeurs
        test_size = config.get('PROJECT_CONFIG.TEST_SIZE')
        print(f"  ✅ Accès valeur: TEST_SIZE = {test_size}")
        
        # Test mise à jour
        config.update('PROJECT_CONFIG.SCORING', 'f1')
        scoring = config.get('PROJECT_CONFIG.SCORING')
        print(f"  ✅ Mise à jour: SCORING = {scoring}")
        
        return True, config
        
    except Exception as e:
        print(f"  ❌ Erreur: {e}")
        traceback.print_exc()
        return False, None

def test_environment_module():
    """Test du module environment"""
    print("\n🧪 Test: modules.config.environment")
    print("-" * 40)
    
    try:
        from environment import (
            detect_environment,
            import_core_libraries,
            import_optional_libraries,
            setup_random_seeds,
            verify_environment
        )
        
        # Test détection environnement
        env = detect_environment()
        print(f"  ✅ Environnement: {env}")
        
        # Test import bibliothèques principales
        core_libs = import_core_libraries()
        print(f"  ✅ Bibliothèques principales: {len(core_libs)}")
        
        # Test import bibliothèques optionnelles
        opt_libs = import_optional_libraries()
        available_count = sum(1 for k, v in opt_libs.items() 
                            if k.endswith('_AVAILABLE') and v)
        print(f"  ✅ Bibliothèques optionnelles: {available_count} disponibles")
        
        # Test graines aléatoires
        seed = setup_random_seeds(42)
        print(f"  ✅ Graines aléatoires: {seed}")
        
        # Test vérification environnement
        env_info = verify_environment()
        print(f"  ✅ Informations système: {len(env_info)} items")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Erreur: {e}")
        traceback.print_exc()
        return False

def test_full_setup():
    """Test de la configuration complète"""
    print("\n🧪 Test: Configuration complète")
    print("-" * 40)
    
    try:
        from environment import setup_environment
        
        project_info = {
            'name': 'Test STA211',
            'author': 'Test User',
            'version': '1.0',
            'target_metric': 'F1-Score'
        }
        
        config, paths, metadata = setup_environment(
            project_info=project_info,
            random_state=42,
            install_packages=False,  # Pas d'installation pour les tests
            quiet=True
        )
        
        print(f"  ✅ Configuration complète: {type(config).__name__}")
        print(f"  ✅ Chemins: {len(paths)} chemins")
        print(f"  ✅ Métadonnées: {len(metadata)} sections")
        
        # Test validation
        from environment import validate_environment_setup
        is_valid = validate_environment_setup(config, paths, metadata)
        print(f"  ✅ Validation: {'Succès' if is_valid else 'Échec'}")
        
        return True, (config, paths, metadata)
        
    except Exception as e:
        print(f"  ❌ Erreur: {e}")
        traceback.print_exc()
        return False, None

def test_config_integration():
    """Test d'intégration des modules config"""
    print("\n🧪 Test: Intégration modules config")
    print("-" * 40)
    
    try:
        # Import des modules individuels
        from environment import setup_environment, detect_environment
        from project_config import ProjectConfig, create_config
        from paths_config import setup_project_paths
        from display_config import setup_display_config
        
        print(f"  ✅ Import modules: tous les modules disponibles")
        
        # Test workflow complet
        env = detect_environment()
        paths = setup_project_paths(env, create_structure=False)
        display_config = setup_display_config()
        
        project_config = create_config(
            project_name="Integration Test",
            version="1.0",
            author="Test",
            paths=paths
        )
        
        print(f"  ✅ Workflow complet: configuration intégrée")
        return True
        
    except Exception as e:
        print(f"  ❌ Erreur: {e}")
        traceback.print_exc()
        return False

def test_config_persistence():
    """Test de la persistance de configuration"""
    print("\n🧪 Test: Persistance configuration")
    print("-" * 40)
    
    try:
        from project_config import create_config
        import tempfile
        import json
        
        # Création config temporaire
        test_paths = {'ROOT_DIR': Path('.')}
        config = create_config("Test", "1.0", "Test", test_paths)
        
        # Test sauvegarde
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = Path(f.name)
        
        config.save_config(temp_path)
        print(f"  ✅ Sauvegarde: {temp_path.name}")
        
        # Test chargement
        with open(temp_path, 'r') as f:
            loaded_data = json.load(f)
        
        print(f"  ✅ Chargement: {len(loaded_data)} sections")
        
        # Nettoyage
        temp_path.unlink()
        print(f"  ✅ Nettoyage: fichier temporaire supprimé")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Erreur: {e}")
        traceback.print_exc()
        return False

def run_all_tests():
    """Exécute tous les tests de configuration"""
    print("🔬 Début des tests des modules de configuration")
    print("=" * 60)
    
    results = {}
    test_data = {}
    
    # Tests individuels des modules
    results['paths_config'], test_data['paths'] = test_paths_config()
    results['display_config'] = test_display_config()
    results['project_config'], test_data['config'] = test_project_config()
    results['environment'] = test_environment_module()
    
    # Tests d'intégration
    results['full_setup'], test_data['full'] = test_full_setup()
    results['integration'] = test_config_integration()
    results['persistence'] = test_config_persistence()
    
    # Résumé des résultats
    print("\n📊 Résumé des tests")
    print("=" * 30)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} {test_name.replace('_', ' ').title()}")
        if result:
            passed += 1
    
    success_rate = (passed / total) * 100
    print(f"\n🎯 Résultat global: {passed}/{total} tests réussis ({success_rate:.1f}%)")
    
    if success_rate == 100:
        print("🎉 Tous les tests sont passés - Modules config prêts !")
        status = "SUCCESS"
    elif success_rate >= 80:
        print("⚠️  La plupart des tests passent - Configuration utilisable")
        status = "WARNING"
    else:
        print("❌ Plusieurs tests échouent - Vérifiez la configuration")
        status = "ERROR"
    
    return status, results, test_data

def main():
    """Fonction principale pour exécuter les tests"""
    try:
        status, results, test_data = run_all_tests()
        
        # Code de sortie
        exit_codes = {
            "SUCCESS": 0,
            "WARNING": 1, 
            "ERROR": 2
        }
        
        return exit_codes.get(status, 2)
        
    except Exception as e:
        print(f"\n💥 Erreur critique lors des tests: {e}")
        traceback.print_exc()
        return 3

if __name__ == "__main__":
    print("🚀 Script de test des modules de configuration STA211")
    print("=" * 60)
    
    # Vérification du répertoire courant
    current_dir = Path.cwd()
    print(f"📁 Répertoire courant: {current_dir}")
    
    # Vérification que nous sommes dans le bon dossier
    if current_dir.name != "config":
        print("⚠️  Attention: vous n'êtes pas dans le dossier config/")
        print("📍 Changez vers le dossier modules/config/ avant d'exécuter ce script")
    
    # Exécution des tests
    exit_code = main()
    
    print(f"\n🏁 Tests terminés (code de sortie: {exit_code})")
    sys.exit(exit_code)