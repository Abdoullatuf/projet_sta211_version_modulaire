# Test de la nouvelle structure
def test_new_structure():
    """Test que tous les imports fonctionnent"""
    print("🧪 Test de la nouvelle structure de modules...")
    
    try:
        # Test config
        from modules.config import setup_project_paths, ProjectConfig
        print("✅ Config importé")
        
        # Test utils
        from modules.utils import logger, timer, RANDOM_STATE
        print("✅ Utils importé")
        
        # Test exploration
        from modules.exploration import bivariate_analysis, save_fig
        print("✅ Exploration importé")
        
        # Test preprocessing
        from modules.preprocessing import load_data, analyze_missing_values
        print("✅ Preprocessing importé")
        
        print("\n🎉 Tous les modules fonctionnent correctement!")
        
    except ImportError as e:
        print(f"❌ Erreur d'import : {e}")

# Exécuter le test
test_new_structure()