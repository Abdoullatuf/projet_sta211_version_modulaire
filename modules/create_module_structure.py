"""
Script pour créer la structure de modules du projet STA211
"""
import os
from pathlib import Path

def create_module_structure(base_path="modules"):
    """
    Crée la structure complète des modules avec les fichiers __init__.py
    """
    # Structure des dossiers
    structure = {
        "config": ["__init__.py", "project_config.py", "paths_config.py"],
        "exploration": ["__init__.py", "eda_analysis.py", "visualization.py", "statistics.py"],
        "preprocessing": ["__init__.py", "data_loader.py", "missing_values.py", 
                         "transformations.py", "feature_engineering.py", "pipeline.py"],
        "modeling": ["__init__.py", "base_models.py", "ensemble.py", 
                    "evaluation.py", "optimization.py", "model_selection.py"],
        "utils": ["__init__.py", "imports.py", "helpers.py", "logger.py"]
    }
    
    # Créer le dossier principal
    base_dir = Path(base_path)
    base_dir.mkdir(exist_ok=True)
    
    # Créer __init__.py principal
    (base_dir / "__init__.py").touch()
    
    # Créer chaque sous-module
    for module, files in structure.items():
        module_dir = base_dir / module
        module_dir.mkdir(exist_ok=True)
        
        # Créer les fichiers
        for file in files:
            file_path = module_dir / file
            if not file_path.exists():
                file_path.touch()
                print(f"✅ Créé : {file_path}")
            else:
                print(f"⚠️  Existe déjà : {file_path}")
    
    print("\n✅ Structure de modules créée avec succès!")
    
    # Afficher la structure
    print("\n📁 Structure créée :")
    for root, dirs, files in os.walk(base_path):
        level = root.replace(base_path, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            print(f"{subindent}{file}")

if __name__ == "__main__":
    # Exécuter dans le dossier modules de votre projet
    create_module_structure()