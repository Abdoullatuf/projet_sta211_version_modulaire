# project_setup.py
import os
import sys
from pathlib import Path
from typing import Dict

def is_colab() -> bool:
    """Détecte si le code s'exécute dans Google Colab."""
    try:
        import google.colab
        return True
    except ImportError:
        return False

def setup_project_paths() -> Dict[str, Path]:
    """
    Initialise les chemins du projet selon l'environnement,
    crée les répertoires nécessaires, et retourne les chemins utiles.
    Note: Google Drive doit déjà être monté si on est sur Colab.
    """
    
    # Détection automatique de l'environnement
    if is_colab():
        # On suppose que Drive est déjà monté
        root_dir = Path("/content/drive/MyDrive/projet_sta211")
        if not root_dir.exists():
            print("⚠️ Le dossier projet n'existe pas dans Drive. Création...")
            root_dir.mkdir(parents=True, exist_ok=True)
    else:
        root_dir = Path("G:/Mon Drive/projet_sta211")
    
    paths = {
        "ROOT_DIR": root_dir,
        "MODULE_DIR": root_dir / "modules",
        "RAW_DATA_DIR": root_dir / "data" / "raw",
        "DATA_PROCESSED": root_dir / "data" / "processed",
        "MODELS_DIR": root_dir / "models",
        "FIGURES_DIR": root_dir / "outputs" / "figures"
    }
    
    # Crée les dossiers s'ils n'existent pas
    for folder_path in paths.values():
        folder_path.mkdir(parents=True, exist_ok=True)
    
    # Ajoute le dossier modules au sys.path si nécessaire
    module_dir = paths["MODULE_DIR"]
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))
    
    return paths

if __name__ == "__main__":
    paths = setup_project_paths()
    print(paths)

