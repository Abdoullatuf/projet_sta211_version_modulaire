# config/paths_config.py
"""
Configuration des chemins pour le projet STA211.
Remplace l'ancien project_setup.py
"""
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
    Initialise les chemins du projet selon l'environnement (Colab ou local),
    crée les répertoires nécessaires, ajoute MODULE_DIR au sys.path,
    et retourne les chemins utiles.
    """
    # Détection de l'environnement
    if is_colab():
        # Montage de Google Drive uniquement si non monté
        if not Path("/content/drive").exists():
            from google.colab import drive
            drive.mount('/content/drive', force_remount=True)
        root_dir = Path("/content/drive/MyDrive/projet_sta211")
    else:
        root_dir = Path("G:/Mon Drive/projet_sta211")

    # Définition des chemins utiles
    paths = {
        "ROOT_DIR": root_dir,
        "MODULE_DIR": root_dir / "modules",
        "RAW_DATA_DIR": root_dir / "data" / "raw",
        "DATA_PROCESSED": root_dir / "data" / "processed",
        "MODELS_DIR": root_dir / "models",
        "FIGURES_DIR": root_dir / "outputs" / "figures"
    }

    # Création des dossiers si nécessaires
    for folder_path in paths.values():
        folder_path.mkdir(parents=True, exist_ok=True)

    # Ajout de MODULE_DIR au sys.path si absent
    module_dir = paths["MODULE_DIR"]
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))

    return paths

# Mode exécution directe
if __name__ == "__main__":
    paths = setup_project_paths()
    print("📁 Chemins configurés :")
    for name, path in paths.items():
        print(f"  {name}: {path}")
