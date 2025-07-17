"""
paths_config.py
Créer et renvoyer un dictionnaire de chemins Project-friendly.
"""

from __future__ import annotations
from pathlib import Path

__all__ = ["setup_project_paths", "is_colab"]  # Ajout explicite d'is_colab

def setup_project_paths(root_dir: str | Path | None = None) -> dict[str, Path]:
    root = Path(root_dir).expanduser().resolve() if root_dir else Path(__file__).resolve().parents[2]
    paths = {
        "ROOT_DIR": root,
        "MODULE_DIR": root / "modules",
        "RAW_DATA_DIR": root / "data" / "raw",
        "DATA_PROCESSED": root / "data" / "processed",
        "MODELS_DIR": root / "models",
        "FIGURES_DIR": root / "outputs" / "figures",
        "OUTPUTS_DIR": root / "outputs"
    }
    # Création physique des dossiers si besoin
    for p in paths.values():
        if p.suffix:  # Ignore les fichiers
            continue
        p.mkdir(parents=True, exist_ok=True)
    return paths

def is_colab() -> bool:
    """Détecte si l'on est dans un environnement Google Colab."""
    try:
        import google.colab
        return True
    except ImportError:
        return False