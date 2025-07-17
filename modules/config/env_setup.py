"""env_setup.py – Initialisation unique de l'environnement STA211
----------------------------------------------------------------
Prépare le contexte (local ou Colab) :
  • ajoute ROOT_DIR **et MODULE_DIR** au PYTHONPATH ;
  • crée/valide les dossiers de sortie ;
  • configure Pandas / Matplotlib ;
  • journalisation Rich si présent ;
  • export des versions pour traçabilité.
Un singleton interne évite la double exécution.
"""
from __future__ import annotations

import importlib
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Final

import numpy as np
import pandas as pd

# ───────────────────────── Logger ──────────────────────────
try:
    from rich.logging import RichHandler
    from rich.table import Table
    from rich.console import Console

    _rich_console = Console()

    logging.basicConfig(
        level="INFO",
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True, markup=True, console=_rich_console)],
        force=True,
    )
except ModuleNotFoundError:
    _rich_console = None  # type: ignore
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
log = logging.getLogger(__name__)

# ─────────────────────── Singleton guards ───────────────────────
_INITIALISED: bool = False
_CACHED_RESULT: Dict[str, Any] | None = None

# ───────────────────────── Constantes ───────────────────────────
RANDOM_STATE: Final[int] = 42
np.random.seed(RANDOM_STATE)

# ────────────────────────── Helpers ─────────────────────────────

def _find_project_root() -> Path:
    """Remonte les parents jusqu'à trouver un dossier contenant 'modules/'."""
    env_path = os.getenv("PROJET_STA211_PATH")
    if env_path and (Path(env_path) / "modules").exists():
        return Path(env_path).resolve()

    # Colab par défaut
    try:
        import google.colab  # type: ignore

        colab_default = Path("/content/drive/MyDrive/projet_sta211")
        if (colab_default / "modules").exists():
            return colab_default.resolve()
    except ImportError:
        pass

    for p in [Path.cwd(), *Path.cwd().parents]:
        if (p / "modules").exists():
            return p.resolve()
    raise FileNotFoundError("Racine projet introuvable : aucun dossier 'modules/' trouvé.")


def _create_dirs(paths: Dict[str, Path]) -> None:
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)


def set_display_options() -> None:
    pd.set_option("display.max_columns", 120)
    pd.set_option("display.width", 160)
    try:
        import matplotlib.pyplot as plt

        plt.rcParams.update({"figure.figsize": (8, 5), "figure.dpi": 110})
    except ModuleNotFoundError:
        pass


def display_paths(paths: Dict[str, Path]) -> None:
    """Affiche un tableau des chemins dans la console Rich ou en texte brut."""
    if _rich_console is not None:
        table = Table(title="Chemins du projet STA211", show_header=True, header_style="bold magenta")
        table.add_column("Clé", style="cyan", no_wrap=True)
        table.add_column("Chemin", style="green")
        for k, v in paths.items():
            table.add_row(k, str(v))
        _rich_console.print(table)
    else:
        log.info("\n" + "\n".join(f"{k:<20}: {v}" for k, v in paths.items()))

# ─────────────────────── init_project() ─────────────────────────

def init_project() -> Dict[str, Any]:
    """Initialise le projet STA211 (exécuté une seule fois par session)."""

    global _INITIALISED, _CACHED_RESULT

    if _INITIALISED and _CACHED_RESULT is not None:
        log.info("🔄 init_project() déjà exécuté – cache réutilisé.")
        return _CACHED_RESULT

    # 1. Localiser ROOT_DIR et l'ajouter au PYTHONPATH
    root = _find_project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
        log.info(f"PYTHONPATH ← {root}")

    # 1.b Ajouter MODULE_DIR (root / 'modules')
    module_dir = root / "modules"
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))
        log.info(f"PYTHONPATH ← {module_dir}")

    # 2. Dossiers clés
    data_dir = root / "data"
    outputs_dir = root / "outputs"

    paths: Dict[str, Path] = {
        "ROOT_DIR": root,
        "MODULE_DIR": module_dir,
        "RAW_DATA_DIR": data_dir / "raw",
        "DATA_PROCESSED": data_dir / "processed",
        "MODELS_DIR": root / "models",
        "FIGURES_DIR": outputs_dir / "figures",
        "OUTPUTS_DIR": outputs_dir,
        #"FIGURES_MODELING_DIR": outputs_dir / "figures" / "modeling",
        "THRESHOLDS_DIR": outputs_dir / "modeling" / "thresholds",
    }
    _create_dirs(paths)

    # 3. Versions librairies principales
    def _v(pkg: str) -> str:
        try:
            return importlib.import_module(pkg).__version__  # type: ignore[attr-defined]
        except Exception:
            return "–"

    versions = {k: _v(k) for k in ("pandas", "numpy", "sklearn", "xgboost", "imblearn", "catboost")}
    for lib, ver in versions.items():
        log.info(f"· {lib:<10}: {ver}")

    # 4. Options d'affichage
    set_display_options()

    # 5. Cache & retour
    _CACHED_RESULT = {
        "paths": paths,
        "RANDOM_STATE": RANDOM_STATE,
        "LIB_VERSIONS": versions,
        "PROJECT_ROOT": root,
        "TIMESTAMP": datetime.now().isoformat(timespec="seconds"),
    }
    _INITIALISED = True
    log.info("✅ init_project() terminé.")
    return _CACHED_RESULT

# ───────────────────────── Fin du module ─────────────────────────
