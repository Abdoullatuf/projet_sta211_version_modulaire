# modules/config/project_config.py
"""
project_config.py
Objet de configuration central pour la modélisation STA211.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import json, datetime

__all__ = ["ProjectConfig", "create_config"]

@dataclass
class ProjectConfig:
    project_name : str = "STA211 Ads"
    author       : str = "Maoulida Abdoullatuf"
    version      : str = "1.3"
    random_state : int = 42
    test_size    : float = 0.2
    scoring      : str = "f1"
    cv           : int = 5
    model_dir    : Path | None = None

    # -------- utilitaires
    def to_dict(self) -> dict:         return asdict(self)
    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, ensure_ascii=False)
    @classmethod
    def load_json(cls, path: str | Path) -> "ProjectConfig":
        with Path(path).open(encoding="utf-8") as fh:
            data = json.load(fh)
        return cls(**data)
    def tag(self) -> str:
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"{self.project_name.replace(' ', '_')}_{self.version}_{ts}"

# -------- helper “facultatif” pour créer puis sauvegarder
def create_config(**kwargs) -> ProjectConfig:
    cfg = ProjectConfig(**kwargs)
    return cfg