# modules/config/display_config.py
"""
display_config.py
Options d’affichage (pandas, matplotlib) et thème couleur projet.
"""

from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt

__all__ = ["set_display_options"]

def set_display_options() -> None:
    # pandas
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", 120)
    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.precision", 4)

    # matplotlib : thème simple et police taille 11
    plt.rcParams.update(
        {
            "figure.figsize": (8, 5),
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "font.size": 11,
            "grid.linestyle": "--",
            "grid.alpha": 0.4,
        }
    )
