"""imports_sta211.py – Hub d'importations pour les notebooks STA211
-----------------------------------------------------------------
Un simple :
    from imports_sta211 import *
met à disposition les bibliothèques usuelles (NumPy/Pandas, visualisation,
scikit‑learn, imbalanced‑learn, XGBoost, CatBoost, etc.) et applique quelques
réglages de confort :
  • style Matplotlib + options Pandas ;
  • warnings filtrés ;
  • constantes `RANDOM_STATE` et `N_SPLITS`.

Les noms exposés sont contrôlés via `__all__` pour éviter de polluer le
namespace global.  Ajoutez ou retirez les packages à votre convenance.
"""
# ==========================================================================
# BIBLIOTHÈQUES STANDARD
# ==========================================================================
import logging
import warnings
from pathlib import Path
from typing import List, Tuple, Optional, Union

# Suppression de l'avertissement tqdm/ipywidgets
warnings.filterwarnings("ignore", message="IProgress not found")

# --------------------------------------------------------------------------
log = logging.getLogger(__name__)

# ==========================================================================
# DATA & NUMPY STACK
# ==========================================================================
import numpy as np
import pandas as pd

# ==========================================================================
# VISUALISATION
# ==========================================================================
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================================================
# MACHINE LEARNING – SCIKIT‑LEARN + EXTENSIONS
# ==========================================================================
import sklearn
from sklearn.model_selection import (
    StratifiedKFold,
    GridSearchCV,
    RandomizedSearchCV,
)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    make_scorer,
    roc_auc_score,
)
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import RFECV, SelectKBest, f_classif

# ==========================================================================
# IMBALANCED‑LEARN
# ==========================================================================
import imblearn
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, BorderlineSMOTE

# ==========================================================================
# GRADIENT BOOSTING & AUTRES LIBS
# ==========================================================================
import xgboost as xgb
from xgboost import XGBClassifier
# try:
#     from catboost import CatBoostClassifier
#except ModuleNotFoundError:
#    CatBoostClassifier = None  # type: ignore

import scipy
import joblib
# Import tqdm avec gestion d'erreur pour éviter l'avertissement ipywidgets
try:
    from tqdm.auto import tqdm
except ImportError:
    from tqdm import tqdm

# ==========================================================================
# CONFIGURATION GRAPHIQUE
# ==========================================================================
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "figure.figsize": (10, 6),
        "font.size": 10,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "legend.fontsize": 10,
        "figure.dpi": 100,
    }
)
sns.set_palette("husl")
sns.set_context("notebook")

# ==========================================================================
# OPTIONS D'AFFICHAGE PANDAS
# ==========================================================================
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 100)
pd.set_option("display.width", 1200)
pd.set_option("display.float_format", "{:.4f}".format)

# ==========================================================================
# WARNINGS
# ==========================================================================
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.exceptions import ConvergenceWarning, FitFailedWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FitFailedWarning)

# ==========================================================================
# RNG SEED GLOBAL & VALIDATION CONFIG
# ==========================================================================
RANDOM_STATE: int = 42
N_SPLITS: int = 5
np.random.seed(RANDOM_STATE)

# ==========================================================================
# # FONCTIONS UTILITAIRES
# # ==========================================================================

# def print_shape_change(df_before: pd.DataFrame, df_after: pd.DataFrame, operation: str) -> None:
#     """Affiche le changement de dimensions après une opération de pré‑traitement."""
#     diff = df_before.shape[0] - df_after.shape[0]
#     print(f"\n{operation}: {df_before.shape} → {df_after.shape} (−{diff} lignes)")

# ==========================================================================
# IMPORT DES FONCTIONS D'OPTIMISATION DE SEUIL
# ==========================================================================
#  Import retiré pour éviter les imports circulaires
#  Utiliser directement : from modules.modeling.optimize_threshold import optimize_threshold

# Fonctions placeholder pour éviter les erreurs
def optimize_threshold(*args, **kwargs):
    """Fonction placeholder - importer directement depuis modules.modeling.optimize_threshold"""
    raise ImportError("Utiliser directement: from modules.modeling.optimize_threshold import optimize_threshold")

def optimize_multiple(*args, **kwargs):
    """Fonction placeholder - importer directement depuis modules.modeling.optimize_threshold"""
    raise ImportError("Utiliser directement: from modules.modeling.optimize_threshold import optimize_multiple")

def optimize_multiple_models(*_, **__):  # type: ignore
    """Fonction placeholder - importer directement depuis modules.modeling.optimize_threshold_extended"""
    raise ImportError("Utiliser directement: from modules.modeling.optimize_threshold_extended import optimize_multiple_models")

# ==========================================================================
# __all__  – symboles exportés par `from imports_sta211 import *`
# ==========================================================================
__all__ = [
    # Data & viz
    "np", "pd", "matplotlib", "plt", "sns",
    # ML core
    "sklearn", "StratifiedKFold", "GridSearchCV", "RandomizedSearchCV",
    "StandardScaler", "RandomForestClassifier", "StackingClassifier",
    "LogisticRegression", "SVC", "MLPClassifier", "DecisionTreeClassifier",
    "KNeighborsClassifier",
    # Metrics & utils
    "classification_report", "confusion_matrix", "f1_score", "accuracy_score",
    "precision_score", "recall_score", "make_scorer", "roc_auc_score",
    "permutation_importance", "RFECV", "SelectKBest", "f_classif",
    # Imbalanced‑learn
    "imblearn", "ImbPipeline", "SMOTE", "BorderlineSMOTE",
    # Boosters & autres
    "xgb", "XGBClassifier", 
    #"CatBoostClassifier",
    "scipy", "joblib", "tqdm",
    # Constantes & helpers
    "RANDOM_STATE", "N_SPLITS", 
    #"print_shape_change",
    # Fonctions seuil - importées directement si nécessaire
    # "optimize_threshold", "optimize_multiple", "optimize_multiple_models",
]
