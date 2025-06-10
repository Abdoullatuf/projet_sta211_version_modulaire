"""
Module d'import nettoyé pour le projet STA211 (version Colab-ready).
Contient uniquement les bibliothèques utilisées dans le notebook de modélisation.
"""

# === Bibliothèques standard ===
import os
import sys
import warnings
from pathlib import Path
from typing import List, Tuple, Optional, Union

# === Gestion des données ===
import pandas as pd
import numpy as np

# === Visualisation ===
import matplotlib.pyplot as plt
import seaborn as sns

# === Machine Learning - Prétraitement ===
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier, 
    StackingClassifier
)
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
    make_scorer
)

# === Imbalanced Learning ===
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# === Modèles avancés ===
import xgboost as xgb
from xgboost import XGBClassifier

import lightgbm as lgb
from lightgbm import LGBMClassifier

# === Utilitaires ===
from pathlib import Path
from tqdm.auto import tqdm

# === Configuration graphique et environnement ===
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 100

sns.set_palette("husl")
sns.set_context("notebook")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:.4f}'.format)

# === Warnings (silencieux) ===
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.exceptions import ConvergenceWarning, FitFailedWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FitFailedWarning)

# === Définition d'une seed globale ===
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# === Fonctions utilitaires ===
def print_shape_change(df_before: pd.DataFrame, df_after: pd.DataFrame, operation: str) -> None:
    """Affiche le changement de dimensions après une opération."""
    print(f"\n{operation}:")
    print(f"  Avant : {df_before.shape}")
    print(f"  Après : {df_after.shape}")
    print(f"  Lignes supprimées : {df_before.shape[0] - df_after.shape[0]}")
