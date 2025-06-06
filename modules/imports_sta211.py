# imports_sta211.py
"""
Module centralisé pour tous les imports du projet STA211.
Organise et configure l'environnement de travail.
"""

# === Bibliothèques standard ===
import os
import sys
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import pickle
from datetime import datetime

# === Gestion des données ===
import pandas as pd
import numpy as np
import sklearn


# === Visualisation ===
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
import plotly.graph_objects as go
import plotly.express as px

# === Machine Learning - Preprocessing ===
from sklearn.model_selection import (
    train_test_split, 
    StratifiedKFold, 
    cross_val_score,
    cross_validate,
    GridSearchCV, 
    RandomizedSearchCV
)
from sklearn.preprocessing import (
    StandardScaler, 
    MinMaxScaler,
    RobustScaler,
    PowerTransformer,
    LabelEncoder
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.feature_selection import (
    SelectKBest,
    f_classif,
    mutual_info_classif,
    RFE
)

# === Machine Learning - Modèles ===
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier,
    AdaBoostClassifier,
    VotingClassifier,
    StackingClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB

# === Machine Learning - Métriques ===
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score, 
    roc_curve,
    f1_score,
    accuracy_score, 
    precision_score, 
    recall_score, 
    make_scorer, 
    precision_recall_curve,
    average_precision_score,
    matthews_corrcoef
)

# === Machine Learning - Pipelines ===
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

# === Imbalanced Learning ===
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.combine import SMOTETomek, SMOTEENN

# === Modèles avancés ===
try:
    import xgboost as xgb
    print("✅ XGBoost importé")
except ImportError:
    print("⚠️ XGBoost non disponible")
    xgb = None

try:
    import lightgbm as lgb
    print("✅ LightGBM importé")
except ImportError:
    print("⚠️ LightGBM non disponible")
    lgb = None

# === Utilitaires ===
import joblib
from tqdm.auto import tqdm
from scipy import stats
from scipy.stats import yeojohnson

# === Modules projet (avec gestion d'erreur) ===
try:
    from eda_module_sta211 import *
    #from exploratory_analysis import save_fig
    from data_preprocessing import load_data, handle_missing_values
    from final_preprocessing import prepare_final_dataset
    from project_setup import setup_project_paths
    print("✅ Modules projet importés")
except ImportError as e:
    print(f"⚠️ Certains modules projet non disponibles : {e}")

# === Configuration globale ===
# Configuration matplotlib
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 100

# Configuration seaborn
sns.set_palette("husl")
sns.set_context("notebook")

# Configuration pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:.4f}'.format)

# Configuration warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.exceptions import ConvergenceWarning, FitFailedWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FitFailedWarning)

# === Seed pour la reproductibilité ===
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# === Fonctions utilitaires globales ===
def print_shape_change(df_before: pd.DataFrame, df_after: pd.DataFrame, operation: str) -> None:
    """Affiche le changement de dimensions après une opération."""
    print(f"\n{operation}:")
    print(f"  Avant : {df_before.shape}")
    print(f"  Après : {df_after.shape}")
    print(f"  Lignes supprimées : {df_before.shape[0] - df_after.shape[0]}")

def check_gpu_availability() -> None:
    """Vérifie la disponibilité du GPU pour XGBoost."""
    if xgb is not None:
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                print(f"✅ GPU détecté : {gpus[0].name}")
            else:
                print("⚠️ Aucun GPU détecté")
        except:
            print("ℹ️ GPUtil non installé - impossible de vérifier le GPU")

# === Message de confirmation ===
print("="*50)
print("✅ Imports STA211 chargés avec succès")
print(f"📊 Pandas version : {pd.__version__}")
print(f"🔢 NumPy version : {np.__version__}")
print(f"📈 Scikit-learn version : {sklearn.__version__}")
print(f"🌱 Random State : {RANDOM_STATE}")
print("="*50)

# Vérifier la disponibilité du GPU
check_gpu_availability()
