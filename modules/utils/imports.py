# utils/imports.py
"""
Imports centralisés pour le projet STA211.
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

# Imports optionnels
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

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
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    xgb = None

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    lgb = None

# === Utilitaires ===
import joblib
from scipy import stats
from scipy.stats import yeojohnson

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

print("✅ Imports chargés avec succès")
if XGB_AVAILABLE:
    print("  ✓ XGBoost disponible")
if LGB_AVAILABLE:
    print("  ✓ LightGBM disponible")
if PLOTLY_AVAILABLE:
    print("  ✓ Plotly disponible")