# modules/data_preprocessing.py

"""
Module de prétraitement des données pour le projet STA211.
Ce module contient des fonctions pour :
  - détecter l'environnement Colab (facultatif)
  - charger et nettoyer un CSV,
  - analyser les valeurs manquantes,
  - rechercher k optimal pour KNN,
  - imputer les valeurs manquantes (simple, KNN, multiple).
"""

import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Union, List

# Visualisation
import matplotlib.pyplot as plt
from IPython.display import display

# Pour l'imputation multiple (MICE)
from sklearn.experimental import enable_iterative_imputer  # ⚠️ doit précéder l'import suivant
from sklearn.impute import KNNImputer, IterativeImputer, SimpleImputer

# Autres outils
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Chargement des chemins du projet de manière dynamique
from project_setup import setup_project_paths
paths = setup_project_paths()
RAW_DATA_DIR = str(paths["RAW_DATA_DIR"])

__all__ = [
    'is_colab',
    'clean_data',
    'load_data',
    'analyze_missing_values',
    'find_optimal_k',
    'handle_missing_values'
]


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie un DataFrame :
      - strip + supprime guillemets des noms de colonnes,
      - strip + supprime guillemets des valeurs string,
      - gère les colonnes dupliquées finissant par '.1' de façon sécurisée.
    """
    df = df.copy()
    df.columns = df.columns.str.strip().str.replace('"', '')

    # Nettoyage des valeurs string
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip().str.replace('"', '')

    # Gérer le cas où seule 'X4.1' existe (et pas 'X4')
    for base_col in ['X4', 'X1', 'X2', 'X3']:
        if f"{base_col}.1" in df.columns and base_col not in df.columns:
            print(f"⚠️ Renommage de {base_col}.1 en {base_col} (pas d'autre version détectée)")
            df = df.rename(columns={f"{base_col}.1": base_col})

    # Suppression des colonnes dupliquées (garde X4 si présent)
    cols_with_dot1 = [
        col for col in df.columns
        if col.endswith('.1') and col[:-2] in df.columns
    ]
    if cols_with_dot1:
        print(f"🚨 Colonnes dupliquées supprimées : {cols_with_dot1}")
        df = df.drop(columns=cols_with_dot1)

    return df






def load_data(
    file_path: Union[str, Path],
    require_outcome: bool = True,
    display_info: bool = True,
    raw_data_dir: Optional[Union[str, Path]] = None,
    encode_target: bool = False  # Nouvelle option
) -> pd.DataFrame:
    """
    Charge un fichier CSV, nettoie les données et affiche un aperçu.

    - file_path : chemin absolu ou relatif vers un fichier (str ou Path).
    - require_outcome : True pour exiger la colonne 'outcome'.
    - display_info : affiche les infos (info, head...).
    - raw_data_dir : chemin de base pour fichiers relatifs.
    - encode_target : si True, encode 'outcome' en numérique (ad. -> 1, noad. -> 0).
    """
    # Convertit en string au cas où on a reçu un Path
    file_path = Path(file_path)

    if raw_data_dir is None:
        paths = setup_project_paths()
        raw_data_dir = paths["RAW_DATA_DIR"]
    else:
        raw_data_dir = Path(raw_data_dir)

    # Résolution du chemin réel
    if file_path.is_absolute():
        real_path = file_path
    else:
        real_path = raw_data_dir / file_path

    if not real_path.exists():
        raise FileNotFoundError(f"📂 Fichier introuvable : {real_path}")

    # Lecture CSV
    try:
        df = pd.read_csv(real_path, sep='\t')
        if df.shape[1] == 1:
            df = pd.read_csv(real_path, sep=',')
    except Exception as e:
        raise RuntimeError(f"❌ Erreur lecture {real_path}: {e}")

    # Vérifie outcome
    if require_outcome and 'outcome' not in df.columns:
        raise ValueError("🚨 La colonne 'outcome' est manquante.")

    # Nettoyage
    df = clean_data(df)

    # Encodage de la cible si requis
    if require_outcome and encode_target and 'outcome' in df.columns:
        if set(df['outcome'].unique()).issubset({'ad.', 'noad.'}):
            df['outcome'] = df['outcome'].map({'ad.': 1, 'noad.': 0})
            print("✅ Colonne 'outcome' encodée en numérique (ad. -> 1, noad. -> 0)")
        else:
            raise ValueError(f"❌ Valeurs inattendues dans 'outcome': {df['outcome'].unique()}")

    # Affichage
    if display_info:
        print(f"\n✅ Fichier chargé : {real_path}")
        print(f"🔢 Dimensions : {df.shape}")
        print("📋 Infos colonnes :")
        df.info()
        print("\n🔎 Premières lignes :")
        display(df.head())

    return df



def analyze_missing_values(df: pd.DataFrame) -> dict:
    """
    Analyse des valeurs manquantes :
      - Total et % global
      - Par colonne : high (>30%), medium (5-30%), low (≤5%)
      - Top 5 colonnes manquantes
    Retourne un dict de statistiques.
    """
    total = df.size
    miss = df.isnull().sum().sum()
    pct = miss / total * 100

    by_col = df.isnull().sum()
    by_col = by_col[by_col > 0]
    pct_col = by_col / len(df) * 100

    high = pct_col[pct_col > 30]
    med  = pct_col[(pct_col > 5) & (pct_col <= 30)]
    low  = pct_col[pct_col <= 5]

    print(f"Total missing       : {miss} ({pct:.2f}%)")
    print(f"Colonnes affectées  : {len(by_col)} "
          f"(haut: {len(high)}, moyen: {len(med)}, bas: {len(low)})")
    print("Top 5 colonnes manquantes :")
    print(pct_col.sort_values(ascending=False).head())

    return {
        'total_missing':      int(miss),
        'percent_missing':    pct,
        'cols_missing':       by_col.to_dict(),
        'percent_per_col':    pct_col.to_dict(),
        'high_missing':       high.to_dict(),
        'medium_missing':     med.to_dict(),
        'low_missing':        low.to_dict(),
    }


def find_optimal_k(
    data: pd.DataFrame,
    continuous_cols: list[str],
    k_range: range = range(1, 21),
    cv_folds: int = 5,
    sample_size: int = 1000
) -> int:
    """
    Trouve k optimal pour KNNImputer via CV sur un petit échantillon.
    Affiche la courbe MSE vs k, et renvoie le k minimisant le MSE.
    """
    X = data[continuous_cols].copy()
    if len(X) > sample_size:
        X = X.sample(n=sample_size, random_state=42)

    rng = np.random.RandomState(42)
    mask = rng.rand(*X.shape) < 0.2
    X_missing = X.copy()
    X_missing[mask] = np.nan

    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    mse_scores = []

    for k in k_range:
        fold_mse = []
        imputer = KNNImputer(n_neighbors=k)
        for train_idx, val_idx in cv.split(X):
            X_tr   = X_missing.iloc[train_idx]
            X_val  = X_missing.iloc[val_idx]
            X_true = X.iloc[val_idx]

            imputer.fit(X_tr)
            X_imp = pd.DataFrame(
                imputer.transform(X_val),
                columns=X_val.columns,
                index=X_val.index
            )

            mask_val = mask[val_idx]
            y_true = X_true.to_numpy(dtype=float)[mask_val]
            y_pred = X_imp.to_numpy(dtype=float)[mask_val]
            valid = (~np.isnan(y_true)) & (~np.isnan(y_pred))
            if valid.any():
                fold_mse.append(
                    mean_squared_error(y_true[valid], y_pred[valid])
                )

        mse_scores.append(np.mean(fold_mse))

    # Plot
    plt.figure(figsize=(8, 4))
    plt.plot(list(k_range), mse_scores, marker='o')
    plt.xlabel('k (nombre de voisins)')
    plt.ylabel('Mean Squared Error')
    plt.title('KNN Imputation Performance')
    plt.grid(True)
    plt.show()

    best_k = int(k_range[np.argmin(mse_scores)])
    print(f"→ k optimal : {best_k}")
    return best_k


def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = 'mixed_mar_mcar',
    mar_method: str = 'knn',
    knn_k: Optional[int] = None,
    mar_cols: Optional[List[str]] = None,
    mcar_cols: Optional[List[str]] = None,
    display_info: bool = True,
    save_results: bool = True,
    processed_data_dir: Optional[str] = None,
    models_dir: Optional[str] = None
) -> pd.DataFrame:
    """
    Gère les valeurs manquantes dans un DataFrame selon différentes stratégies :
    - Imputation par médiane globale
    - Imputation mixte MAR / MCAR avec KNN ou MICE + médiane

    Args :
        df : DataFrame à traiter
        strategy : 'all_median' ou 'mixed_mar_mcar'
        mar_method : méthode d'imputation pour MAR ('knn' ou 'mice')
        knn_k : nombre de voisins pour KNN (si méthode knn)
        mar_cols : liste des colonnes MAR (ex : ['X1_trans', 'X2_trans', 'X3_trans'])
        mcar_cols : liste des colonnes MCAR à imputer par médiane
        ...

    Returns :
        df_proc : DataFrame imputé
    """
    df_proc = df.copy()
    suffix = ''

    # STRATEGY 1 : tout par médiane
    if strategy == 'all_median':
        num_cols = df_proc.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            df_proc[col] = df_proc[col].fillna(df_proc[col].median())
        suffix = 'median_all'
        if display_info:
            print(f"→ Imputation par médiane sur {len(num_cols)} colonnes numériques.")

    # STRATEGY 2 : MAR + MCAR séparés
    elif strategy == 'mixed_mar_mcar':
        if mar_cols is None:
            mar_cols = ['X1_trans', 'X2_trans', 'X3_trans']
        if mcar_cols is None:
            mcar_cols = ['X4']

        if all(c in df_proc.columns for c in mar_cols):
            if mar_method == 'knn':
                if knn_k is None:
                    knn_k = 5
                    if display_info:
                        print("⚠️ k non spécifié, utilisation de k = 5 par défaut.")
                imputer = KNNImputer(n_neighbors=knn_k)
                df_proc[mar_cols] = imputer.fit_transform(df_proc[mar_cols])
                suffix = f'knn_k{knn_k}'

            elif mar_method == 'mice':
                imputer = IterativeImputer(random_state=42, max_iter=10)
                df_proc[mar_cols] = imputer.fit_transform(df_proc[mar_cols])
                suffix = 'mice'

            else:
                raise ValueError("❌ mar_method doit être 'knn' ou 'mice'.")

            if save_results and models_dir:
                os.makedirs(models_dir, exist_ok=True)
                imp_path = os.path.join(models_dir, f"imputer_{suffix}.pkl")
                joblib.dump(imputer, imp_path)

            if display_info:
                print(f"✅ Imputation {mar_method} appliquée sur : {mar_cols}")

        # Imputation médiane pour MCAR
        for col in mcar_cols:
            if col in df_proc.columns:
                val = df_proc[col].median()
                df_proc[col] = df_proc[col].fillna(val)
                if display_info:
                    print(f"→ Médiane imputée pour {col} : {val:.4f}")

    else:
        raise ValueError("❌ strategy doit être 'all_median' ou 'mixed_mar_mcar'.")

    # Sauvegarde
    if save_results:
        if processed_data_dir is None:
            raise ValueError("processed_data_dir doit être fourni si save_results=True.")
        os.makedirs(processed_data_dir, exist_ok=True)
        filename = f"df_imputed_{suffix}.csv"
        filepath = os.path.join(processed_data_dir, filename)
        df_proc.to_csv(filepath, index=False)
        if display_info:
            print(f"✔ Données imputées sauvegardées dans '{filepath}'")

    return df_proc



def plot_imputation_comparison(
    df_raw: pd.DataFrame,
    df_imputed: pd.DataFrame,
    columns: list,
    method_name: str = "imputation",
    save_path: Optional[Union[str, Path]] = None,
    k: Optional[int] = None,
    figsize: tuple = (10, 2.5)
):
    """
    Compare les distributions avant/après imputation pour une liste de colonnes.

    Args:
        df_raw (pd.DataFrame): DataFrame contenant les NaN
        df_imputed (pd.DataFrame): DataFrame après imputation
        columns (list): liste des colonnes à comparer
        method_name (str): nom de la méthode ("KNN", "MICE", etc.)
        save_path (str ou Path): chemin du fichier .png à sauvegarder
        k (int): valeur de k si méthode = KNN (optionnel, affiché dans le titre)
        figsize (tuple): taille d’un subplot (par variable)

    Returns:
        None
    """
    n = len(columns)
    fig, axes = plt.subplots(n, 2, figsize=(figsize[0], figsize[1] * n))

    for i, col in enumerate(columns):
        # Avant imputation
        axes[i, 0].hist(df_raw[col].dropna(), bins=30, color="gray", alpha=0.6)
        axes[i, 0].set_title(f"{col} — avant imputation")

        # Après imputation
        axes[i, 1].hist(df_imputed[col], bins=30, color="skyblue", alpha=0.9)
        if method_name.lower() == "knn" and k is not None:
            title = f"{col} — après imputation {method_name} (k={k})"
        else:
            title = f"{col} — après imputation {method_name}"
        axes[i, 1].set_title(title)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"✅ Graphique enregistré dans : {save_path}")
    plt.show()


