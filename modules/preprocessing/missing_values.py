#missing_values.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from scipy import stats
import joblib
import os





def analyze_missing_values(df: pd.DataFrame, columns: Optional[List[str]] = None, plot: bool = False) -> dict:
    """
    Analyse des valeurs manquantes sur l’ensemble ou un sous-ensemble des colonnes.

    Args:
        df (pd.DataFrame): Le DataFrame à analyser.
        columns (list, optional): Colonnes à considérer. Si None, toutes les colonnes sont utilisées.
        plot (bool, optional): Affiche un barplot des pourcentages de valeurs manquantes.

    Returns:
        dict: Statistiques résumant les valeurs manquantes.
    """
    df_check = df[columns] if columns else df.copy()

    total = df_check.size
    miss = df_check.isnull().sum().sum()
    pct = miss / total * 100

    by_col = df_check.isnull().sum()
    by_col = by_col[by_col > 0]
    pct_col = by_col / len(df_check) * 100

    high = pct_col[pct_col > 30]
    med = pct_col[(pct_col > 5) & (pct_col <= 30)]
    low = pct_col[pct_col <= 5]

    print(f"Total missing       : {miss} ({pct:.2f}%)")
    print(f"Colonnes affectées  : {len(by_col)} (haut: {len(high)}, moyen: {len(med)}, bas: {len(low)})")
    print("Top 5 colonnes manquantes :")
    print(pct_col.sort_values(ascending=False).head())

    if plot and not pct_col.empty:
        plt.figure(figsize=(10, 4))
        sns.barplot(x=pct_col.index, y=pct_col.values, palette='coolwarm')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('% de valeurs manquantes')
        plt.title('Pourcentage de valeurs manquantes par variable')
        plt.tight_layout()
        plt.show()

    return {
        'total_missing': int(miss),
        'percent_missing': pct,
        'cols_missing': by_col.to_dict(),
        'percent_per_col': pct_col.to_dict(),
        'high_missing': high.to_dict(),
        'medium_missing': med.to_dict(),
        'low_missing': low.to_dict(),
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