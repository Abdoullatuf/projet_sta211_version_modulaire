# preprocessing/final_preprocessing.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
from typing import List, Optional, Union, Dict, Tuple

from sklearn.preprocessing import PowerTransformer

from config.paths_config import setup_project_paths
from preprocessing.missing_values import handle_missing_values
from preprocessing.outliers import detect_and_remove_outliers
from preprocessing.data_loader import load_data
from exploration.visualization import save_fig


# === 1. Utilitaires ===

def convert_X4_to_int(df: pd.DataFrame, column: str = "X4", verbose: bool = True) -> pd.DataFrame:
    if column not in df.columns:
        if verbose:
            print(f"⚠️ Colonne '{column}' absente du DataFrame.")
        return df.copy()

    unique_vals = df[column].dropna().unique()
    if set(unique_vals).issubset({0, 1}):
        df[column] = df[column].astype("Int64")
        if verbose:
            print(f"✅ Colonne '{column}' convertie en Int64 (binaire).")
    elif verbose:
        print(f"❌ Colonne '{column}' contient {unique_vals}. Conversion ignorée.")

    return df


def apply_yeojohnson(
    df: pd.DataFrame,
    columns: List[str],
    standardize: bool = False,
    save_model: bool = False,
    model_path: Optional[Union[str, Path]] = None,
    return_transformer: bool = False
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, PowerTransformer]]:
    """
    Applique la transformation Yeo-Johnson sur les colonnes spécifiées.
    """
    df_transformed = df.copy()

    if model_path and Path(model_path).exists():
        pt = joblib.load(model_path)
        print(f"🔄 Transformateur rechargé depuis : {model_path}")
    else:
        pt = PowerTransformer(method="yeo-johnson", standardize=standardize)
        pt.fit(df[columns])
        if save_model and model_path:
            Path(model_path).parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(pt, model_path)
            print(f"✅ Transformateur Yeo-Johnson sauvegardé à : {model_path}")

    transformed_values = pt.transform(df[columns])
    for i, col in enumerate(columns):
        df_transformed[f"{col}_trans"] = transformed_values[:, i]

    return (df_transformed, pt) if return_transformer else df_transformed


# === 2. Corrélation et réduction ===

def find_highly_correlated_groups(
    df: pd.DataFrame,
    threshold: float = 0.90,
    exclude_cols: Optional[List[str]] = None,
    show_plot: bool = False,
    save_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (10, 8)
) -> Dict[str, Union[List[List[str]], List[str]]]:
    """
    Identifie les groupes de variables fortement corrélées et retourne les colonnes à supprimer.
    """
    df_corr = df.drop(columns=exclude_cols) if exclude_cols else df.copy()
    corr_matrix = df_corr.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    groups, visited = [], set()
    for col in upper.columns:
        if col in visited:
            continue
        correlated = upper[col][upper[col] > threshold].index.tolist()
        if correlated:
            group = sorted(set([col] + correlated))
            groups.append(group)
            visited.update(group)

    to_drop = [var for group in groups for var in group[1:]]

    if show_plot:
        plt.figure(figsize=figsize)
        sns.heatmap(corr_matrix, cmap="coolwarm", center=0, square=True, cbar_kws={"shrink": 0.75})
        plt.title(f"Matrice de corrélation (>{threshold})")
        plt.tight_layout()
        if save_path:
            save_fig(Path(save_path).name, directory=Path(save_path).parent, figsize=figsize)
        else:
            plt.show()

    return {"groups": groups, "to_drop": to_drop}


def drop_correlated_duplicates(
    df: pd.DataFrame,
    groups: List[List[str]],
    target_col: str = "outcome",
    extra_cols: List[str] = None,
    verbose: bool = False,
    summary: bool = True
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Supprime toutes les variables d'un groupe corrélé sauf la première.

    Returns:
        - df_reduced: DataFrame nettoyé
        - to_drop: colonnes supprimées
        - to_keep: colonnes conservées
    """
    to_drop, to_keep = [], []

    for group in groups:
        if not group:
            continue
        keep = group[0]
        drop = [col for col in group[1:] if col in df.columns]
        to_keep.append(keep)
        to_drop.extend(drop)
        if verbose:
            print(f"🧹 Groupe : {group} → garde {keep}, retire {drop}")

    to_drop = sorted(set(to_drop))
    to_keep = sorted(set(to_keep))

    # Colonnes binaires restantes (non corrélées)
    all_binary = [col for col in df.select_dtypes(include='int64').columns if col != target_col]
    untouched = [col for col in all_binary if col not in to_drop + to_keep]

    # Construction sécurisée de la liste des colonnes finales
    final_cols = to_keep + untouched
    if extra_cols:
        final_cols += [col for col in extra_cols if col in df.columns]
    if target_col and target_col in df.columns:
        final_cols.append(target_col)

    # Filtrage explicite des colonnes existantes uniquement
    existing_cols = [col for col in final_cols if col in df.columns]
    missing_cols = list(set(final_cols) - set(existing_cols))
    if missing_cols:
        print(f"⚠️ Colonnes manquantes ignorées : {sorted(missing_cols)}")

    df_reduced = df[existing_cols].copy()

    if summary:
        print(f"\n📊 Réduction : {len(to_drop)} supprimées, {len(to_keep)} gardées, {len(untouched)} intactes.")
        if extra_cols:
            print(f"🧩 {len(extra_cols)} ajoutées : {extra_cols}")
        print(f"📐 Dimensions : {df_reduced.shape}")

    return df_reduced, to_drop, to_keep



def apply_collinearity_filter(df: pd.DataFrame, cols_to_drop: List[str], display_info: bool = True) -> pd.DataFrame:
    """
    Supprime les colonnes corrélées.
    """
    df_filtered = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    if display_info:
        print(f"✅ Colonnes supprimées : {len(cols_to_drop)}")
        print(f"📏 Dimensions finales : {df_filtered.shape}")
    return df_filtered


# === 3. Pipeline final ===
def prepare_final_dataset(
    file_path: Union[str, Path],
    strategy: str = "mixed_mar_mcar",
    mar_method: str = "knn",
    knn_k: Optional[int] = None,
    mar_cols: List[str] = ["X1_trans", "X2_trans", "X3_trans"],
    mcar_cols: List[str] = ["X4"],
    drop_outliers: bool = False,
    correlation_threshold: float = 0.90,
    save_transformer: bool = False,
    processed_data_dir: Optional[Union[str, Path]] = None,
    models_dir: Optional[Union[str, Path]] = None,
    display_info: bool = True,
    raw_data_dir: Optional[Union[str, Path]] = None,
    require_outcome: bool = True
) -> pd.DataFrame:
    paths = setup_project_paths()

    # 1. Chargement
    df = load_data(
        file_path=file_path,
        require_outcome=require_outcome,
        display_info=display_info,
        raw_data_dir=raw_data_dir
    )

    # 2. Convertir X4
    df = convert_X4_to_int(df)

    # 3. Yeo-Johnson
    df = apply_yeojohnson(
        df=df,
        columns=["X1", "X2", "X3"],
        standardize=False,
        save_model=save_transformer,
        model_path=models_dir / "yeojohnson.pkl" if save_transformer and models_dir else None,
        return_transformer=False
    )
    df.drop(columns=["X1", "X2", "X3"], inplace=True, errors="ignore")

    # 4. Imputation
    df = handle_missing_values(
        df=df,
        strategy=strategy,
        mar_method=mar_method,
        knn_k=knn_k,
        mar_cols=mar_cols,
        mcar_cols=mcar_cols,
        display_info=display_info,
        save_results=False,
        processed_data_dir=processed_data_dir,
        models_dir=models_dir
    )

    # 5. Réduction de la colinéarité
    # ✅ Correction ici : accepte les int classiques et Int64 nullables
    binary_vars = [col for col in df.columns if pd.api.types.is_integer_dtype(df[col]) and col != "outcome"]
    if display_info:
        print(f"🔢 Variables binaires candidates : {len(binary_vars)}")

    groups_corr = find_highly_correlated_groups(df[binary_vars], threshold=correlation_threshold)
    target_col = "outcome" if "outcome" in df.columns and require_outcome else None

    df_reduced, _, _ = drop_correlated_duplicates(
        df=df,
        groups=groups_corr["groups"],
        target_col=target_col,
        extra_cols=mar_cols + mcar_cols,
        summary=display_info
    )

    # 6. Suppression des outliers (si demandé)
    if drop_outliers and target_col:
        df_reduced = detect_and_remove_outliers(
            df=df_reduced,
            columns=mar_cols,
            method='iqr',
            remove=True,
            verbose=display_info
        )

    # 7. Suppression des colonnes dupliquées éventuelles
    duplicate_cols = df_reduced.columns[df_reduced.columns.duplicated()].tolist()
    if duplicate_cols:
        df_reduced = df_reduced.loc[:, ~df_reduced.columns.duplicated()]
        if display_info:
            print(f"⚠️ Colonnes dupliquées détectées : {duplicate_cols}")
            print(f"🔹 Étape 7 – Duplication supprimée : {df_reduced.shape}")

    # 8. Affichage
    if display_info:
        print(f"✅ Pipeline complet terminé – Dimensions finales : {df_reduced.shape}")

    # 9. Sauvegarde parquet
    if processed_data_dir:
        processed_data_dir = Path(processed_data_dir)
        processed_data_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"{mar_method}{'_no_outliers' if drop_outliers else ''}"
        filename = f"final_dataset_{suffix}.parquet"
        df_reduced.to_parquet(processed_data_dir / filename, index=False)
        if display_info:
            print(f"💾 Sauvegarde Parquet : {processed_data_dir / filename}")

    return df_reduced


