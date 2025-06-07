
# final_preprocessing.py

import os
import joblib
import numpy as np
import pandas as pd
from typing import Optional, Union, List
from pathlib import Path

# sklearn
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import KNNImputer, IterativeImputer, SimpleImputer
from sklearn.preprocessing import PowerTransformer

# Projet STA211
from data_preprocessing import load_data, handle_missing_values
from project_setup import setup_project_paths
from exploratory_analysis import (
    find_highly_correlated_groups,
    drop_correlated_duplicates,
    detect_outliers_iqr
)


def convert_X4_to_int(df: pd.DataFrame) -> pd.DataFrame:
    if "X4" in df.columns:
        df["X4"] = df["X4"].astype("Int64")
    return df


def apply_yeojohnson(
    df: pd.DataFrame,
    columns: list = ["X1", "X2", "X3"],
    standardize: bool = False,
    save_model: bool = False,
    model_path: Optional[Union[str, Path]] = None
) -> tuple:
    df_transformed = df.copy()
    pt = PowerTransformer(method="yeo-johnson", standardize=standardize)
    transformed_data = pt.fit_transform(df[columns])
    for i, col in enumerate(columns):
        df_transformed[f"{col}_trans"] = transformed_data[:, i]
    if save_model and model_path:
        joblib.dump(pt, model_path)
    return df_transformed, pt


# final_preprocessing.py

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
    require_outcome: bool = True,  # <-- NOUVEAU PAR DÉFAUT True
    return_objects: bool = False
) -> pd.DataFrame:
    paths = setup_project_paths()

    df = load_data(
        file_path=file_path,
        require_outcome=require_outcome,
        display_info=display_info,
        raw_data_dir=raw_data_dir
    )

    df = convert_X4_to_int(df)
    df, transformer = apply_yeojohnson(
        df,
        columns=["X1", "X2", "X3"],
        standardize=False,
        save_model=save_transformer,
        model_path=models_dir / "yeojohnson.pkl" if models_dir else None
    )
    df.drop(columns=["X1", "X2", "X3"], inplace=True, errors="ignore")

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

    # === Gestion de la sélection/réduction des variables corrélées ===
    binary_vars = [col for col in df.select_dtypes(include='int64').columns if col != "outcome"]
    groups_corr = find_highly_correlated_groups(df[binary_vars], threshold=correlation_threshold)

    # Détermine si 'outcome' est là (True en train, False en test)
    outcome_present = "outcome" in df.columns and require_outcome

    # --- Si on n'a pas la colonne outcome, adapte drop_correlated_duplicates
    if outcome_present:
        target_col = "outcome"
    else:
        target_col = None  # pas de target, pas de pondération par classe/target

    # drop_correlated_duplicates doit gérer target_col=None sans erreur !
    df_reduced, _, _ = drop_correlated_duplicates(
        df=df,
        groups=groups_corr["groups"],
        target_col=target_col,
        extra_cols=mar_cols + mcar_cols,
        verbose=False,
        summary=display_info
    )

    # Outliers (que si outcome dispo… mais ici, OK de le faire sur test aussi)
    if drop_outliers and outcome_present:
        mask = ~detect_outliers_iqr(df_reduced["X1_trans"]) & \
               ~detect_outliers_iqr(df_reduced["X2_trans"]) & \
               ~detect_outliers_iqr(df_reduced["X3_trans"])
        df_reduced = df_reduced[mask]
        if display_info:
            print(f"🧹 {df.shape[0] - df_reduced.shape[0]} lignes supprimées à cause des outliers.")

    if display_info:
        print(f"📐 DataFrame final : {df_reduced.shape}")
        from IPython.display import display
        display(df_reduced.head(3))

    return df_reduced

