# modules/preprocessing/main_pipeline.py

from typing import Union, Optional, List
from pathlib import Path
import pandas as pd

from config.paths_config import setup_project_paths
from preprocessing import (
    apply_yeojohnson,
    convert_X4_to_int,
    detect_and_remove_outliers,
    drop_correlated_duplicates,
    find_highly_correlated_groups,
    handle_missing_values,
    load_data,
    validate_x4_presence
)


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
    require_outcome: bool = True,
    protect_x4: bool = True
) -> pd.DataFrame:
    """
    Pipeline de prétraitement complet avec protection de X4.
    """
    paths = setup_project_paths()
    protected_cols = ['X4'] if protect_x4 else []

    df = load_data(
        file_path=file_path,
        require_outcome=require_outcome,
        display_info=display_info,
        raw_data_dir=raw_data_dir,
        encode_target=True
    )

    if protect_x4:
        validate_x4_presence(df, "Après chargement", display_info)

    df = convert_X4_to_int(df, verbose=display_info)

    if protect_x4:
        validate_x4_presence(df, "Après conversion X4", display_info)

    df = apply_yeojohnson(
        df=df,
        columns=["X1", "X2", "X3"],
        standardize=False,
        save_model=save_transformer,
        model_path=models_dir / "yeojohnson.pkl" if save_transformer and models_dir else None,
        return_transformer=False
    )
    df.drop(columns=["X1", "X2", "X3"], inplace=True, errors="ignore")

    if protect_x4:
        validate_x4_presence(df, "Après Yeo-Johnson", display_info)

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

    if protect_x4:
        validate_x4_presence(df, "Après imputation", display_info)

    binary_vars = [col for col in df.columns if pd.api.types.is_integer_dtype(df[col]) and col != "outcome"]
    if display_info:
        print(f"🔢 Variables binaires candidates : {len(binary_vars)}")

    groups_corr = find_highly_correlated_groups(
        df[binary_vars], 
        threshold=correlation_threshold,
        protected_cols=protected_cols
    )

    target_col = "outcome" if "outcome" in df.columns and require_outcome else None

    df_reduced, dropped_cols, kept_cols = drop_correlated_duplicates(
        df=df,
        groups=groups_corr["groups"],
        target_col=target_col,
        extra_cols=mar_cols + mcar_cols,
        protected_cols=protected_cols,
        priority_cols=["X1_trans", "X2_trans", "X3_trans", "X4"],
        verbose=False,
        summary=display_info
    )

    df = df_reduced

    if protect_x4:
        validate_x4_presence(df, "Après réduction colinéarité", display_info)

    if drop_outliers and target_col:
        df = detect_and_remove_outliers(
            df=df,
            columns=mar_cols,
            method='iqr',
            remove=True,
            verbose=display_info
        )

        if protect_x4:
            validate_x4_presence(df, "Après suppression outliers", display_info)

    duplicate_cols = df.columns[df.columns.duplicated()].tolist()
    if duplicate_cols:
        if 'X4' in duplicate_cols and protect_x4:
            print("🔫 ALERTE: X4 détectée comme dupliquée - protection activée")
            df = df.loc[:, ~df.columns.duplicated(keep='first')]
        else:
            df = df.loc[:, ~df.columns.duplicated()]

        if display_info:
            print(f"⚠️ Colonnes dupliquées détectées : {duplicate_cols}")
            print(f"🔹 Après suppression doublons : {df.shape}")

        if protect_x4:
            validate_x4_presence(df, "Après suppression doublons", display_info)

    if display_info:
        print(f"✅ Pipeline terminé – Dimensions finales : {df.shape}")
        if protect_x4:
            final_status = validate_x4_presence(df, "VALIDATION FINALE", True)
            if not final_status:
                raise ValueError("X4 a été perdue pendant le preprocessing !")

    if processed_data_dir:
        processed_data_dir = Path(processed_data_dir)
        processed_data_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"{mar_method}{'_no_outliers' if drop_outliers else ''}"
        filename = f"final_dataset_{suffix}.parquet"
        df.to_parquet(processed_data_dir / filename, index=False)
        if display_info:
            print(f"📂 Sauvegarde : {processed_data_dir / filename}")

    return df
