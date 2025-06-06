# modules/preprocessing/outliers.py

import pandas as pd
import numpy as np
from pathlib import Path

def detect_and_remove_outliers(
    df: pd.DataFrame,
    columns: list,
    method: str = 'iqr',
    iqr_multiplier: float = 1.5,
    verbose: bool = True,
    save_path: Path = None
) -> pd.DataFrame:
    """
    Détecte et supprime les outliers selon la méthode de l'IQR sur les colonnes spécifiées.

    Args:
        df (pd.DataFrame): DataFrame d'entrée.
        columns (list): Liste des colonnes à analyser.
        method (str): Méthode de détection ('iqr' uniquement pour l'instant).
        iqr_multiplier (float): Facteur multiplicatif pour la règle de l’IQR.
        verbose (bool): Si True, affiche des statistiques.
        save_path (Path): Chemin facultatif pour sauvegarder le DataFrame résultant (.csv ou .parquet).

    Returns:
        pd.DataFrame: DataFrame sans les outliers détectés.
    """
    df_clean = df.copy()
    initial_shape = df.shape

    if method != 'iqr':
        raise NotImplementedError("Seule la méthode 'iqr' est actuellement implémentée.")

    mask = pd.Series([True] * df.shape[0], index=df.index)

    for col in columns:
        if col not in df.columns:
            continue

        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - iqr_multiplier * iqr
        upper_bound = q3 + iqr_multiplier * iqr

        outlier_mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
        mask &= outlier_mask

        if verbose:
            n_outliers = (~outlier_mask).sum()
            print(f"📉 {col} : {n_outliers} outliers détectés")

    df_clean = df[mask]

    if verbose:
        print(f"\n✅ Nombre total de lignes supprimées : {initial_shape[0] - df_clean.shape[0]}")
        print(f"🔢 Nouvelle taille du dataset : {df_clean.shape}")

    # Sauvegarde facultative
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if save_path.suffix == ".csv":
            df_clean.to_csv(save_path, index=False)
        elif save_path.suffix in [".parquet", ".pq"]:
            df_clean.to_parquet(save_path, index=False)
        else:
            raise ValueError("❌ Format de fichier non supporté. Utilisez .csv ou .parquet")

        if verbose:
            print(f"💾 Données sauvegardées à : {save_path}")

    return df_clean

