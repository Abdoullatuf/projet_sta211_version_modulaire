# final_preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer
from typing import List, Optional, Union
from pathlib import Path
import joblib


def apply_yeojohnson(
    df: pd.DataFrame,
    columns: List[str],
    standardize: bool = False,
    save_model: bool = False,
    model_path: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """
    Applique la transformation Yeo-Johnson sur des colonnes sélectionnées.

    Args:
        df (pd.DataFrame): Le DataFrame contenant les colonnes à transformer.
        columns (List[str]): Liste des colonnes à transformer.
        standardize (bool): Si True, applique une standardisation après la transformation.
        save_model (bool): Si True, sauvegarde le transformateur PowerTransformer.
        model_path (str or Path, optional): Chemin pour sauvegarder le modèle.

    Returns:
        pd.DataFrame: DataFrame avec les nouvelles colonnes transformées (suffixées `_trans`)
    """
    df_transformed = df.copy()

    # Initialiser le transformateur
    pt = PowerTransformer(method="yeo-johnson", standardize=standardize)

    # Appliquer la transformation
    try:
        transformed_values = pt.fit_transform(df[columns])

        for i, col in enumerate(columns):
            new_col = f"{col}_trans"
            df_transformed[new_col] = transformed_values[:, i]

        if save_model and model_path is not None:
            model_path = Path(model_path)
            model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(pt, model_path)
            print(f"✅ Transformateur Yeo-Johnson sauvegardé à : {model_path}")

        return df_transformed

    except Exception as e:
        print(f"❌ Erreur lors de la transformation Yeo-Johnson : {e}")
        return df
