# data loader.py

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Optional
from config.paths_config import setup_project_paths
from IPython.display import display


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie un DataFrame :
      - strip + supprime guillemets des noms de colonnes,
      - strip + supprime guillemets des valeurs string,
      - gère les colonnes dupliquées finissant par '.1' de façon sécurisée.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("❌ L'objet passé à clean_data n'est pas un DataFrame.")

    df = df.copy()
    df.columns = df.columns.str.strip().str.replace('"', '')

    # Nettoyage des valeurs string
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip().str.replace('"', '')

    # Gérer les colonnes .1 si pas de version principale
    for base_col in ['X1', 'X2', 'X3', 'X4']:
        if f"{base_col}.1" in df.columns and base_col not in df.columns:
            print(f"⚠️ Renommage de {base_col}.1 en {base_col} (pas d'autre version détectée)")
            df = df.rename(columns={f"{base_col}.1": base_col})

    # Supprimer les colonnes dupliquées
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
    encode_target: bool = False
) -> pd.DataFrame:
    """
    Charge un fichier CSV, nettoie les données et affiche un aperçu.

    - file_path : nom ou chemin vers le fichier à charger
    - require_outcome : True pour forcer la présence de la colonne 'outcome'
    - display_info : affiche les infos principales
    - raw_data_dir : dossier contenant les fichiers si chemin relatif
    - encode_target : transforme outcome ('ad.', 'noad.') en 1 et 0
    """
    file_path = Path(file_path)

    if raw_data_dir is None:
        paths = setup_project_paths()
        raw_data_dir = paths["RAW_DATA_DIR"]
    else:
        raw_data_dir = Path(raw_data_dir)

    real_path = file_path if file_path.is_absolute() else raw_data_dir / file_path

    if not real_path.exists():
        raise FileNotFoundError(f"📂 Fichier introuvable : {real_path}")

    # Chargement CSV
    try:
        df = pd.read_csv(real_path, sep='\t', encoding='utf-8', encoding_errors='ignore')
        if df.shape[1] == 1:
            df = pd.read_csv(real_path, sep=',', encoding='utf-8', encoding_errors='ignore')
    except Exception as e:
        raise RuntimeError(f"❌ Erreur lecture {real_path}: {e}")

    # Vérification de la cible
    if require_outcome and 'outcome' not in df.columns:
        raise ValueError("🚨 La colonne 'outcome' est manquante.")

    # Nettoyage
    df = clean_data(df)

    # Encodage de la cible
    if require_outcome and encode_target and 'outcome' in df.columns:
        valid_values = {'ad.', 'noad.'}
        unique_outcomes = set(str(val).strip().lower() for val in df['outcome'].unique())
        if unique_outcomes.issubset(valid_values):
            df['outcome'] = df['outcome'].str.strip().str.lower().map({'ad.': 1, 'noad.': 0})
            print("✅ Colonne 'outcome' encodée en numérique (ad. → 1, noad. → 0)")
        else:
            raise ValueError(f"❌ Valeurs inattendues dans 'outcome': {df['outcome'].unique()}")

    # Affichage infos
    if display_info:
        print(f"\n✅ Fichier chargé : {real_path}")
        print(f"🔢 Dimensions : {df.shape}")
        print("📋 Infos colonnes :")
        df.info()
        print("\n🔎 Premières lignes :")
        try:
            display(df.head())
        except:
            print(df.head())

    return df
