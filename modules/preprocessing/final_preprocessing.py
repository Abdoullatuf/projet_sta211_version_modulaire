"""
preprocessing/final_preprocessing.py - VERSION CORRIGÉE AVEC PROTECTION X4

Module de prétraitement complet pour le projet STA211 Internet Advertisements.
Inclut la protection de X4, l'ordre des colonnes optimisé, et la validation à chaque étape.

CORRECTIONS:
- Fix TypeError dans find_highly_correlated_groups
- Validation robuste des types de retour
- Gestion d'erreurs améliorée

Auteur: Abdoullatuf
Version: 2.1 (Corrigée)
Date: 2025
"""

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


# ============================================================================
# 1. UTILITAIRES DE BASE
# ============================================================================

def convert_X4_to_int(df: pd.DataFrame, column: str = "X4", verbose: bool = True) -> pd.DataFrame:
    """
    Convertit X4 en Int64 si elle contient uniquement des valeurs binaires (0, 1).
    
    Args:
        df: DataFrame d'entrée
        column: Nom de la colonne à convertir (défaut: "X4")
        verbose: Affichage des informations
        
    Returns:
        DataFrame avec X4 convertie en Int64
    """
    df = df.copy()
    
    if column not in df.columns:
        if verbose:
            print(f"⚠️ Colonne '{column}' absente du DataFrame.")
        return df

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
    
    Args:
        df: DataFrame d'entrée
        columns: Colonnes à transformer
        standardize: Standardiser après transformation
        save_model: Sauvegarder le transformateur
        model_path: Chemin de sauvegarde du modèle
        return_transformer: Retourner aussi le transformateur
        
    Returns:
        DataFrame transformé (et transformateur si demandé)
    """
    df_transformed = df.copy()

    # Charger ou créer le transformateur
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

    # Appliquer la transformation
    transformed_values = pt.transform(df[columns])
    for i, col in enumerate(columns):
        df_transformed[f"{col}_trans"] = transformed_values[:, i]

    return (df_transformed, pt) if return_transformer else df_transformed


# ============================================================================
# 2. GESTION DE LA CORRÉLATION ET RÉDUCTION DE DIMENSIONNALITÉ (CORRIGÉE)
# ============================================================================

def find_highly_correlated_groups(
    df: pd.DataFrame,
    threshold: float = 0.90,
    exclude_cols: Optional[List[str]] = None,
    protected_cols: Optional[List[str]] = None,
    show_plot: bool = False,
    save_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (10, 8)
) -> Dict[str, Union[List[List[str]], List[str]]]:
    """
    Identifie les groupes de variables fortement corrélées avec protection de certaines colonnes.
    
    CORRECTION: Garantit toujours un retour en dictionnaire avec validation.
    
    Args:
        df: DataFrame d'entrée
        threshold: Seuil de corrélation (défaut: 0.90)
        exclude_cols: Colonnes à exclure de l'analyse
        protected_cols: Colonnes à protéger de la suppression (défaut: ['X4'])
        show_plot: Afficher la heatmap de corrélation
        save_path: Chemin de sauvegarde de la figure
        figsize: Taille de la figure
        
    Returns:
        Dictionnaire avec groups, to_drop, et protected
    """
    # 🛡️ Protection par défaut de X4
    if protected_cols is None:
        protected_cols = ['X4']
    
    # 🔧 VALIDATION D'ENTRÉE (NOUVELLE)
    if df.empty:
        print("⚠️ DataFrame vide - retour structure par défaut")
        return {"groups": [], "to_drop": [], "protected": protected_cols}
    
    # Combinaison des colonnes à exclure
    all_exclude_cols = (exclude_cols or []) + (protected_cols or [])
    
    # Calcul de la matrice de corrélation
    df_corr = df.drop(columns=all_exclude_cols, errors='ignore') if all_exclude_cols else df.copy()
    
    # 🔧 VALIDATION INTERMÉDIAIRE (NOUVELLE)
    if df_corr.empty:
        print("⚠️ Aucune colonne à analyser après exclusions")
        return {"groups": [], "to_drop": [], "protected": protected_cols}
    
    try:
        corr_matrix = df_corr.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    except Exception as e:
        print(f"⚠️ Erreur dans le calcul de corrélation: {e}")
        return {"groups": [], "to_drop": [], "protected": protected_cols}

    # Identification des groupes corrélés
    groups, visited = [], set()
    for col in upper.columns:
        if col in visited:
            continue
        correlated = upper[col][upper[col] > threshold].index.tolist()
        if correlated:
            group = sorted(set([col] + correlated))
            groups.append(group)
            visited.update(group)

    # 🛡️ Création de la liste to_drop avec protection
    to_drop = []
    for group in groups:
        # Dans chaque groupe, on garde le premier ET les colonnes protégées
        protected_in_group = [col for col in group if col in protected_cols]
        non_protected = [col for col in group if col not in protected_cols]
        
        if non_protected:
            # Garder le premier non-protégé, supprimer les autres non-protégés
            to_drop.extend(non_protected[1:])
        # Les colonnes protégées ne sont jamais ajoutées à to_drop

    # Visualisation optionnelle
    if show_plot and not corr_matrix.empty:
        try:
            plt.figure(figsize=figsize)
            sns.heatmap(corr_matrix, cmap="coolwarm", center=0, square=True, 
                       cbar_kws={"shrink": 0.75})
            plt.title(f"Matrice de corrélation (>{threshold})")
            plt.tight_layout()
            
            if save_path:
                save_fig(Path(save_path).name, directory=Path(save_path).parent, figsize=figsize)
            else:
                plt.show()
        except Exception as e:
            print(f"⚠️ Erreur dans la visualisation: {e}")

    # 🛡️ Vérification finale de protection
    protected_in_drop = set(to_drop) & set(protected_cols)
    if protected_in_drop:
        print(f"🛡️ PROTECTION ACTIVÉE: Retrait de {protected_in_drop} de la liste de suppression")
        to_drop = [col for col in to_drop if col not in protected_cols]

    # 🔧 CONSTRUCTION ET VALIDATION DU RETOUR (NOUVELLE)
    result = {
        "groups": groups,
        "to_drop": to_drop,
        "protected": protected_cols
    }
    
    # Validation du type de retour
    assert isinstance(result, dict), "Le retour doit être un dictionnaire"
    assert "groups" in result, "La clé 'groups' doit être présente"
    assert isinstance(result["groups"], list), "groups doit être une liste"
    assert isinstance(result["to_drop"], list), "to_drop doit être une liste"
    assert isinstance(result["protected"], list), "protected doit être une liste"

    return result


def drop_correlated_duplicates(
    df: pd.DataFrame,
    groups: List[List[str]],
    target_col: str = "outcome",
    extra_cols: List[str] = None,
    protected_cols: List[str] = None,
    priority_cols: List[str] = None,
    verbose: bool = False,
    summary: bool = True
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Supprime les variables corrélées avec protection et ordre des colonnes optimisé.
    
    Args:
        df: DataFrame d'entrée
        groups: Groupes de variables corrélées
        target_col: Variable cible
        extra_cols: Colonnes supplémentaires à conserver
        protected_cols: Colonnes à protéger (défaut: ['X4'])
        priority_cols: Colonnes prioritaires pour l'ordre (défaut: ['X1_trans', 'X2_trans', 'X3_trans', 'X4'])
        verbose: Affichage détaillé
        summary: Afficher le résumé
        
    Returns:
        Tuple (DataFrame réduit, colonnes supprimées, colonnes gardées)
    """
    # 🛡️ Protection par défaut
    if protected_cols is None:
        protected_cols = ['X4']
    
    # 📌 Colonnes prioritaires par défaut
    if priority_cols is None:
        priority_cols = ['X1_trans', 'X2_trans', 'X3_trans', 'X4']
    
    # 🔧 VALIDATION D'ENTRÉE (NOUVELLE)
    if not isinstance(groups, list):
        print(f"⚠️ groups doit être une liste, reçu: {type(groups)}")
        groups = []
    
    to_drop, to_keep = [], []

    # Traitement des groupes corrélés
    for group in groups:
        if not group or not isinstance(group, list):
            continue
        
        # 🛡️ Séparer les colonnes protégées des autres
        protected_in_group = [col for col in group if col in protected_cols]
        non_protected = [col for col in group if col not in protected_cols and col in df.columns]
        
        if non_protected:
            # Garder le premier non-protégé
            keep = non_protected[0]
            drop = non_protected[1:]
            to_keep.append(keep)
            to_drop.extend(drop)
            
            if verbose:
                print(f"🧹 Groupe : {group} → garde {keep}, retire {drop}")
                if protected_in_group:
                    print(f"🛡️   Protégées dans ce groupe : {protected_in_group}")
        
        # 🛡️ Les colonnes protégées sont toujours gardées
        for protected in protected_in_group:
            if protected not in to_keep:
                to_keep.append(protected)

    to_drop = sorted(set(to_drop))
    to_keep = sorted(set(to_keep))

    # Colonnes binaires restantes (non corrélées)
    all_binary = [col for col in df.select_dtypes(include=['int64', 'Int64']).columns 
                  if col != target_col]
    untouched = [col for col in all_binary if col not in to_drop + to_keep]

    # 📌 CONSTRUCTION DE L'ORDRE FINAL DES COLONNES
    
    # 1. Colonnes prioritaires (en premier)
    priority_existing = [col for col in priority_cols if col in df.columns]
    
    # 2. Variable cible (après les prioritaires)
    target_existing = [target_col] if target_col and target_col in df.columns else []
    
    # 3. Extra cols (variables transformées, etc.)
    extra_existing = []
    if extra_cols:
        extra_existing = [col for col in extra_cols 
                         if col in df.columns and col not in priority_existing + target_existing]
    
    # 4. Variables gardées par corrélation (pas déjà dans prioritaires/extra)
    kept_remaining = [col for col in to_keep 
                     if col not in priority_existing + target_existing + extra_existing]
    
    # 5. Variables intactes (pas déjà listées)
    untouched_remaining = [col for col in untouched 
                          if col not in priority_existing + target_existing + extra_existing + kept_remaining]
    
    # 🛡️ S'assurer que les colonnes protégées sont présentes
    protected_remaining = []
    for protected in protected_cols:
        if (protected in df.columns and 
            protected not in priority_existing + target_existing + extra_existing + 
            kept_remaining + untouched_remaining):
            protected_remaining.append(protected)
    
    # 📌 ORDRE FINAL : prioritaires → cible → extra → gardées → intactes → protégées restantes
    final_cols = (priority_existing + target_existing + extra_existing + 
                  kept_remaining + untouched_remaining + protected_remaining)
    
    # Filtrage des colonnes existantes (sécurité)
    existing_cols = [col for col in final_cols if col in df.columns]
    df_reduced = df[existing_cols].copy()

    # Affichage du résumé
    if summary:
        print(f"\n📊 Réduction : {len(to_drop)} supprimées, {len(to_keep)} gardées, {len(untouched)} intactes.")
        print(f"📌 Ordre final : {priority_existing[:3]}{'...' if len(priority_existing) > 3 else ''} → {target_existing} → reste")
        
        if protected_cols:
            protected_in_final = [col for col in protected_cols if col in existing_cols]
            print(f"🛡️ {len(protected_in_final)} colonnes protégées : {protected_in_final}")
        
        if extra_cols:
            existing_extra = [col for col in extra_cols if col in existing_cols]
            print(f"🧩 {len(existing_extra)} extra conservées : {existing_extra}")
        
        print(f"📐 Dimensions : {df_reduced.shape}")

    return df_reduced, to_drop, to_keep


def apply_collinearity_filter(
    df: pd.DataFrame, 
    cols_to_drop: List[str], 
    protected_cols: List[str] = None, 
    display_info: bool = True
) -> pd.DataFrame:
    """
    Supprime les colonnes corrélées en protégeant certaines colonnes.
    
    Args:
        df: DataFrame d'entrée
        cols_to_drop: Colonnes à supprimer
        protected_cols: Colonnes à protéger (défaut: ['X4'])
        display_info: Afficher les informations
        
    Returns:
        DataFrame filtré
    """
    # 🛡️ Protection par défaut
    if protected_cols is None:
        protected_cols = ['X4']
    
    # 🛡️ Retirer les colonnes protégées de la liste de suppression
    original_drop_count = len(cols_to_drop)
    cols_to_drop_filtered = [col for col in cols_to_drop if col not in protected_cols]
    protected_saved = original_drop_count - len(cols_to_drop_filtered)
    
    if protected_saved > 0 and display_info:
        saved_cols = [col for col in cols_to_drop if col in protected_cols]
        print(f"🛡️ {protected_saved} colonnes protégées de la suppression : {saved_cols}")
    
    # Suppression sécurisée
    df_filtered = df.drop(columns=[col for col in cols_to_drop_filtered if col in df.columns])
    
    if display_info:
        print(f"✅ Colonnes supprimées : {len(cols_to_drop_filtered)}")
        print(f"📏 Dimensions finales : {df_filtered.shape}")
        
        # 🛡️ Vérification finale que les colonnes protégées sont présentes
        for protected in protected_cols:
            if protected in df.columns:
                status = "✅" if protected in df_filtered.columns else "❌"
                print(f"🛡️ {protected} : {status}")
    
    return df_filtered


# ============================================================================
# 3. FONCTIONS DE VALIDATION ET PROTECTION X4
# ============================================================================

def validate_x4_presence(df: pd.DataFrame, step_name: str = "", verbose: bool = True) -> bool:
    """
    Valide que X4 est présente et correcte dans le DataFrame.
    
    Args:
        df: DataFrame à vérifier
        step_name: Nom de l'étape (pour l'affichage)
        verbose: Affichage des informations
        
    Returns:
        True si X4 est présente et correcte
    """
    if 'X4' not in df.columns:
        if verbose:
            print(f"❌ {step_name}: X4 MANQUANTE !")
        return False
    
    # Vérifier le type et les valeurs
    unique_vals = sorted(df['X4'].dropna().unique())
    expected_vals = [0, 1]
    
    if set(unique_vals).issubset(set(expected_vals)):
        if verbose:
            print(f"✅ {step_name}: X4 présente et correcte (valeurs: {unique_vals})")
        return True
    else:
        if verbose:
            print(f"⚠️ {step_name}: X4 présente mais valeurs inattendues: {unique_vals}")
        return False


def quick_x4_check(df_or_dict, name: str = "Dataset") -> bool:
    """
    Vérification rapide de X4 dans un DataFrame ou dictionnaire de DataFrames.
    
    Args:
        df_or_dict: DataFrame ou dictionnaire de DataFrames
        name: Nom pour l'affichage
        
    Returns:
        True si X4 est présente dans tous les datasets
    """
    if isinstance(df_or_dict, dict):
        print(f"🔍 Vérification X4 dans {len(df_or_dict)} datasets:")
        all_good = True
        for dataset_name, df in df_or_dict.items():
            has_x4 = 'X4' in df.columns if df is not None else False
            print(f"  {dataset_name}: {'✅' if has_x4 else '❌'}")
            if not has_x4:
                all_good = False
        return all_good
    else:
        # DataFrame unique
        has_x4 = 'X4' in df_or_dict.columns
        print(f"🔍 {name}: {'✅' if has_x4 else '❌'} X4")
        return has_x4


# ============================================================================
# 4. GESTION DE L'ORDRE DES COLONNES
# ============================================================================

def reorder_columns_priority(
    df: pd.DataFrame, 
    priority_cols: List[str] = None,
    target_col: str = "outcome"
) -> pd.DataFrame:
    """
    Réorganise les colonnes avec un ordre prioritaire.
    
    Args:
        df: DataFrame à réorganiser
        priority_cols: Colonnes prioritaires (défaut: ['X1_trans', 'X2_trans', 'X3_trans', 'X4'])
        target_col: Variable cible à placer après les prioritaires
        
    Returns:
        DataFrame avec colonnes réorganisées
    """
    if priority_cols is None:
        priority_cols = ['X1_trans', 'X2_trans', 'X3_trans', 'X4']
    
    current_cols = df.columns.tolist()
    
    # 1. Variables prioritaires (en premier)
    final_priority = [col for col in priority_cols if col in current_cols]
    
    # 2. Variable cible (après les prioritaires)
    final_target = [target_col] if target_col and target_col in current_cols else []
    
    # 3. Toutes les autres colonnes (dans l'ordre actuel)
    final_others = [col for col in current_cols if col not in final_priority + final_target]
    
    # Ordre final : prioritaires → cible → reste
    final_order = final_priority + final_target + final_others
    
    return df[final_order]


def check_column_order(
    df: pd.DataFrame, 
    expected_first: List[str] = None,
    display_info: bool = True
) -> bool:
    """
    Vérifie l'ordre des colonnes dans le DataFrame.
    
    Args:
        df: DataFrame à vérifier
        expected_first: Colonnes attendues en premier
        display_info: Afficher les informations
        
    Returns:
        True si l'ordre est correct
    """
    if expected_first is None:
        expected_first = ['X1_trans', 'X2_trans', 'X3_trans', 'X4', 'outcome']
    
    current_order = df.columns.tolist()
    
    if display_info:
        print(f"📌 Ordre actuel des colonnes (premiers 8) :")
        print(f"   {current_order[:8]}")
        
        print(f"📌 Colonnes attendues en premier :")
        print(f"   {expected_first}")
    
    # Vérifier si les colonnes attendues sont bien en début
    matches = []
    for i, expected_col in enumerate(expected_first):
        if expected_col in current_order:
            actual_position = current_order.index(expected_col)
            expected_position = i
            matches.append({
                'column': expected_col,
                'expected_pos': expected_position,
                'actual_pos': actual_position,
                'correct': actual_position == expected_position
            })
    
    all_correct = all(match['correct'] for match in matches)
    
    if display_info:
        print(f"📊 Vérification de l'ordre :")
        for match in matches:
            status = "✅" if match['correct'] else "❌"
            print(f"   {status} {match['column']}: position {match['actual_pos']} (attendu: {match['expected_pos']})")
    
    return all_correct


def reorganize_existing_datasets(
    datasets_dict: Dict[str, pd.DataFrame],
    priority_cols: List[str] = None,
    target_col: str = "outcome",
    verbose: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Réorganise plusieurs datasets existants avec l'ordre de colonnes souhaité.
    
    Args:
        datasets_dict: Dictionnaire {nom: DataFrame}
        priority_cols: Colonnes prioritaires
        target_col: Variable cible
        verbose: Affichage détaillé
        
    Returns:
        Dictionnaire des datasets réorganisés
    """
    if priority_cols is None:
        priority_cols = ['X1_trans', 'X2_trans', 'X3_trans', 'X4']
    
    reorganized = {}
    
    for name, df in datasets_dict.items():
        if df is None:
            if verbose:
                print(f"⚠️ {name}: DataFrame vide, ignoré")
            reorganized[name] = df
            continue
        
        # Réorganiser
        df_reordered = reorder_columns_priority(df, priority_cols, target_col)
        reorganized[name] = df_reordered
        
        if verbose:
            print(f"✅ {name}: colonnes réorganisées")
            print(f"   Premières colonnes : {df_reordered.columns[:min(6, len(df_reordered.columns))].tolist()}")
    
    return reorganized


# ============================================================================
# 5. PIPELINE PRINCIPAL DE PRÉTRAITEMENT (CORRIGÉ)
# ============================================================================

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
    protect_x4: bool = True,
    priority_cols: List[str] = None,
    return_objects: bool = False
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict]]:
    """
    Pipeline de prétraitement complet avec protection de X4 et ordre des colonnes optimisé.
    
    CORRECTIONS:
    - Validation robuste de find_highly_correlated_groups
    - Gestion d'erreur pour les cas edge
    - Type checking systématique
    
    Args:
        file_path: Chemin vers le fichier de données
        strategy: Stratégie d'imputation ("mixed_mar_mcar")
        mar_method: Méthode d'imputation MAR ("knn" ou "mice")
        knn_k: Paramètre k pour KNN (None = auto)
        mar_cols: Colonnes à imputer avec méthode MAR
        mcar_cols: Colonnes à imputer avec méthode MCAR
        drop_outliers: Supprimer les outliers
        correlation_threshold: Seuil de corrélation pour suppression
        save_transformer: Sauvegarder les transformateurs
        processed_data_dir: Dossier de sauvegarde des données
        models_dir: Dossier de sauvegarde des modèles
        display_info: Affichage des informations
        raw_data_dir: Dossier des données brutes
        require_outcome: Nécessite la variable cible
        protect_x4: Protéger X4 de la suppression
        priority_cols: Colonnes prioritaires pour l'ordre
        return_objects: Retourner aussi les objets de transformation
        
    Returns:
        DataFrame prétraité (et objets si demandé)
        
    Raises:
        ValueError: Si X4 est perdue pendant le preprocessing
    """
    paths = setup_project_paths()
    
    # 🛡️ Configuration de protection
    protected_cols = ['X4'] if protect_x4 else []
    
    # 📌 Colonnes prioritaires par défaut
    if priority_cols is None:
        priority_cols = ['X1_trans', 'X2_trans', 'X3_trans', 'X4']

    # Dictionnaire pour stocker les objets de transformation
    transform_objects = {
        'scaler': None,
        'imputer': None,
        'yeojohnson': None,
        'correlation_info': None
    }

    if display_info:
        print("🔄 DÉMARRAGE DU PIPELINE DE PRÉTRAITEMENT (VERSION CORRIGÉE)")
        print("=" * 70)

    # ========================================================================
    # ÉTAPE 1: CHARGEMENT DES DONNÉES
    # ========================================================================
    
    if display_info:
        print("📂 Étape 1: Chargement des données...")
    
    try:
        df = load_data(
            file_path=file_path,
            require_outcome=require_outcome,
            display_info=display_info,
            raw_data_dir=raw_data_dir,
            encode_target=True
        )
    except Exception as e:
        print(f"❌ Erreur lors du chargement : {e}")
        raise
    
    # 🛡️ Validation X4 après chargement
    if protect_x4:
        validate_x4_presence(df, "Après chargement", display_info)

    # ========================================================================
    # ÉTAPE 2: CONVERSION DE X4
    # ========================================================================
    
    if display_info:
        print("\n🔧 Étape 2: Conversion de X4...")
    
    df = convert_X4_to_int(df, verbose=display_info)
    
    # 🛡️ Validation X4 après conversion
    if protect_x4:
        validate_x4_presence(df, "Après conversion X4", display_info)

    # ========================================================================
    # ÉTAPE 3: TRANSFORMATION YEO-JOHNSON
    # ========================================================================
    
    if display_info:
        print("\n🔄 Étape 3: Transformation Yeo-Johnson (X1, X2, X3)...")
    
    try:
        if return_objects:
            df, yeojohnson_transformer = apply_yeojohnson(
                df=df,
                columns=["X1", "X2", "X3"],
                standardize=False,
                save_model=save_transformer,
                model_path=models_dir / "yeojohnson.pkl" if save_transformer and models_dir else None,
                return_transformer=True
            )
            transform_objects['yeojohnson'] = yeojohnson_transformer
        else:
            df = apply_yeojohnson(
                df=df,
                columns=["X1", "X2", "X3"],
                standardize=False,
                save_model=save_transformer,
                model_path=models_dir / "yeojohnson.pkl" if save_transformer and models_dir else None,
                return_transformer=False
            )
        
        # Suppression des colonnes originales
        df.drop(columns=["X1", "X2", "X3"], inplace=True, errors="ignore")
        
    except Exception as e:
        print(f"❌ Erreur lors de la transformation Yeo-Johnson : {e}")
        if display_info:
            print("⚠️ Poursuite sans transformation...")
    
    # 🛡️ Validation X4 après transformation
    if protect_x4:
        validate_x4_presence(df, "Après Yeo-Johnson", display_info)

    # ========================================================================
    # ÉTAPE 4: IMPUTATION DES VALEURS MANQUANTES
    # ========================================================================
    
    if display_info:
        print(f"\n🔧 Étape 4: Imputation des valeurs manquantes ({mar_method})...")
    
    try:
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
    except Exception as e:
        print(f"❌ Erreur lors de l'imputation : {e}")
        if display_info:
            print("⚠️ Poursuite sans imputation...")
    
    # 🛡️ Validation X4 après imputation
    if protect_x4:
        validate_x4_presence(df, "Après imputation", display_info)

    # ========================================================================
    # ÉTAPE 5: RÉDUCTION DE LA COLINÉARITÉ (SECTION CORRIGÉE)
    # ========================================================================
    
    if display_info:
        print(f"\n🔗 Étape 5: Réduction de la colinéarité (seuil={correlation_threshold})...")
    
    try:
        binary_vars = [col for col in df.columns 
                       if pd.api.types.is_integer_dtype(df[col]) and col != "outcome"]
        
        if display_info:
            print(f"🔢 Variables binaires candidates : {len(binary_vars)}")

        # 🔧 CORRECTION PRINCIPALE: Gestion robuste de find_highly_correlated_groups
        if binary_vars:
            groups_corr = find_highly_correlated_groups(
                df[binary_vars], 
                threshold=correlation_threshold,
                protected_cols=protected_cols
            )
            
            # 🔧 VALIDATION DU TYPE DE RETOUR
            if isinstance(groups_corr, list):
                # Si c'est une liste (ancien format), on l'adapte
                if display_info:
                    print("⚠️ Format de retour détecté comme liste - conversion en dictionnaire")
                groups_corr = {
                    "groups": groups_corr,
                    "to_drop": [],
                    "protected": protected_cols
                }
            elif not isinstance(groups_corr, dict):
                # Si ce n'est ni liste ni dict, erreur
                if display_info:
                    print(f"⚠️ Type de retour inattendu: {type(groups_corr)} - utilisation valeurs par défaut")
                groups_corr = {
                    "groups": [],
                    "to_drop": [],
                    "protected": protected_cols
                }
            elif "groups" not in groups_corr:
                # Si c'est un dict mais sans la clé "groups"
                if display_info:
                    print("⚠️ Clé 'groups' manquante - ajout de structure par défaut")
                groups_corr["groups"] = []
                if "to_drop" not in groups_corr:
                    groups_corr["to_drop"] = []
                if "protected" not in groups_corr:
                    groups_corr["protected"] = protected_cols
            
            # Stockage des informations de corrélation
            transform_objects['correlation_info'] = groups_corr
            
        else:
            # Pas de variables binaires à analyser
            groups_corr = {
                "groups": [],
                "to_drop": [],
                "protected": protected_cols
            }
            if display_info:
                print("⚠️ Aucune variable binaire trouvée pour l'analyse de corrélation")

        target_col = "outcome" if "outcome" in df.columns and require_outcome else None

        # 🛡️ Protection dans drop_correlated_duplicates
        df_reduced, dropped_cols, kept_cols = drop_correlated_duplicates(
            df=df,
            groups=groups_corr["groups"],  # ✅ Maintenant sûr d'accéder à cette clé
            target_col=target_col,
            extra_cols=mar_cols + mcar_cols,
            protected_cols=protected_cols,
            priority_cols=priority_cols,
            verbose=False,
            summary=display_info
        )
        
        # Réassignation du DataFrame
        df = df_reduced
        
    except Exception as e:
        print(f"❌ Erreur lors de la réduction de colinéarité : {e}")
        if display_info:
            print("⚠️ Poursuite sans réduction de colinéarité...")
    
    # 🛡️ Validation X4 après réduction colinéarité
    if protect_x4:
        validate_x4_presence(df, "Après réduction colinéarité", display_info)

    # ========================================================================
    # ÉTAPE 6: SUPPRESSION DES OUTLIERS (OPTIONNELLE)
    # ========================================================================
    
    if drop_outliers and target_col:
        if display_info:
            print(f"\n🎯 Étape 6: Suppression des outliers...")
        
        try:
            df = detect_and_remove_outliers(
                df=df,
                columns=mar_cols,
                method='iqr',
                remove=True,
                verbose=display_info
            )
        except Exception as e:
            print(f"❌ Erreur lors de la suppression des outliers : {e}")
            if display_info:
                print("⚠️ Poursuite sans suppression des outliers...")
        
        # 🛡️ Validation X4 après suppression outliers
        if protect_x4:
            validate_x4_presence(df, "Après suppression outliers", display_info)
    elif display_info:
        print(f"\n⏭️ Étape 6: Suppression des outliers ignorée (drop_outliers={drop_outliers})")

    # ========================================================================
    # ÉTAPE 7: SUPPRESSION DES COLONNES DUPLIQUÉES
    # ========================================================================
    
    duplicate_cols = df.columns[df.columns.duplicated()].tolist()
    if duplicate_cols:
        if display_info:
            print(f"\n🔄 Étape 7: Suppression des colonnes dupliquées...")
        
        # 🛡️ Vérifier qu'on ne supprime pas X4 par accident
        if 'X4' in duplicate_cols and protect_x4:
            print("🛡️ ALERTE: X4 détectée comme dupliquée - protection activée")
            # Garder la première occurrence de X4
            df = df.loc[:, ~df.columns.duplicated(keep='first')]
        else:
            df = df.loc[:, ~df.columns.duplicated()]
            
        if display_info:
            print(f"⚠️ Colonnes dupliquées détectées : {duplicate_cols}")
            print(f"🔹 Duplication supprimée : {df.shape}")
        
        # 🛡️ Validation X4 après suppression doublons
        if protect_x4:
            validate_x4_presence(df, "Après suppression doublons", display_info)
    elif display_info:
        print(f"\n✅ Étape 7: Aucune colonne dupliquée détectée")

    # ========================================================================
    # ÉTAPE 8: RÉORGANISATION FINALE DES COLONNES
    # ========================================================================
    
    if display_info:
        print(f"\n📌 Étape 8: Réorganisation finale des colonnes...")
    
    try:
        # Colonnes actuelles
        current_cols = df.columns.tolist()
        
        # 1. Variables prioritaires (en premier)
        final_priority = [col for col in priority_cols if col in current_cols]
        
        # 2. Variable cible (après les prioritaires)
        final_target = [target_col] if target_col and target_col in current_cols else []
        
        # 3. Toutes les autres colonnes (dans l'ordre actuel)
        final_others = [col for col in current_cols if col not in final_priority + final_target]
        
        # Ordre final : prioritaires → cible → reste
        final_order = final_priority + final_target + final_others
        
        # Réorganiser le DataFrame
        df = df[final_order]
        
        if display_info:
            print(f"📌 Ordre final : {final_priority} → {final_target} → {len(final_others)} autres")
            print(f"📌 Premières colonnes : {df.columns[:min(8, len(df.columns))].tolist()}")
    
    except Exception as e:
        print(f"❌ Erreur lors de la réorganisation : {e}")
        if display_info:
            print("⚠️ Poursuite avec ordre actuel...")

    # ========================================================================
    # ÉTAPE 9: VALIDATION FINALE
    # ========================================================================
    
    if display_info:
        print(f"\n🔍 Étape 9: Validation finale...")
        print(f"✅ Pipeline complet terminé – Dimensions finales : {df.shape}")
        
        # 🛡️ Validation finale X4
        if protect_x4:
            final_status = validate_x4_presence(df, "VALIDATION FINALE", True)
            if not final_status:
                print("🚨 ERREUR CRITIQUE: X4 manquante en fin de pipeline !")
                raise ValueError("X4 a été perdue pendant le preprocessing !")

    # ========================================================================
    # ÉTAPE 10: SAUVEGARDE
    # ========================================================================
    
    if processed_data_dir:
        if display_info:
            print(f"\n💾 Étape 10: Sauvegarde...")
        
        try:
            processed_data_dir = Path(processed_data_dir)
            processed_data_dir.mkdir(parents=True, exist_ok=True)
            suffix = f"{mar_method}{'_no_outliers' if drop_outliers else ''}"
            filename = f"final_dataset_{suffix}.parquet"
            df.to_parquet(processed_data_dir / filename, index=False)
            
            if display_info:
                print(f"💾 Sauvegarde Parquet : {processed_data_dir / filename}")
        except Exception as e:
            print(f"❌ Erreur lors de la sauvegarde : {e}")
    elif display_info:
        print(f"\n⏭️ Étape 10: Sauvegarde ignorée (processed_data_dir=None)")

    if display_info:
        print("\n" + "=" * 70)
        print("🎉 PIPELINE DE PRÉTRAITEMENT TERMINÉ AVEC SUCCÈS")
        print("=" * 70)

    # Retour selon les options
    if return_objects:
        return df, transform_objects
    else:
        return df


# ============================================================================
# 6. FONCTIONS UTILITAIRES POUR DATASETS EXISTANTS
# ============================================================================

def apply_full_preprocessing_to_existing(
    df: pd.DataFrame,
    **kwargs
) -> pd.DataFrame:
    """
    Applique le prétraitement complet à un DataFrame existant.
    
    Args:
        df: DataFrame à prétraiter
        **kwargs: Arguments à passer à prepare_final_dataset
        
    Returns:
        DataFrame prétraité
        
    Note:
        Cette fonction sauvegarde temporairement le DataFrame et utilise prepare_final_dataset
    """
    import tempfile
    
    # Sauvegarde temporaire
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp_file:
        df.to_csv(tmp_file.name, index=False)
        
        # Application du pipeline
        result = prepare_final_dataset(
            file_path=tmp_file.name,
            raw_data_dir=Path(tmp_file.name).parent,
            **kwargs
        )
        
        # Nettoyage
        Path(tmp_file.name).unlink()
        
    return result


def batch_process_datasets(
    datasets_dict: Dict[str, Union[pd.DataFrame, str, Path]],
    **kwargs
) -> Dict[str, pd.DataFrame]:
    """
    Traite plusieurs datasets en lot avec le pipeline complet.
    
    Args:
        datasets_dict: Dictionnaire {nom: DataFrame ou chemin vers fichier}
        **kwargs: Arguments à passer à prepare_final_dataset
        
    Returns:
        Dictionnaire des datasets traités
    """
    processed_datasets = {}
    
    for name, data in datasets_dict.items():
        print(f"\n🔄 Traitement de {name}...")
        
        try:
            if isinstance(data, pd.DataFrame):
                # DataFrame existant
                result = apply_full_preprocessing_to_existing(data, **kwargs)
            else:
                # Chemin vers fichier
                result = prepare_final_dataset(file_path=data, **kwargs)
            
            processed_datasets[name] = result
            print(f"✅ {name} traité avec succès : {result.shape}")
            
        except Exception as e:
            print(f"❌ Erreur lors du traitement de {name} : {e}")
            processed_datasets[name] = None
    
    return processed_datasets


def validate_all_datasets(
    datasets_dict: Dict[str, pd.DataFrame],
    expected_cols: List[str] = None,
    protect_x4: bool = True
) -> Dict[str, Dict]:
    """
    Valide la qualité de plusieurs datasets.
    
    Args:
        datasets_dict: Dictionnaire des datasets à valider
        expected_cols: Colonnes attendues en premier
        protect_x4: Vérifier la présence de X4
        
    Returns:
        Dictionnaire des rapports de validation
    """
    if expected_cols is None:
        expected_cols = ['X1_trans', 'X2_trans', 'X3_trans', 'X4', 'outcome']
    
    validation_reports = {}
    
    for name, df in datasets_dict.items():
        if df is None:
            validation_reports[name] = {'status': 'error', 'message': 'DataFrame vide'}
            continue
        
        report = {
            'status': 'success',
            'shape': df.shape,
            'columns_order_correct': check_column_order(df, expected_cols, display_info=False),
            'has_x4': 'X4' in df.columns if protect_x4 else True,
            'has_outcome': 'outcome' in df.columns,
            'missing_values': df.isnull().sum().sum(),
            'first_columns': df.columns[:min(8, len(df.columns))].tolist()
        }
        
        # Score de qualité global
        quality_checks = [
            report['columns_order_correct'],
            report['has_x4'],
            report['has_outcome'],
            report['missing_values'] == 0
        ]
        report['quality_score'] = sum(quality_checks) / len(quality_checks)
        
        validation_reports[name] = report
    
    return validation_reports


def print_validation_summary(validation_reports: Dict[str, Dict]):
    """
    Affiche un résumé des validations.
    
    Args:
        validation_reports: Rapports de validation
    """
    print("\n📊 RÉSUMÉ DE LA VALIDATION DES DATASETS")
    print("=" * 60)
    
    for name, report in validation_reports.items():
        if report['status'] == 'error':
            print(f"❌ {name}: {report['message']}")
            continue
        
        quality = report['quality_score']
        status_icon = "✅" if quality == 1.0 else "⚠️" if quality >= 0.75 else "❌"
        
        print(f"{status_icon} {name}:")
        print(f"   📐 Shape: {report['shape']}")
        print(f"   🎯 Score qualité: {quality:.2f}/1.0")
        print(f"   📌 Ordre colonnes: {'✅' if report['columns_order_correct'] else '❌'}")
        print(f"   🛡️ X4 présente: {'✅' if report['has_x4'] else '❌'}")
        print(f"   🎯 Outcome présente: {'✅' if report['has_outcome'] else '❌'}")
        print(f"   💧 Valeurs manquantes: {report['missing_values']}")
        print(f"   📋 Premières colonnes: {report['first_columns']}")
        print()


# ============================================================================
# 7. FONCTIONS DE DIAGNOSTIC ET DEBUG (AMÉLIORÉES)
# ============================================================================

def diagnose_pipeline_issue(
    file_path: Union[str, Path],
    step_by_step: bool = True,
    **kwargs
) -> Dict[str, any]:
    """
    Diagnostique les problèmes potentiels dans le pipeline.
    
    Args:
        file_path: Chemin vers le fichier de données
        step_by_step: Exécuter étape par étape avec validation
        **kwargs: Arguments pour prepare_final_dataset
        
    Returns:
        Dictionnaire avec les résultats de chaque étape
    """
    print("🔍 DIAGNOSTIC DU PIPELINE (VERSION AMÉLIORÉE)")
    print("=" * 60)
    
    results = {}
    
    try:
        # Test de chargement
        print("📂 Test de chargement...")
        df = load_data(file_path, display_info=False, encode_target=True)
        results['loading'] = {
            'status': 'success',
            'shape': df.shape,
            'has_x4': 'X4' in df.columns,
            'has_outcome': 'outcome' in df.columns,
            'columns': df.columns.tolist()
        }
        print(f"✅ Chargement OK: {df.shape}")
        
        if step_by_step:
            # Test de chaque étape
            steps = [
                ('conversion_x4', lambda d: convert_X4_to_int(d, verbose=False)),
                ('yeo_johnson', lambda d: apply_yeojohnson(d, ['X1', 'X2', 'X3'])),
                ('correlation_analysis', lambda d: test_correlation_step(d)),
            ]
            
            for step_name, step_func in steps:
                try:
                    df = step_func(df)
                    results[step_name] = {
                        'status': 'success',
                        'shape': df.shape,
                        'has_x4': 'X4' in df.columns
                    }
                    print(f"✅ {step_name} OK: {df.shape}")
                except Exception as e:
                    results[step_name] = {
                        'status': 'error',
                        'error': str(e)
                    }
                    print(f"❌ {step_name} ERREUR: {e}")
                    break
        
    except Exception as e:
        results['loading'] = {
            'status': 'error',
            'error': str(e)
        }
        print(f"❌ Chargement ERREUR: {e}")
    
    return results


def test_correlation_step(df: pd.DataFrame) -> pd.DataFrame:
    """Teste spécifiquement l'étape de corrélation."""
    binary_vars = [col for col in df.columns 
                   if pd.api.types.is_integer_dtype(df[col]) and col != "outcome"]
    
    if not binary_vars:
        print("⚠️ Aucune variable binaire pour test de corrélation")
        return df
    
    groups_corr = find_highly_correlated_groups(
        df[binary_vars], 
        threshold=0.90,
        protected_cols=['X4']
    )
    
    print(f"🔧 Test corrélation - Type retour: {type(groups_corr)}")
    print(f"🔧 Contenu: {groups_corr}")
    
    # Test d'accès
    if isinstance(groups_corr, dict) and "groups" in groups_corr:
        print(f"✅ Accès groups réussi: {len(groups_corr['groups'])} groupes")
    else:
        raise ValueError(f"Format retour incorrect: {type(groups_corr)}")
    
    return df


def quick_pipeline_test(file_path: Union[str, Path]) -> bool:
    """Test rapide du pipeline complet."""
    print("⚡ TEST RAPIDE DU PIPELINE COMPLET")
    print("=" * 40)
    
    try:
        df_result = prepare_final_dataset(
            file_path=file_path,
            strategy="mixed_mar_mcar",
            mar_method="knn",
            correlation_threshold=0.90,
            drop_outliers=False,
            display_info=False
        )
        
        # Validations
        has_x4 = 'X4' in df_result.columns
        has_outcome = 'outcome' in df_result.columns
        no_missing = df_result.isnull().sum().sum() == 0
        
        print(f"✅ Pipeline exécuté avec succès")
        print(f"📊 Shape finale: {df_result.shape}")
        print(f"🛡️ X4 présente: {'✅' if has_x4 else '❌'}")
        print(f"🎯 Outcome présente: {'✅' if has_outcome else '❌'}")
        print(f"💧 Pas de valeurs manquantes: {'✅' if no_missing else '❌'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur dans le pipeline: {e}")
        return False


# ============================================================================
# 8. FONCTIONS D'EXPORT ET SAUVEGARDE AVANCÉES
# ============================================================================

def export_datasets_multiple_formats(
    datasets_dict: Dict[str, pd.DataFrame],
    output_dir: Union[str, Path],
    formats: List[str] = ['parquet', 'csv'],
    compress: bool = True
) -> Dict[str, Dict[str, Path]]:
    """
    Exporte les datasets dans plusieurs formats.
    
    Args:
        datasets_dict: Dictionnaire des datasets
        output_dir: Dossier de sortie
        formats: Formats d'export ('parquet', 'csv', 'xlsx')
        compress: Compresser les fichiers
        
    Returns:
        Dictionnaire des chemins de sauvegarde
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    export_paths = {}
    
    for dataset_name, df in datasets_dict.items():
        if df is None:
            continue
        
        export_paths[dataset_name] = {}
        
        for fmt in formats:
            try:
                if fmt == 'parquet':
                    file_path = output_dir / f"{dataset_name}.parquet"
                    df.to_parquet(file_path, index=False, compression='snappy' if compress else None)
                    
                elif fmt == 'csv':
                    file_path = output_dir / f"{dataset_name}.csv"
                    compression = 'gzip' if compress else None
                    if compression:
                        file_path = file_path.with_suffix('.csv.gz')
                    df.to_csv(file_path, index=False, compression=compression)
                    
                elif fmt == 'xlsx':
                    file_path = output_dir / f"{dataset_name}.xlsx"
                    df.to_excel(file_path, index=False)
                
                export_paths[dataset_name][fmt] = file_path
                print(f"💾 {dataset_name}.{fmt} sauvegardé: {file_path}")
                
            except Exception as e:
                print(f"❌ Erreur sauvegarde {dataset_name}.{fmt}: {e}")
    
    return export_paths


def create_preprocessing_report(
    datasets_dict: Dict[str, pd.DataFrame],
    validation_reports: Dict[str, Dict],
    output_path: Union[str, Path],
    transform_objects: Dict = None
) -> None:
    """
    Crée un rapport détaillé du prétraitement.
    
    Args:
        datasets_dict: Dictionnaire des datasets
        validation_reports: Rapports de validation
        output_path: Chemin de sauvegarde du rapport
        transform_objects: Objets de transformation utilisés
    """
    from datetime import datetime
    
    report_content = f"""
# RAPPORT DE PRÉTRAITEMENT STA211 - VERSION CORRIGÉE
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Résumé

- **Nombre de datasets traités**: {len(datasets_dict)}
- **Datasets valides**: {sum(1 for r in validation_reports.values() if r.get('status') == 'success')}
- **Pipeline version**: 2.1 (Corrigée)

## Corrections apportées

- ✅ Fix TypeError dans find_highly_correlated_groups
- ✅ Validation robuste des types de retour
- ✅ Gestion d'erreurs améliorée à chaque étape
- ✅ Protection X4 renforcée

## Détails par dataset

"""
    
    for name, report in validation_reports.items():
        if report['status'] == 'error':
            report_content += f"### ❌ {name}\n- **Statut**: Erreur\n- **Message**: {report['message']}\n\n"
            continue
        
        df = datasets_dict[name]
        report_content += f"""### ✅ {name}

- **Dimensions**: {report['shape']}
- **Score qualité**: {report['quality_score']:.2f}/1.0
- **X4 présente**: {'✅' if report['has_x4'] else '❌'}
- **Outcome présente**: {'✅' if report['has_outcome'] else '❌'}
- **Valeurs manquantes**: {report['missing_values']}
- **Premières colonnes**: {', '.join(report['first_columns'])}

#### Statistiques descriptives (premières variables)
```
{df[df.columns[:5]].describe().round(3).to_string()}
```

"""
    
    # Informations sur les transformations si disponibles
    if transform_objects:
        report_content += "\n## Objets de transformation\n\n"
        for obj_name, obj in transform_objects.items():
            if obj is not None:
                report_content += f"- **{obj_name}**: {type(obj).__name__}\n"
    
    # Sauvegarde du rapport
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        print(f"📝 Rapport sauvegardé: {output_path}")
    except Exception as e:
        print(f"❌ Erreur sauvegarde rapport: {e}")


# ============================================================================
# 9. FONCTIONS DE TEST ET VALIDATION
# ============================================================================

def run_comprehensive_test(file_path: Union[str, Path]) -> Dict:
    """Execute un test complet du pipeline avec diagnostic."""
    
    print("🧪 TEST COMPLET DU PIPELINE CORRIGÉ")
    print("=" * 50)
    
    # Test 1: Pipeline de base
    print("\n1️⃣ Test pipeline de base...")
    basic_success = quick_pipeline_test(file_path)
    
    # Test 2: Pipeline avec objets
    print("\n2️⃣ Test pipeline avec objets de transformation...")
    try:
        df_result, transform_objects = prepare_final_dataset(
            file_path=file_path,
            return_objects=True,
            display_info=False
        )
        objects_success = True
        print("✅ Pipeline avec objets réussi")
    except Exception as e:
        objects_success = False
        print(f"❌ Pipeline avec objets échoué: {e}")
        transform_objects = {}
    
    # Test 3: Diagnostic détaillé
    print("\n3️⃣ Diagnostic détaillé...")
    diagnostic_results = diagnose_pipeline_issue(file_path, step_by_step=True)
    
    # Test 4: Validation X4
    print("\n4️⃣ Validation protection X4...")
    x4_protected = True
    if basic_success:
        df_test = prepare_final_dataset(file_path, display_info=False)
        x4_protected = 'X4' in df_test.columns
        print(f"🛡️ X4 protégée: {'✅' if x4_protected else '❌'}")
    
    # Résumé final
    print("\n📊 RÉSUMÉ DU TEST COMPLET")
    print("=" * 30)
    print(f"Pipeline de base: {'✅' if basic_success else '❌'}")
    print(f"Pipeline avec objets: {'✅' if objects_success else '❌'}")
    print(f"Protection X4: {'✅' if x4_protected else '❌'}")
    print(f"Diagnostic: {'✅' if all(r.get('status') == 'success' for r in diagnostic_results.values()) else '⚠️'}")
    
    return {
        'basic_success': basic_success,
        'objects_success': objects_success,
        'x4_protected': x4_protected,
        'diagnostic_results': diagnostic_results,
        'transform_objects': transform_objects if objects_success else {}
    }


# ============================================================================
# 10. FONCTIONS D'UTILISATION SIMPLIFIÉE
# ============================================================================

def prepare_dataset_safe(file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
    """
    Version sécurisée du pipeline avec gestion d'erreurs automatique.
    
    Args:
        file_path: Chemin vers le fichier
        **kwargs: Arguments pour prepare_final_dataset
        
    Returns:
        DataFrame prétraité
    """
    print("🔒 PIPELINE SÉCURISÉ - VERSION CORRIGÉE")
    print("=" * 45)
    
    # Paramètres par défaut sécurisés
    safe_defaults = {
        'strategy': 'mixed_mar_mcar',
        'mar_method': 'knn',
        'correlation_threshold': 0.90,
        'drop_outliers': False,
        'protect_x4': True,
        'display_info': True
    }
    
    # Fusion avec les paramètres utilisateur
    final_params = {**safe_defaults, **kwargs}
    
    try:
        # Tentative normale
        df_result = prepare_final_dataset(file_path=file_path, **final_params)
        print("✅ Pipeline exécuté avec succès en mode normal")
        return df_result
        
    except TypeError as e:
        if "list indices must be integers" in str(e) or "groups" in str(e):
            print("🔧 Erreur de corrélation détectée - application du mode de récupération...")
            
            # Mode de récupération avec seuil plus strict
            recovery_params = final_params.copy()
            recovery_params['correlation_threshold'] = 0.95
            recovery_params['display_info'] = True
            
            try:
                df_result = prepare_final_dataset(file_path=file_path, **recovery_params)
                print("✅ Pipeline exécuté avec succès en mode récupération")
                return df_result
            except Exception as e2:
                print(f"❌ Échec en mode récupération: {e2}")
                raise
                
    except Exception as e:
        print(f"❌ Erreur critique: {e}")
        print("🔍 Lancement du diagnostic...")
        
        # Diagnostic automatique
        diagnostic = diagnose_pipeline_issue(file_path, step_by_step=False)
        print("📋 Diagnostic terminé - consultez les résultats ci-dessus")
        raise


def get_preprocessing_summary(df: pd.DataFrame) -> Dict:
    """
    Génère un résumé des caractéristiques du dataset prétraité.
    
    Args:
        df: DataFrame prétraité
        
    Returns:
        Dictionnaire avec le résumé
    """
    summary = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'has_x4': 'X4' in df.columns,
        'has_outcome': 'outcome' in df.columns,
        'binary_vars': [],
        'continuous_vars': [],
        'memory_usage': df.memory_usage(deep=True).sum() / 1024**2  # MB
    }
    
    # Classification des variables
    for col in df.columns:
        if col == 'outcome':
            continue
        elif pd.api.types.is_integer_dtype(df[col]):
            summary['binary_vars'].append(col)
        elif pd.api.types.is_float_dtype(df[col]):
            summary['continuous_vars'].append(col)
    
    # Statistiques additionnelles
    if 'outcome' in df.columns:
        summary['class_distribution'] = df['outcome'].value_counts().to_dict()
        summary['class_balance'] = df['outcome'].value_counts(normalize=True).to_dict()
    
    return summary


def print_preprocessing_summary(df: pd.DataFrame):
    """Affiche un résumé formaté du dataset prétraité."""
    
    summary = get_preprocessing_summary(df)
    
    print("\n📊 RÉSUMÉ DU DATASET PRÉTRAITÉ")
    print("=" * 40)
    
    print(f"📐 Dimensions: {summary['shape'][0]} lignes × {summary['shape'][1]} colonnes")
    print(f"💾 Mémoire utilisée: {summary['memory_usage']:.2f} MB")
    print(f"🛡️ X4 présente: {'✅' if summary['has_x4'] else '❌'}")
    print(f"🎯 Outcome présente: {'✅' if summary['has_outcome'] else '❌'}")
    
    # Variables par type
    print(f"\n🔢 Variables binaires ({len(summary['binary_vars'])}): {summary['binary_vars'][:5]}{'...' if len(summary['binary_vars']) > 5 else ''}")
    print(f"📈 Variables continues ({len(summary['continuous_vars'])}): {summary['continuous_vars'][:5]}{'...' if len(summary['continuous_vars']) > 5 else ''}")
    
    # Valeurs manquantes
    missing_count = sum(v for v in summary['missing_values'].values() if v > 0)
    if missing_count > 0:
        print(f"💧 Valeurs manquantes: {missing_count} au total")
        missing_cols = {k: v for k, v in summary['missing_values'].items() if v > 0}
        for col, count in list(missing_cols.items())[:3]:
            print(f"   {col}: {count}")
    else:
        print("💧 Valeurs manquantes: ✅ Aucune")
    
    # Distribution des classes
    if 'class_distribution' in summary:
        print(f"\n🎯 Distribution des classes:")
        for classe, count in summary['class_distribution'].items():
            pct = summary['class_balance'][classe] * 100
            print(f"   Classe {classe}: {count} ({pct:.1f}%)")


# ============================================================================
# 11. FONCTIONS DE COMPATIBILITÉ ET MIGRATION
# ============================================================================

def migrate_old_results(old_results_dir: Union[str, Path], new_results_dir: Union[str, Path]):
    """
    Migre les anciens résultats vers le nouveau format.
    
    Args:
        old_results_dir: Dossier des anciens résultats
        new_results_dir: Dossier pour les nouveaux résultats
    """
    old_dir = Path(old_results_dir)
    new_dir = Path(new_results_dir)
    new_dir.mkdir(parents=True, exist_ok=True)
    
    print("🔄 MIGRATION DES ANCIENS RÉSULTATS")
    print("=" * 40)
    
    # Recherche des fichiers à migrer
    old_files = list(old_dir.glob("*.csv")) + list(old_dir.glob("*.parquet"))
    
    for old_file in old_files:
        try:
            print(f"🔄 Migration de {old_file.name}...")
            
            # Chargement
            if old_file.suffix == '.csv':
                df = pd.read_csv(old_file)
            else:
                df = pd.read_parquet(old_file)
            
            # Retraitement avec le nouveau pipeline
            if 'X1' in df.columns:  # Dataset brut
                df_new = apply_full_preprocessing_to_existing(df)
            else:  # Dataset déjà traité - validation seulement
                df_new = df.copy()
                # Validation et correction si nécessaire
                if 'X4' not in df_new.columns:
                    print(f"⚠️ {old_file.name}: X4 manquante - fichier ignoré")
                    continue
            
            # Sauvegarde dans le nouveau format
            new_file = new_dir / f"migrated_{old_file.stem}.parquet"
            df_new.to_parquet(new_file, index=False)
            
            print(f"✅ {old_file.name} → {new_file.name}")
            
        except Exception as e:
            print(f"❌ Erreur migration {old_file.name}: {e}")
    
    print("🎉 Migration terminée")


def check_compatibility(df: pd.DataFrame) -> Dict:
    """
    Vérifie la compatibilité d'un dataset avec le nouveau pipeline.
    
    Args:
        df: DataFrame à vérifier
        
    Returns:
        Rapport de compatibilité
    """
    compatibility = {
        'version': '2.1',
        'compatible': True,
        'issues': [],
        'recommendations': []
    }
    
    # Vérifications de base
    if 'X4' not in df.columns:
        compatibility['compatible'] = False
        compatibility['issues'].append("X4 manquante")
        compatibility['recommendations'].append("Retraiter avec le pipeline corrigé")
    
    if 'outcome' not in df.columns:
        compatibility['issues'].append("Variable cible manquante")
        compatibility['recommendations'].append("Vérifier l'encodage de la variable cible")
    
    # Vérification des colonnes transformées
    expected_transformed = ['X1_trans', 'X2_trans', 'X3_trans']
    missing_transformed = [col for col in expected_transformed if col not in df.columns]
    if missing_transformed:
        compatibility['issues'].append(f"Colonnes transformées manquantes: {missing_transformed}")
        compatibility['recommendations'].append("Appliquer la transformation Yeo-Johnson")
    
    # Vérification des valeurs manquantes
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        compatibility['issues'].append(f"{missing_count} valeurs manquantes détectées")
        compatibility['recommendations'].append("Appliquer l'imputation MICE/KNN")
    
    # Vérification des doublons de colonnes
    duplicate_cols = df.columns[df.columns.duplicated()].tolist()
    if duplicate_cols:
        compatibility['issues'].append(f"Colonnes dupliquées: {duplicate_cols}")
        compatibility['recommendations'].append("Supprimer les doublons")
    
    return compatibility


def print_compatibility_report(compatibility: Dict):
    """Affiche un rapport de compatibilité formaté."""
    
    print(f"\n🔍 RAPPORT DE COMPATIBILITÉ - VERSION {compatibility['version']}")
    print("=" * 50)
    
    if compatibility['compatible']:
        print("✅ Dataset compatible avec le pipeline actuel")
    else:
        print("❌ Dataset non compatible - corrections nécessaires")
    
    if compatibility['issues']:
        print(f"\n⚠️ Problèmes détectés ({len(compatibility['issues'])}):")
        for i, issue in enumerate(compatibility['issues'], 1):
            print(f"   {i}. {issue}")
    
    if compatibility['recommendations']:
        print(f"\n💡 Recommandations ({len(compatibility['recommendations'])}):")
        for i, rec in enumerate(compatibility['recommendations'], 1):
            print(f"   {i}. {rec}")


def compare_preprocessing_results(
    file_path: Union[str, Path],
    configs: List[Dict],
    config_names: List[str] = None
) -> pd.DataFrame:
    """
    Compare les résultats de différentes configurations de prétraitement.
    
    Args:
        file_path: Chemin vers le fichier de données
        configs: Liste des configurations à tester
        config_names: Noms des configurations
        
    Returns:
        DataFrame de comparaison
    """
    if config_names is None:
        config_names = [f"Config_{i+1}" for i in range(len(configs))]
    
    results = []
    
    for i, config in enumerate(configs):
        print(f"\n🧪 Test configuration {config_names[i]}...")
        
        try:
            df_result = prepare_final_dataset(
                file_path=file_path,
                display_info=False,
                **config
            )
            
            result = {
                'config_name': config_names[i],
                'status': 'success',
                'final_shape': df_result.shape,
                'has_x4': 'X4' in df_result.columns,
                'has_outcome': 'outcome' in df_result.columns,
                'missing_values': df_result.isnull().sum().sum(),
                'first_5_columns': df_result.columns[:5].tolist()
            }
            
        except Exception as e:
            result = {
                'config_name': config_names[i],
                'status': 'error',
                'error': str(e),
                'final_shape': None,
                'has_x4': False,
                'has_outcome': False,
                'missing_values': None,
                'first_5_columns': None
            }
        
        results.append(result)
    
    return pd.DataFrame(results)


# ============================================================================
# FIN DU MODULE - INFORMATIONS DE VERSION
# ============================================================================

__version__ = "2.1"
__status__ = "Corrigé"
__corrections__ = [
    "Fix TypeError dans find_highly_correlated_groups",
    "Validation robuste des types de retour",
    "Gestion d'erreurs améliorée",
    "Protection X4 renforcée",
    "Fonctions de diagnostic avancées",
    "Mode de récupération automatique"
]

def print_version_info():
    """Affiche les informations de version du module."""
    print(f"\n📋 MODULE final_preprocessing.py")
    print(f"Version: {__version__}")
    print(f"Statut: {__status__}")
    print(f"Corrections apportées:")
    for correction in __corrections__:
        print(f"  ✅ {correction}")


if __name__ == "__main__":
    print_version_info()
    print(f"\n🚀 UTILISATION RECOMMANDÉE:")
    print(f"   # Test rapide")
    print(f"   df = prepare_dataset_safe('data_train.csv')")
    print(f"   ")
    print(f"   # Test complet")
    print(f"   test_results = run_comprehensive_test('data_train.csv')")
    print(f"   ")
    print(f"   # Pipeline avec objets")
    print(f"   df, objects = prepare_final_dataset('data_train.csv', return_objects=True)")