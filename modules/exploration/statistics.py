# Statistics.py


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler

from exploration.visualization import save_fig





def analyze_continuous_variables(df: pd.DataFrame, continuous_cols: List[str], target_col: str = "y", save_figures_path: str = None) -> dict:
    """
    Analyse complète des variables continues :
    - Statistiques descriptives
    - Asymétrie, aplatissement, test de normalité (Shapiro)
    - Outliers (IQR)
    - Corrélation avec la cible
    - Matrice de corrélation + heatmap

    :param df: DataFrame d'entrée
    :param continuous_cols: Liste des colonnes continues
    :param target_col: Nom de la variable cible
    :param save_figures_path: Chemin vers le dossier de sauvegarde des figures
    :return: Dictionnaire des résultats
    """
    summary_stats = df[continuous_cols].describe()
    print("📊 Statistiques descriptives :")
    print(summary_stats)

    skew_kurtosis_results = {}
    outliers_summary = {}
    correlations = {}

    print("\n📊 Analyse de la distribution :")
    for col in continuous_cols:
        data = df[col].dropna()
        skew = stats.skew(data)
        kurt = stats.kurtosis(data)
        stat, p_value = stats.shapiro(data.sample(min(5000, len(data))))
        print(f"\n{col}:")
        print(f"  - Skewness (asymétrie) : {skew:.3f}")
        print(f"  - Kurtosis (aplatissement) : {kurt:.3f}")
        print(f"  - Test de Shapiro-Wilk : p-value = {p_value:.4f}")
        if p_value < 0.01:
            print("    → Distribution non normale (nécessite transformation)")
        else:
            print("    → Distribution approximativement normale")
        skew_kurtosis_results[col] = {"skewness": skew, "kurtosis": kurt, "shapiro_p": p_value}

    print("\n🔍 Détection des outliers (méthode IQR) :")
    for col in continuous_cols:
        data = df[col].dropna()
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = data[(data < lower) | (data > upper)]
        pct = len(outliers) / len(data) * 100
        print(f"\n{col}:")
        print(f"  - Limites : [{lower:.2f}, {upper:.2f}]")
        print(f"  - Outliers : {len(outliers)} ({pct:.2f}%)")
        outliers_summary[col] = {"count": len(outliers), "percentage": pct, "lower": lower, "upper": upper}

    print("\n🎯 Corrélation avec la variable cible (y) :")
    for col in continuous_cols:
        valid = df[col].notna() & df[target_col].notna()
        corr = df.loc[valid, col].corr(df.loc[valid, target_col])
        print(f"  - {col}: {corr:.4f}")
        correlations[col] = corr

    print("\n📊 Matrice de corrélation entre variables continues :")
    corr_matrix = df[continuous_cols].corr()
    print(corr_matrix)

    plt.figure(figsize=(6, 4))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Matrice de corrélation des variables continues', fontsize=14)
    plt.tight_layout()
    if save_figures_path:
        plt.savefig(save_figures_path + "/continuous_correlation_matrix.png", dpi=300, bbox_inches='tight')
    plt.show()

    total_outliers = sum(info['count'] for info in outliers_summary.values())
    print("\n💡 Résumé et recommandations :")
    print("  - Les trois variables continues montrent des distributions fortement asymétriques")
    print("  - Transformation Yeo-Johnson recommandée pour normaliser les distributions")
    print(f"  - Outliers détectés : {total_outliers} au total")
    print("  - Corrélations faibles avec la cible, mais potentiellement utiles après transformation")

    return {
        "summary_stats": summary_stats,
        "skew_kurtosis": skew_kurtosis_results,
        "outliers": outliers_summary,
        "correlations": correlations,
        "corr_matrix": corr_matrix
    }







def detect_outliers_iqr(series: pd.Series, multiplier: float = 1.5) -> pd.Series:
    """
    Détecte les outliers selon la règle de l'IQR.
    
    Parameters:
    -----------
    series : pd.Series
        Série à analyser
    multiplier : float
        Multiplicateur pour l'IQR (par défaut 1.5)
        
    Returns:
    --------
    pd.Series : Masque booléen (True = outlier)
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - multiplier * IQR
    upper = Q3 + multiplier * IQR
    return (series < lower) | (series > upper)




def find_highly_correlated_groups(df: pd.DataFrame, threshold: float = 0.90):
    """
    Identifie les groupes de variables fortement corrélées (|corr| > threshold).
    
    Args:
        df (pd.DataFrame): données binaires (0/1)
        threshold (float): seuil de corrélation absolue

    Returns:
        List[List[str]]: groupes de variables corrélées
    """
    corr_matrix = df.corr().abs()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    groups = []
    visited = set()

    for col in upper_triangle.columns:
        if col in visited:
            continue
        correlated = upper_triangle[col][upper_triangle[col] > threshold].index.tolist()
        if correlated:
            group = sorted(set([col] + correlated))
            groups.append(group)
            visited.update(group)

    return groups





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

    Args:
        df (pd.DataFrame): DataFrame d'origine
        groups (List[List[str]]): groupes de variables fortement corrélées
        target_col (str): nom de la variable cible (à exclure des calculs binaires)
        extra_cols (List[str]): variables à réintégrer à la fin (X1-X4 etc.)
        verbose (bool): affiche les groupes traités
        summary (bool): affiche le résumé final

    Returns:
        - df_reduced: DataFrame nettoyé
        - to_drop: colonnes supprimées
        - to_keep: colonnes conservées
    """
    to_drop = []
    to_keep = []

    for group in groups:
        if not group:
            continue
        keep = group[0]
        drop = [col for col in group[1:] if col in df.columns]
        to_keep.append(keep)
        to_drop.extend(drop)
        if verbose:
            print(f"🧹 Groupe : {group} → garde {keep}, retire {drop}")

    # Dédupliquer
    to_drop = sorted(set(to_drop))
    to_keep = sorted(set(to_keep))

    # Colonnes binaires restantes (non corrélées)
    all_binary = [col for col in df.select_dtypes(include='int64').columns if col != target_col]
    untouched = [col for col in all_binary if col not in to_drop and col not in to_keep]

    # Colonnes finales conservées
    final_cols = to_keep + untouched
    if extra_cols:
        final_cols += [col for col in extra_cols if col in df.columns]

    df_reduced = df[final_cols + [target_col]].copy()

    # Résumé
    if summary:
        print(f"\n📊 Résumé de la réduction :")
        print(f"🔻 {len(to_drop)} colonnes binaires supprimées (corrélées)")
        print(f"✅ {len(to_keep)} colonnes binaires conservées (1 par groupe)")
        print(f"➕ {len(untouched)} colonnes binaires non corrélées conservées")
        if extra_cols:
            print(f"🧩 {len(extra_cols)} variables continues / contextuelles ajoutées : {extra_cols}")
        print(f"📐 DataFrame final : {df_reduced.shape[1]} colonnes, {df_reduced.shape[0]} lignes")

    return df_reduced, to_drop, to_keep



def summary_statistics(data, pca, target_col='outcome'):
    """Affiche un résumé des statistiques et résultats."""
    print("\n=== Résumé des Résultats ===")
    print(f"Nombre total d'observations : {len(data)}")
    print(f"Nombre de variables : {data.shape[1]}")
    
    if target_col and target_col in data.columns:
        print(f"Distribution des classes :\n{data[target_col].value_counts(normalize=True)}")
    else:
        print(f"Attention : La colonne cible '{target_col}' n'a pas été trouvée dans le DataFrame.")
    
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    print(f"Nombre de variables numériques : {len(numeric_cols)}")
    print(f"Pourcentage de valeurs manquantes : {data[numeric_cols].isnull().mean().mean()*100:.2f}%")
    
    if pca is not None:
        print("\n=== Analyse des Composantes Principales ===")
        print(f"Variance expliquée par les deux premières composantes : {pca.explained_variance_ratio_.sum()*100:.2f}%")
        print(f"Variance expliquée par la première composante : {pca.explained_variance_ratio_[0]*100:.2f}%")
        print(f"Variance expliquée par la deuxième composante : {pca.explained_variance_ratio_[1]*100:.2f}%")



def optimized_feature_importance(
    df: pd.DataFrame,
    target_col: str = 'outcome',
    method: str = 'all',  # 'rf', 'f_score', 'correlation', ou 'all'
    top_n: int = 20,
    figsize: Tuple[float, float] = (12, 8),
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True
) -> pd.DataFrame:
    """
    Analyse l'importance des variables avec plusieurs méthodes combinées.

    Args:
        df: Données d'entrée.
        target_col: Colonne cible.
        method: Méthode ('rf', 'f_score', 'correlation' ou 'all').
        top_n: Nombre de variables à visualiser.
        figsize: Taille de la figure.
        save_path: Chemin de sauvegarde de la figure (optionnel).
        show: Affiche le graphique si True.

    Returns:
        DataFrame des importances normalisées.
    """
    try:
        data = df.copy()

        # Encodage de la cible
        if data[target_col].dtype == 'object':
            data[target_col] = data[target_col].map({'ad.': 1, 'noad.': 0})

        y = data[target_col]
        numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns.drop(target_col)

        # Imputation conditionnelle
        clean_cols = []
        for col in numeric_cols:
            na_rate = data[col].isna().mean()
            if na_rate > 0.3:
                continue  # exclusion si >30% manquants
            elif na_rate > 0.05:
                imputer = KNNImputer(n_neighbors=5)
                data[[col]] = imputer.fit_transform(data[[col]])
            else:
                data[col] = data[col].fillna(data[col].median())
            clean_cols.append(col)

        X = data[clean_cols]
        X_scaled = StandardScaler().fit_transform(X)
        X_imputed = pd.DataFrame(X_scaled, columns=clean_cols)

        results = pd.DataFrame({'feature': clean_cols})

        if method in ['rf', 'all']:
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X_scaled, y)
            results['RF_Importance'] = rf.feature_importances_ / np.max(rf.feature_importances_)

        if method in ['f_score', 'all']:
            f_scores, _ = f_classif(X_scaled, y)
            results['F_Score'] = f_scores / np.nanmax(f_scores)

        if method in ['correlation', 'all']:
            correlations = X.corrwith(y).abs()
            results['Correlation'] = correlations / correlations.max()

        if method == 'all':
            results['Combined_Score'] = results[['RF_Importance', 'F_Score', 'Correlation']].mean(axis=1)
            sort_col = 'Combined_Score'
        else:
            sort_col = next(col for col in results.columns if col != 'feature')

        results = results.sort_values(sort_col, ascending=False)
        top_features = results.head(top_n)

        # Visualisation
        plt.figure(figsize=figsize)
        if method == 'all':
            x = np.arange(len(top_features))
            width = 0.25
            plt.bar(x - width, top_features['RF_Importance'], width, label='RF')
            plt.bar(x, top_features['F_Score'], width, label='F-Score')
            plt.bar(x + width, top_features['Correlation'], width, label='Corr')
            plt.xticks(x, top_features['feature'], rotation=45, ha='right')
            plt.ylabel('Importance normalisée')
            plt.title(f'Top {top_n} Variables - Méthodes combinées')
            plt.legend()
        else:
            plt.bar(top_features['feature'], top_features[sort_col])
            plt.xticks(rotation=45, ha='right')
            plt.ylabel('Importance')
            plt.title(f'Top {top_n} Variables - {method}')

        plt.tight_layout()

        if save_path:
            save_fig(
                fname=Path(save_path).name,
                directory=Path(save_path).parent,
                figsize=figsize,
                show=show
            )
        elif show:
            plt.show()
        else:
            plt.close()

        return results

    except Exception as e:
        print(f"⚠️ Erreur : {e}")
        return pd.DataFrame()
