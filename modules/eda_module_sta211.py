"""
Module d'analyse exploratoire pour le projet STA211.
Ce module contient des fonctions pour analyser les données du dataset Internet Advertisements.
"""
import os
from pathlib import Path
from project_setup import setup_project_paths
paths = setup_project_paths()

import pandas as pd
from scipy.stats import pointbiserialr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Optional, Union
import warnings

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2_contingency
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, f_classif
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from imblearn.over_sampling import SMOTE

# Import optionnel pour UMAP et prince
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    warnings.warn("UMAP not installed. Install with: pip install umap-learn")

try:
    import prince
    PRINCE_AVAILABLE = True
except ImportError:
    PRINCE_AVAILABLE = False
    warnings.warn("Prince not installed. Install with: pip install prince")

warnings.filterwarnings('ignore')

# Configuration des graphiques
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

__all__ = [
    'univariate_analysis',
    'bivariate_analysis',
    'multivariate_analysis',
    'dimension_reduction',
    'umap_visualization',
    'famd_analysis',
    'summary_statistics',
    'advanced_dimension_reduction',
    'perform_exploratory_analysis',
    'compare_visualization_methods',
    'analyze_feature_importance',
    'create_polynomial_features',
    'create_interaction_features',
    'enhance_features',
    'optimize_hyperparameters',
    'evaluate_optimized_models',
    'analyze_categorical_binaries_vs_target',
    'save_fig',
    'detect_outliers_iqr',
    'plot_correlation_heatmap',
    'analyze_class_distribution',
    'plot_missing_patterns',
    'plot_binary_correlation_heatmap',
    'find_highly_correlated_groups',
    'drop_correlated_duplicates'
]




def analyze_class_distribution(df: pd.DataFrame, target_col: str = 'y', 
                             figsize: Tuple[float, float] = (10, 5)) -> Dict[str, float]:
    """
    Analyse la distribution de la variable cible avec visualisations.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame contenant les données
    target_col : str
        Nom de la colonne cible
    figsize : tuple
        Taille de la figure
        
    Returns:
    --------
    dict : Statistiques sur la distribution
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Distribution counts
    if df[target_col].dtype == 'object':
        # Mapper les valeurs textuelles
        value_map = {'ad.': 1, 'noad.': 0}
        target_numeric = df[target_col].map(value_map)
        target_counts = target_numeric.value_counts()
    else:
        target_counts = df[target_col].value_counts()
    
    target_pct = target_counts / len(df) * 100
    
    # Barplot
    target_counts.plot(kind='bar', ax=ax1, color=['#3498db', '#e74c3c'])
    ax1.set_title('Distribution des classes')
    ax1.set_xlabel('Classe')
    ax1.set_ylabel("Nombre d'échantillons")
    ax1.set_xticklabels(['Non-publicité (0)', 'Publicité (1)'], rotation=0)
    
    # Pie chart
    target_pct.plot(kind='pie', ax=ax2, colors=['#3498db', '#e74c3c'], 
                    autopct='%1.1f%%', startangle=90)
    ax2.set_title('Proportion des classes')
    ax2.set_ylabel('')
    
    plt.tight_layout()
    
    # Calcul des statistiques
    imbalance_ratio = target_counts.max() / target_counts.min()
    
    stats = {
        'count_class_0': int(target_counts.get(0, 0)),
        'count_class_1': int(target_counts.get(1, 0)),
        'pct_class_0': float(target_pct.get(0, 0)),
        'pct_class_1': float(target_pct.get(1, 0)),
        'imbalance_ratio': float(imbalance_ratio),
        'minority_class': int(target_counts.idxmin()),
        'majority_class': int(target_counts.idxmax())
    }
    
    return stats






def famd_analysis(df: pd.DataFrame, n_components: int = 2, 
                  target_col: str = 'outcome', sample_size: int = 1000) -> Tuple[pd.DataFrame, object]:
    """
    Effectue une Analyse Factorielle pour Données Mixtes (FAMD).
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame avec données mixtes
    n_components : int
        Nombre de composantes
    target_col : str
        Nom de la colonne cible
    sample_size : int
        Taille de l'échantillon pour performance
        
    Returns:
    --------
    tuple : (coordonnées, modèle FAMD)
    """
    if not PRINCE_AVAILABLE:
        print("⚠️ Package 'prince' non installé. Utilisation de PCA à la place...")
        return None, None
    
    # Préparation des données
    df_sample = df.sample(min(sample_size, len(df)), random_state=42)
    
    # Séparation des variables
    continuous_vars = df_sample.select_dtypes(include=['float64']).columns.tolist()
    categorical_vars = df_sample.select_dtypes(include=['int64', 'object']).columns.tolist()
    
    if target_col in categorical_vars:
        categorical_vars.remove(target_col)
    
    # FAMD
    famd = prince.FAMD(n_components=n_components, random_state=42)
    
    # Préparation du DataFrame pour FAMD
    X = df_sample[continuous_vars + categorical_vars]
    
    # Fit FAMD
    famd.fit(X)
    
    # Obtenir les coordonnées
    coords = famd.row_coordinates(X)
    
    # Visualisation
    plt.figure(figsize=(10, 6))
    
    # Encoder la cible si nécessaire
    if df_sample[target_col].dtype == 'object':
        colors = df_sample[target_col].map({'ad.': 1, 'noad.': 0})
    else:
        colors = df_sample[target_col]
    
    scatter = plt.scatter(coords[0], coords[1], c=colors, cmap='coolwarm', alpha=0.6)
    plt.xlabel(f'Dimension 1 ({famd.explained_inertia_[0]:.1%})')
    plt.ylabel(f'Dimension 2 ({famd.explained_inertia_[1]:.1%})')
    plt.title('Analyse Factorielle pour Données Mixtes (FAMD)')
    plt.colorbar(scatter, label='Classe')
    plt.tight_layout()
    plt.show()
    
    return coords, famd




def analyze_feature_importance(df: pd.DataFrame, target_col: str = 'outcome',
                             method: str = 'all', top_n: int = 20,
                             figsize: Tuple[float, float] = (12, 8)) -> pd.DataFrame:
    """
    Analyse l'importance des features avec plusieurs méthodes.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame contenant les données
    target_col : str
        Nom de la colonne cible
    method : str
        'rf', 'f_score', 'correlation', ou 'all'
    top_n : int
        Nombre de features à afficher
    figsize : tuple
        Taille de la figure
        
    Returns:
    --------
    pd.DataFrame : Importance des features
    """
    try:
        # Préparation des données
        data = df.copy()
        
        # Conversion de la cible
        if data[target_col].dtype == 'object':
            data[target_col] = data[target_col].map({'ad.': 1, 'noad.': 0})
        
        # Variables numériques
        numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)
        
        # Gestion des valeurs manquantes
        X = data[numeric_cols]
        y = data[target_col]
        
        # Imputation
        imputer = SimpleImputer(strategy='median')
        X_imputed = imputer.fit_transform(X)
        X_imputed = pd.DataFrame(X_imputed, columns=numeric_cols)
        
        # Standardisation
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)
        
        results = pd.DataFrame({'feature': numeric_cols})
        
        # 1. Random Forest
        if method in ['rf', 'all']:
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X_scaled, y)
            results['RF_Importance'] = rf.feature_importances_
            results['RF_Importance'] = results['RF_Importance'] / results['RF_Importance'].max()
        
        # 2. F-Score
        if method in ['f_score', 'all']:
            f_scores, _ = f_classif(X_scaled, y)
            results['F_Score'] = f_scores
            results['F_Score'] = results['F_Score'] / results['F_Score'].max()
        
        # 3. Correlation
        if method in ['correlation', 'all']:
            correlations = X_imputed.corrwith(y).abs()
            results['Correlation'] = correlations.values
            results['Correlation'] = results['Correlation'] / results['Correlation'].max()
        
        # Score combiné
        if method == 'all':
            results['Combined_Score'] = results[['RF_Importance', 'F_Score', 'Correlation']].mean(axis=1)
            sort_col = 'Combined_Score'
        else:
            sort_col = [col for col in results.columns if col != 'feature'][0]
        
        results = results.sort_values(sort_col, ascending=False)
        
        # Visualisation
        top_features = results.head(top_n)
        
        plt.figure(figsize=figsize)
        
        if method == 'all':
            # Barplot groupé
            x = np.arange(len(top_features))
            width = 0.25
            
            plt.bar(x - width, top_features['RF_Importance'], width, label='Random Forest')
            plt.bar(x, top_features['F_Score'], width, label='F-Score')
            plt.bar(x + width, top_features['Correlation'], width, label='Correlation')
            
            plt.xlabel('Features')
            plt.ylabel('Importance normalisée')
            plt.title(f'Top {top_n} Features - Comparaison des méthodes')
            plt.xticks(x, top_features['feature'], rotation=45, ha='right')
            plt.legend()
        else:
            plt.bar(range(len(top_features)), top_features[sort_col])
            plt.xticks(range(len(top_features)), top_features['feature'], rotation=45, ha='right')
            plt.ylabel('Importance')
            plt.title(f'Top {top_n} Features - {method}')
        
        plt.tight_layout()
        plt.show()
        
        return results
        
    except Exception as e:
        print(f"Erreur lors de l'analyse d'importance : {str(e)}")
        return pd.DataFrame()



def dimension_reduction(data, correlation_threshold=0.8, variance_threshold=0.95, display_info=False):
    """
    Réduit la dimensionnalité des données en utilisant la corrélation et l'ACP.
    
    Args:
        data (pd.DataFrame): DataFrame contenant les données
        correlation_threshold (float): Seuil de corrélation pour éliminer les variables redondantes
        variance_threshold (float): Seuil de variance expliquée pour l'ACP
        display_info (bool): Si True, affiche les informations détaillées
        
    Returns:
        tuple: (DataFrame réduit, modèle ACP, liste des variables conservées)
    """
    if display_info:
        print("\n=== Réduction de Dimensionnalité ===")
    
    # 1. Préparation des données
    # Séparer la variable cible
    target = data['outcome'].copy()
    target_numeric = (target == 'ad.').astype(int)
    
    # Sélectionner les variables numériques
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
    numeric_cols = [col for col in numeric_cols if col != 'outcome']
    X = data[numeric_cols].copy()
    
    # Gérer les valeurs manquantes
    if display_info:
        print("\nGestion des valeurs manquantes:")
        missing_counts = X.isnull().sum()
        print(f"Nombre de colonnes avec valeurs manquantes: {(missing_counts > 0).sum()}")
        print(f"Pourcentage moyen de valeurs manquantes: {missing_counts.mean()*100:.2f}%")
    
    # Imputer les valeurs manquantes par la moyenne
    X = X.fillna(X.mean())
    
    # 2. Élimination des variables fortement corrélées
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Identifier les colonnes à supprimer
    to_drop = [column for column in upper.columns if any(upper[column] > correlation_threshold)]
    
    # Garder les variables les plus corrélées avec la cible
    correlations_with_target = pd.Series({
        col: X[col].corr(target_numeric) for col in to_drop
    }).abs()
    
    # Trier les variables par corrélation avec la cible
    to_drop = [col for col in correlations_with_target.sort_values(ascending=True).index]
    
    if display_info:
        print(f"\nNombre de variables éliminées par corrélation : {len(to_drop)}")
        print(f"Variables restantes : {len(X.columns) - len(to_drop)}")
    
    # Créer le DataFrame réduit
    X_reduced = X.drop(columns=to_drop)
    
    # 3. Application de l'ACP
    # Standardisation des données
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_reduced)
    
    # Appliquer l'ACP
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    
    # Calculer la variance expliquée cumulée
    cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(cumulative_variance_ratio >= variance_threshold) + 1
    
    if display_info:
        print(f"\nNombre de composantes principales retenues : {n_components}")
        print(f"Variance expliquée : {cumulative_variance_ratio[n_components-1]*100:.2f}%")
    
    # Créer le DataFrame final avec les composantes principales
    pca_reduced = PCA(n_components=n_components)
    X_pca_reduced = pca_reduced.fit_transform(X_scaled)
    
    # Créer les noms des colonnes pour les composantes principales
    pca_cols = [f"PC{i+1}" for i in range(n_components)]
    X_final = pd.DataFrame(X_pca_reduced, columns=pca_cols, index=X.index)
    
    # Ajouter la variable cible
    X_final['outcome'] = target
    
    # 4. Afficher les composantes principales et leur importance
    if display_info:
        print("\nImportance des composantes principales :")
        for i in range(n_components):
            print(f"PC{i+1}: {pca_reduced.explained_variance_ratio_[i]*100:.2f}%")
    
    # 5. Visualisation des deux premières composantes principales
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_pca_reduced[:, 0], X_pca_reduced[:, 1], 
                         c=target_numeric, cmap='viridis', alpha=0.6)
    plt.xlabel(f"Première composante principale ({pca_reduced.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"Deuxième composante principale ({pca_reduced.explained_variance_ratio_[1]*100:.1f}%)")
    plt.title("Projection des données sur les deux premières composantes principales")
    plt.colorbar(scatter, label='Classe (0: non-pub, 1: pub)')
    plt.tight_layout()
    plt.show()
    
    return X_final, pca_reduced, X_reduced.columns.tolist()


def umap_visualization(df, target_col='outcome'):
    """
    Visualise les données avec UMAP en optimisant les paramètres pour une meilleure séparation des classes.
    """
    try:
        # Copie du DataFrame pour éviter de modifier l'original
        df_clean = df.copy()
        
        # Encodage de la variable cible
        df_clean[target_col] = df_clean[target_col].map({'ad.': 1, 'noad.': 0})
        
        # Sélection des colonnes numériques
        numeric_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns
        X = df_clean[numeric_cols]
        
        # Analyse des valeurs manquantes
        missing_values = X.isnull().sum()
        print("\n=== Analyse des valeurs manquantes ===")
        print("Nombre de valeurs manquantes par colonne :")
        print(missing_values[missing_values > 0])
        
        # Imputation adaptative
        for col in X.columns:
            missing_pct = missing_values[col] / len(X) * 100
            if missing_pct > 30:
                # Suppression des variables avec plus de 30% de valeurs manquantes
                X = X.drop(columns=[col])
            elif missing_pct > 5:
                # Imputation par k-NN pour les variables avec 5-30% de valeurs manquantes
                imputer = KNNImputer(n_neighbors=5)
                X[col] = imputer.fit_transform(X[[col]])
            else:
                # Imputation par médiane pour les variables avec moins de 5% de valeurs manquantes
                imputer = SimpleImputer(strategy='median')
                X[col] = imputer.fit_transform(X[[col]])
        
        # Standardisation
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Optimisation des paramètres UMAP
        n_neighbors = min(50, len(X_scaled) // 10)  # Ajustement dynamique du nombre de voisins
        min_dist = 0.1  # Distance minimale entre les points
        
        # Application de UMAP
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric='euclidean',
            random_state=42
        )
        embedding = reducer.fit_transform(X_scaled)
        
        # Création du graphique
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=df_clean[target_col],
            cmap=plt.cm.viridis,
            alpha=0.6,
            s=20
        )
        
        # Ajout des centroids des classes
        for label in [0, 1]:
            mask = df_clean[target_col] == label
            centroid = np.mean(embedding[mask], axis=0)
            plt.scatter(
                centroid[0],
                centroid[1],
                c='red' if label == 1 else 'blue',
                marker='*',
                s=200,
                edgecolor='black'
            )
        
        # Calcul des métriques de qualité
        silhouette = silhouette_score(embedding, df_clean[target_col])
        calinski = calinski_harabasz_score(embedding, df_clean[target_col])
        
        plt.title(f'Visualisation UMAP (Silhouette: {silhouette:.3f}, Calinski-Harabasz: {calinski:.3f})')
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.colorbar(scatter, label='Classe (0: noad., 1: ad.)')
        
        # Ajout d'une légende pour les centroids
        plt.legend(
            handles=[
                plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='blue', markersize=10, label='Centroid noad.'),
                plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='red', markersize=10, label='Centroid ad.')
            ],
            loc='upper right'
        )
        
        plt.tight_layout()
        plt.show()
        
        # Analyse de la séparation des classes
        print("\n=== Analyse de la séparation des classes ===")
        print(f"Score de silhouette : {silhouette:.3f}")
        print(f"Score de Calinski-Harabasz : {calinski:.3f}")
        
        if silhouette > 0.5:
            print("Bonne séparation des classes")
        elif silhouette > 0.25:
            print("Séparation modérée des classes")
        else:
            print("Faible séparation des classes")
            
    except Exception as e:
        print(f"Erreur lors de la visualisation UMAP : {str(e)}")


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


def advanced_dimension_reduction(data, n_features_to_select=50):
    """
    Applique des techniques avancées de réduction de dimensionnalité et de sélection de variables.
    
    Args:
        data (pd.DataFrame): DataFrame contenant les données
        n_features_to_select (int): Nombre de variables à sélectionner avec Random Forest
        
    Returns:
        tuple: (DataFrame avec variables sélectionnées, modèles de réduction, visualisations)
    """
    print("\n=== Réduction de Dimensionnalité Avancée ===")
    
    # 1. Préparation des données
    target = data['outcome'].copy()
    target_numeric = (target == 'ad.').astype(int)
    
    # Sélectionner les variables numériques
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
    numeric_cols = [col for col in numeric_cols if col != 'outcome']
    X = data[numeric_cols].copy()
    
    # Imputer les valeurs manquantes
    X = X.fillna(X.mean())
    
    # Standardisation des données
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    # 2. Sélection de variables avec Random Forest
    print("\nSélection de variables avec Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_scaled, target_numeric)
    
    # Sélectionner les variables les plus importantes
    selector = SelectFromModel(rf, max_features=n_features_to_select, prefit=True)
    feature_mask = selector.get_support()
    selected_features = X.columns[feature_mask].tolist()
    
    print(f"Nombre de variables sélectionnées : {len(selected_features)}")
    
    # Afficher les 10 variables les plus importantes
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 variables les plus importantes :")
    print(feature_importance.head(10))
    
    # 3. Réduction avec UMAP
    print("\nApplication de UMAP...")
    reducer_umap = umap.UMAP(random_state=42)
    X_umap = reducer_umap.fit_transform(X_scaled)
    
    # 4. Réduction avec t-SNE
    print("\nApplication de t-SNE...")
    reducer_tsne = TSNE(n_components=2, random_state=42)
    X_tsne = reducer_tsne.fit_transform(X_scaled)
    
    # 5. Visualisations
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # UMAP
    scatter1 = ax1.scatter(X_umap[:, 0], X_umap[:, 1], 
                          c=target_numeric, cmap='viridis', alpha=0.6)
    ax1.set_title("Projection UMAP")
    ax1.set_xlabel("UMAP1")
    ax1.set_ylabel("UMAP2")
    plt.colorbar(scatter1, ax=ax1, label='Classe (0: non-pub, 1: pub)')
    
    # t-SNE
    scatter2 = ax2.scatter(X_tsne[:, 0], X_tsne[:, 1], 
                          c=target_numeric, cmap='viridis', alpha=0.6)
    ax2.set_title("Projection t-SNE")
    ax2.set_xlabel("t-SNE1")
    ax2.set_ylabel("t-SNE2")
    plt.colorbar(scatter2, ax=ax2, label='Classe (0: non-pub, 1: pub)')
    
    plt.tight_layout()
    plt.show()
    
    # 6. Créer le DataFrame final avec les variables sélectionnées
    X_selected = X_scaled_df[selected_features].copy()
    X_selected['outcome'] = target
    
    # 7. Analyse des clusters naturels avec UMAP
    print("\nAnalyse des clusters naturels...")
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X_umap)
    
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_umap[:, 0], X_umap[:, 1], 
                         c=clusters, cmap='viridis', alpha=0.6)
    plt.title("Clusters naturels identifiés par UMAP")
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.colorbar(scatter, label='Cluster')
    plt.show()
    
    # 8. Distribution des classes dans chaque cluster
    cluster_dist = pd.DataFrame({
        'cluster': clusters,
        'class': target_numeric
    }).groupby('cluster')['class'].value_counts(normalize=True).unstack()
    
    print("\nDistribution des classes par cluster :")
    print(cluster_dist)
    
    return X_selected, {
        'umap': reducer_umap,
        'tsne': reducer_tsne,
        'rf': rf,
        'selected_features': selected_features
    }


def perform_exploratory_analysis(data, target_col='outcome'):
    """
    Effectue une analyse exploratoire complète des données.
    
    Args:
        data (pd.DataFrame): DataFrame contenant les données à analyser
        target_col (str, optional): Nom de la colonne cible. Defaults to 'outcome'.
    """
    print("=== Début de l'analyse exploratoire ===\n")
    
    # Analyses existantes
    univariate_analysis(data)
    correlations, high_corr_pairs = bivariate_analysis(data)
    multivariate_analysis(data)
    
    # Réduction de dimension et visualisations
    X_final, pca_model, selected_features = dimension_reduction(data)
    umap_visualization(data, target_col)  # Ajout de la visualisation UMAP
    
    # Résumé des statistiques
    summary_statistics(data, pca_model, target_col)
    
    print("\n=== Fin de l'analyse exploratoire ===")
    return X_final, selected_features


def create_polynomial_features(data, variables=None, degree=2):
    """
    Crée des features polynomiales.
    
    Args:
        data (pd.DataFrame): DataFrame contenant les données
        variables (List[str]): Liste des variables à transformer
        degree (int): Degré maximum du polynôme
        
    Returns:
        pd.DataFrame: DataFrame avec les nouvelles features
    """
    if variables is None:
        # Si pas de variables spécifiées, utiliser les colonnes numériques
        numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
        numeric_cols = numeric_cols[numeric_cols != 'outcome']
        # Prendre les 4 premières colonnes numériques
        variables = numeric_cols[:4].tolist()
    
    new_features = {}
    for var in variables:
        if var in data.columns:  # Vérifier que la variable existe dans le DataFrame
            for d in range(2, degree + 1):
                new_features[f"{var}_pow{d}"] = data[var] ** d
    
    return pd.DataFrame(new_features)


def create_interaction_features(data, variables=None):
    """
    Crée des features d'interaction.
    
    Args:
        data (pd.DataFrame): DataFrame contenant les données
        variables (List[str]): Liste des variables pour créer les interactions
        
    Returns:
        pd.DataFrame: DataFrame avec les nouvelles features
    """
    if variables is None:
        # Si pas de variables spécifiées, utiliser les colonnes numériques
        numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
        numeric_cols = numeric_cols[numeric_cols != 'outcome']
        # Prendre les 4 premières colonnes numériques
        variables = numeric_cols[:4].tolist()
    
    new_features = {}
    for i, var1 in enumerate(variables):
        if var1 not in data.columns:  # Skip if variable doesn't exist
            continue
        for var2 in variables[i+1:]:
            if var2 not in data.columns:  # Skip if variable doesn't exist
                continue
            new_features[f"{var1}_{var2}_interact"] = data[var1] * data[var2]
    
    return pd.DataFrame(new_features)


def enhance_features(data, variables=None):
    """
    Crée toutes les nouvelles features et les combine.
    
    Args:
        data (pd.DataFrame): DataFrame contenant les données
        variables (List[str]): Liste des variables à utiliser
        
    Returns:
        pd.DataFrame: DataFrame avec toutes les features
    """
    try:
        poly_features = create_polynomial_features(data, variables)
        interact_features = create_interaction_features(data, variables)
        
        # Combine all features
        enhanced_df = pd.concat([data, poly_features, interact_features], axis=1)
        return enhanced_df
        
    except Exception as e:
        print(f"Erreur lors de la création des features augmentées : {str(e)}")
        return data  # Return original data if enhancement fails


def optimize_hyperparameters(X, y):
    """
    Optimise les hyperparamètres pour plusieurs modèles de classification.
    
    Parameters:
    -----------
    X : array-like
        Features matrix
    y : array-like
        Target variable
    
    Returns:
    --------
    dict
        Dictionary containing optimized models and their parameters
    """
    # Preprocessing
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Handle imbalanced data
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
    
    # Define scoring metrics
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score),
        'recall': make_scorer(recall_score),
        'f1': make_scorer(f1_score)
    }
    
    # Define parameter grids
    param_grids = {
        'RandomForest': {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'DecisionTree': {
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'criterion': ['gini', 'entropy']
        },
        'LogisticRegression': {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear']
        }
    }
    
    # Initialize models
    models = {
        'RandomForest': RandomForestClassifier(random_state=42),
        'DecisionTree': DecisionTreeClassifier(random_state=42),
        'LogisticRegression': LogisticRegression(random_state=42)
    }
    
    # Perform grid search for each model
    results = {}
    for name, model in models.items():
        print(f"\nOptimizing {name}...")
        grid_search = GridSearchCV(
            model,
            param_grids[name],
            cv=5,
            scoring=scoring,
            refit='f1',
            n_jobs=-1
        )
        grid_search.fit(X_resampled, y_resampled)
        
        results[name] = {
            'best_model': grid_search.best_estimator_,
            'best_params': grid_search.best_params_,
            'cv_results': pd.DataFrame(grid_search.cv_results_)
        }
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best F1 score: {grid_search.best_score_:.4f}")
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    model_names = list(results.keys())
    f1_scores = [result['cv_results']['mean_test_f1'].max() for result in results.values()]
    
    plt.bar(model_names, f1_scores)
    plt.title('Model Comparison - F1 Scores')
    plt.ylabel('F1 Score')
    plt.ylim(0, 1)
    
    for i, score in enumerate(f1_scores):
        plt.text(i, score, f'{score:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    return results


def evaluate_optimized_models(models_dict, X, y):
    """
    Evaluate optimized models on test data.
    
    Parameters:
    -----------
    models_dict : dict
        Dictionary containing optimized models
    X : array-like
        Test features
    y : array-like
        Test target
    
    Returns:
    --------
    dict
        Dictionary containing evaluation metrics
    """
    # Preprocessing
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Evaluate each model
    results = {}
    for name, model_info in models_dict.items():
        model = model_info['best_model']
        y_pred = model.predict(X_scaled)
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1': f1_score(y, y_pred)
        }
        
        results[name] = metrics
        
        print(f"\n{name.upper()} Model Evaluation:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
    
    # Plot comparison
    plt.figure(figsize=(15, 5))
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    x = np.arange(len(metrics))
    width = 0.2
    
    for i, (name, model_results) in enumerate(results.items()):
        plt.bar(x + i*width, [model_results[m] for m in metrics], width, label=name)
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Model Comparison on Test Set')
    plt.xticks(x + width, metrics)
    plt.legend()
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()
    
    return results


def analyze_categorical_binaries_vs_target(data, target_col='outcome', show_top=20, pval_threshold=0.05):
    """
    Analyse les relations entre les variables binaires (0/1) et une variable cible catégorielle.

    Affiche les p-values du test du chi^2 pour chaque variable binaire vs. la cible,
    et sélectionne celles en-dessous du seuil pval_threshold.

    Args:
        data (pd.DataFrame): le DataFrame à analyser.
        target_col (str): nom de la variable cible.
        show_top (int): nombre de variables les plus significatives à afficher.
        pval_threshold (float): seuil de significativité des p-values.

    Returns:
        pd.DataFrame: tableau des variables sélectionnées triées par p-value croissante.
    """
    print("\n=== Analyse Bivariée : Variables Binaires vs Cible Catégorielle ===")

    binary_cols = [col for col in data.columns 
                   if col != target_col and data[col].dropna().nunique() == 2]

    results = []

    for col in binary_cols:
        contingency = pd.crosstab(data[col], data[target_col])
        if contingency.shape[0] == 2 and contingency.shape[1] > 1:
            chi2, p, dof, expected = chi2_contingency(contingency)
            if p < pval_threshold:
                results.append({'variable': col, 'p_value': p, 'chi2': chi2})

    results_df = pd.DataFrame(results).sort_values("p_value")

    print(f"\nTop {show_top} variables binaires avec p-value < {pval_threshold} :")
    print(results_df.head(show_top))

    return results_df