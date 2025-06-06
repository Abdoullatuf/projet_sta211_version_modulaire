# Module de visualisation (visualization.py)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Optional, Tuple, Union


from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler



def save_fig(
    fname: str,
    directory: Union[str, Path],
    dpi: int = 100,
    figsize: Optional[Tuple[float, float]] = None,
    format: str = "png",
    quality: Optional[int] = None,
    close: bool = False,
    show: bool = True,
    **kwargs
) -> Path:
    """
    Sauvegarde la figure matplotlib courante avec optimisation.

    Args:
        fname: Nom du fichier de sortie (sans chemin complet).
        directory: Répertoire de sauvegarde, généralement paths["FIGURES_DIR"].
        dpi: Résolution de l'image (défaut à 100 pour réduire la taille).
        figsize: (largeur, hauteur) en pouces. Si None, conserve la taille actuelle.
        format: Format de fichier ("png", "jpeg", "svg", "pdf").
        quality: Qualité pour JPEG (1-100, facultatif).
        close: Si True, ferme la figure après sauvegarde.
        show: Si True, affiche la figure.
        **kwargs: Arguments supplémentaires pour plt.savefig.

    Returns:
        Path: Chemin complet du fichier sauvegardé.

    Raises:
        ValueError: Si aucune figure n'est active ou si le format est invalide.
        IOError: Si la sauvegarde ou l'accès au répertoire échoue.
    """
    # Vérifier si une figure est active
    if plt.get_fignums() == 0:
        raise ValueError("Aucune figure Matplotlib active à sauvegarder.")

    # Valider le format
    supported_formats = ["png", "jpeg", "svg", "pdf"]
    if format not in supported_formats:
        raise ValueError(f"Format {format} non supporté. Formats valides : {supported_formats}")

    # Définir le répertoire
    directory = Path(directory)

    # Vérifier l'accessibilité du répertoire
    try:
        if not directory.exists():
            raise IOError(f"Le répertoire {directory} n'existe pas. Assurez-vous que setup_project_paths() est appelé.")
        if not os.access(directory, os.W_OK):
            raise IOError(f"Le répertoire {directory} n'est pas accessible en écriture.")
    except Exception as e:
        raise IOError(f"Erreur d'accès au répertoire {directory}: {str(e)}")

    # Construire le chemin complet
    path = directory / fname
    if not path.suffix:
        path = path.with_suffix(f".{format}")

    # Ajuster la taille si nécessaire
    if figsize:
        current_figsize = plt.gcf().get_size_inches()
        if not np.allclose(current_figsize, figsize):
            plt.gcf().set_size_inches(figsize)

    # Préparer les arguments pour savefig
    savefig_kwargs = {"dpi": dpi, "bbox_inches": "tight", "format": format}
    if format == "jpeg" and quality is not None:
        savefig_kwargs["quality"] = quality
    savefig_kwargs.update(kwargs)

    # Sauvegarder la figure
    try:
        plt.savefig(path, **savefig_kwargs)
    except Exception as e:
        raise IOError(f"Erreur lors de la sauvegarde de la figure à {path}: {str(e)}")

    # Afficher si demandé
    if show:
        plt.show()

    # Fermer la figure si demandé
    if close:
        plt.close(plt.gcf())

    print(f"✅ Figure sauvegardée : {path}")
    return path




def visualize_distributions_and_boxplots(df: pd.DataFrame, continuous_cols: List[str], output_dir: Path) -> None:
    """
    Affiche et sauvegarde les histogrammes + KDE et boxplots pour chaque variable continue.
    """
    num_cols = len(continuous_cols)
    fig, axes = plt.subplots(2, num_cols, figsize=(6 * num_cols, 10))
    axes = axes.flatten()

    for i, col in enumerate(continuous_cols):
        data = df[col].dropna()

        # Histogramme + KDE
        ax1 = axes[i]
        data.hist(bins=50, ax=ax1, alpha=0.7, color='skyblue', edgecolor='black')
        ax2 = ax1.twinx()
        data.plot(kind='kde', ax=ax2, color='red', linewidth=2)
        ax1.set_title(f'Distribution de {col}')
        ax1.set_xlabel(col)
        ax1.set_ylabel('Fréquence')
        ax2.set_ylabel('Densité')

        # Boxplot
        ax3 = axes[i + num_cols]
        sns.boxplot(y=df[col], ax=ax3, color='lightgrey')
        ax3.set_title(f'Box plot de {col}')
        ax3.set_ylabel('Valeur')

    plt.tight_layout()
    
    # Sauvegarde avec votre fonction centralisée
    fig_path = save_fig(
        fname="continuous_distributions_boxplots.png",
        directory=output_dir,
        dpi=300,
        format="png",
        show=True,
        close=True
    )

    print(f"✅ Visualisations sauvegardées dans : {fig_path}")




def plot_correlation_heatmap(
    df: pd.DataFrame,
    variables: Optional[List[str]] = None,
    binary_only: bool = False,
    top_n: int = 100,
    target_col: Optional[str] = None,
    method: str = "pearson",
    figsize: Tuple[float, float] = (12, 10),
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True
) -> pd.DataFrame:
    """
    Affiche une heatmap de corrélation avec options avancées.

    Args:
        df: DataFrame contenant les données.
        variables: Liste des variables à inclure. Si None, utilise toutes les variables numériques.
        binary_only: Si True, ne sélectionne que les variables binaires.
        top_n: Nombre maximum de variables à inclure (pour variables binaires).
        target_col: Colonne cible à exclure (facultatif).
        method: Méthode de corrélation ("pearson", "spearman", "kendall").
        figsize: Taille de la figure.
        save_path: Chemin de sauvegarde (facultatif).
        show: Si True, affiche la figure.

    Returns:
        Matrice de corrélation.
    """
    paths = setup_project_paths()
    fig_dir = paths.get("FIGURES_DIR", Path("."))

    # Sélection des variables
    if variables is None:
        if binary_only:
            variables = [
                col for col in df.select_dtypes(include="int64").columns
                if (target_col is None or col != target_col) and
                   set(df[col].dropna().unique()).issubset({0, 1})
            ]
            variables = variables[:min(len(variables), top_n)]
        else:
            variables = df.select_dtypes(include=[np.number]).columns.tolist()
            if target_col and target_col in variables:
                variables.remove(target_col)

    if not variables:
        print("❌ Aucune variable sélectionnée pour la corrélation.")
        return pd.DataFrame()

    # Imputation
    df = impute_missing_values(df, variables, strategy="zero" if binary_only else "median")

    # Matrice de corrélation
    corr_matrix = df[variables].corr(method=method)

    # Visualisation
    plt.figure(figsize=figsize)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix, mask=mask, cmap="coolwarm", center=0,
        square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
        annot=len(variables) <= 20, fmt=".2f"
    )
    title = f"Matrice de corrélation ({method.capitalize()})"
    if binary_only:
        title += f" - Top {min(top_n, len(variables))} variables binaires"
    plt.title(title, fontsize=14)
    plt.tight_layout()

    # Sauvegarde avec ta fonction
    if save_path:
        fname = Path(save_path).name
        save_fig(fname, directory=fig_dir, figsize=figsize, show=show, close=True)
    elif show:
        plt.show()

    return corr_matrix



def plot_missing_patterns(
    df: pd.DataFrame,
    figsize: Tuple[float, float] = (12, 6),
    save: bool = True,
    fname: str = "missing_patterns.png",
    directory: Optional[Union[str, Path]] = None,
    show: bool = True
) -> pd.DataFrame:
    """
    Visualise les patterns de valeurs manquantes et les sauvegarde si souhaité.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame à analyser
    figsize : tuple
        Taille de la figure
    save : bool
        Si True, enregistre la figure
    fname : str
        Nom du fichier à sauvegarder
    directory : str or Path, optional
        Répertoire de sauvegarde (par défaut FIGURES_DIR du projet)
    show : bool
        Si True, affiche la figure après création

    Returns:
    --------
    pd.DataFrame : Statistiques des valeurs manquantes
    """
    # Calculer les statistiques
    missing_stats = pd.DataFrame({
        'count': df.isnull().sum(),
        'percentage': df.isnull().sum() / len(df) * 100
    })
    missing_stats = missing_stats[missing_stats['count'] > 0].sort_values('percentage', ascending=False)

    if missing_stats.empty:
        print("✅ Aucune valeur manquante détectée !")
        return missing_stats

    # Visualisation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Bar plot des pourcentages
    missing_stats['percentage'].plot(kind='bar', ax=ax1, color='coral')
    ax1.set_title('Pourcentage de valeurs manquantes par variable')
    ax1.set_ylabel('Pourcentage (%)')
    ax1.set_xlabel('Variables')

    # Heatmap des patterns
    cols_with_missing = missing_stats.index.tolist()
    missing_matrix = df[cols_with_missing].isnull().astype(int)

    sample_size = min(100, len(df))
    sample_indices = np.random.choice(df.index, sample_size, replace=False)

    sns.heatmap(
        missing_matrix.loc[sample_indices].T,
        cmap='RdYlBu', cbar_kws={'label': 'Manquant (1) / Présent (0)'},
        ax=ax2
    )
    ax2.set_title(f'Pattern des valeurs manquantes ({sample_size} échantillons)')
    ax2.set_xlabel('Échantillons')
    ax2.set_ylabel('Variables')

    plt.tight_layout()

    # 🔄 Sauvegarde conditionnelle
    if save:
        from config.paths_config import setup_project_paths  # si ce n'est pas déjà dans le fichier
        paths = setup_project_paths()
        if directory is None:
            directory = paths["FIGURES_DIR"]
        save_fig(fname=fname, directory=directory, figsize=figsize, show=show, close=not show)
    else:
        if show:
            plt.show()

    return missing_stats




def compare_visualization_methods(
    df: pd.DataFrame,
    target_col: str = 'outcome',
    sample_size: int = 1000,
    figsize: Tuple[float, float] = (18, 6),
    save: bool = True,
    fname: str = "dimensionality_comparison.png",
    directory: Optional[Union[str, Path]] = None,
    show: bool = True
):
    """
    Compare différentes méthodes de visualisation (PCA, t-SNE, UMAP) avec option de sauvegarde.

    Args:
        df: DataFrame contenant les données
        target_col: Colonne cible
        sample_size: Taille de l'échantillon utilisé
        figsize: Taille de la figure
        save: Si True, sauvegarde la figure avec save_fig
        fname: Nom du fichier pour sauvegarde
        directory: Répertoire de sauvegarde (par défaut FIGURES_DIR)
        show: Si True, affiche la figure après création
    """
    try:
        df_clean = df.copy()

        if df_clean[target_col].dtype == 'object':
            df_clean[target_col] = df_clean[target_col].map({'ad.': 1, 'noad.': 0})

        numeric_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)

        if len(df_clean) > sample_size:
            df_clean = df_clean.sample(sample_size, random_state=42)

        X = df_clean[numeric_cols]
        y = df_clean[target_col]

        imputer = SimpleImputer(strategy='median')
        X_imputed = imputer.fit_transform(X)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)

        n_methods = 3 if UMAP_AVAILABLE else 2
        fig, axes = plt.subplots(1, n_methods, figsize=figsize)

        # PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        silhouette_pca = silhouette_score(X_pca, y)

        ax = axes[0] if n_methods > 1 else axes
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', alpha=0.6)
        ax.set_title(f'PCA (Silhouette: {silhouette_pca:.3f})')
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        plt.colorbar(scatter, ax=ax)

        # t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        X_tsne = tsne.fit_transform(X_scaled[:min(500, len(X_scaled))])
        y_tsne = y[:min(500, len(y))]
        silhouette_tsne = silhouette_score(X_tsne, y_tsne)

        ax = axes[1] if n_methods > 2 else axes[1] if n_methods > 1 else axes
        scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_tsne, cmap='coolwarm', alpha=0.6)
        ax.set_title(f't-SNE (Silhouette: {silhouette_tsne:.3f})')
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        plt.colorbar(scatter, ax=ax)

        # UMAP
        if UMAP_AVAILABLE and n_methods > 2:
            reducer = umap.UMAP(n_neighbors=min(50, len(X_scaled) // 10), min_dist=0.1, random_state=42)
            X_umap = reducer.fit_transform(X_scaled)
            silhouette_umap = silhouette_score(X_umap, y)

            scatter = axes[2].scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='coolwarm', alpha=0.6)
            axes[2].set_title(f'UMAP (Silhouette: {silhouette_umap:.3f})')
            axes[2].set_xlabel('UMAP 1')
            axes[2].set_ylabel('UMAP 2')
            plt.colorbar(scatter, ax=axes[2])

        plt.tight_layout()

        if save:
            from config.paths_config import setup_project_paths
            paths = setup_project_paths()
            if directory is None:
                directory = paths["FIGURES_DIR"]
            save_fig(fname=fname, directory=directory, figsize=figsize, show=show, close=not show)
        elif show:
            plt.show()

        # Résumé
        print("\n=== Comparaison des méthodes ===")
        print(f"PCA   - Silhouette: {silhouette_pca:.3f}")
        print(f"t-SNE - Silhouette: {silhouette_tsne:.3f}")
        if UMAP_AVAILABLE:
            print(f"UMAP  - Silhouette: {silhouette_umap:.3f}")

    except Exception as e:
        print(f"❌ Erreur lors de la comparaison : {str(e)}")
        import traceback
        traceback.print_exc()


def umap_visualization(
    df: pd.DataFrame,
    target_col: str = 'outcome',
    figsize: Tuple[float, float] = (12, 8),
    save: bool = True,
    fname: str = "umap_visualization.png",
    directory: Optional[Union[str, Path]] = None,
    show: bool = True
):
    """
    Visualise les données avec UMAP en optimisant les paramètres pour une meilleure séparation des classes,
    avec options de sauvegarde et d'affichage.

    Args:
        df: DataFrame contenant les données
        target_col: Colonne cible à mapper (valeurs 'ad.', 'noad.')
        figsize: Taille de la figure
        save: Si True, sauvegarde la figure
        fname: Nom du fichier pour sauvegarde
        directory: Répertoire de sauvegarde (par défaut FIGURES_DIR)
        show: Si True, affiche la figure
    """
    try:
        df_clean = df.copy()
        df_clean[target_col] = df_clean[target_col].map({'ad.': 1, 'noad.': 0})

        numeric_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns
        X = df_clean[numeric_cols]

        missing_values = X.isnull().sum()
        print("\n=== Analyse des valeurs manquantes ===")
        print("Nombre de valeurs manquantes par colonne :")
        print(missing_values[missing_values > 0])

        for col in X.columns:
            missing_pct = missing_values[col] / len(X) * 100
            if missing_pct > 30:
                X = X.drop(columns=[col])
            elif missing_pct > 5:
                imputer = KNNImputer(n_neighbors=5)
                X[col] = imputer.fit_transform(X[[col]])
            else:
                imputer = SimpleImputer(strategy='median')
                X[col] = imputer.fit_transform(X[[col]])

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        reducer = umap.UMAP(
            n_neighbors=min(50, len(X_scaled) // 10),
            min_dist=0.1,
            metric='euclidean',
            random_state=42
        )
        embedding = reducer.fit_transform(X_scaled)

        plt.figure(figsize=figsize)
        scatter = plt.scatter(
            embedding[:, 0], embedding[:, 1],
            c=df_clean[target_col],
            cmap=plt.cm.viridis,
            alpha=0.6, s=20
        )

        for label in [0, 1]:
            mask = df_clean[target_col] == label
            centroid = np.mean(embedding[mask], axis=0)
            plt.scatter(
                centroid[0], centroid[1],
                c='red' if label == 1 else 'blue',
                marker='*', s=200, edgecolor='black'
            )

        silhouette = silhouette_score(embedding, df_clean[target_col])
        calinski = calinski_harabasz_score(embedding, df_clean[target_col])

        plt.title(f'Visualisation UMAP (Silhouette: {silhouette:.3f}, Calinski-Harabasz: {calinski:.3f})')
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.colorbar(scatter, label='Classe (0: noad., 1: ad.)')

        plt.legend(
            handles=[
                plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='blue', markersize=10, label='Centroid noad.'),
                plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='red', markersize=10, label='Centroid ad.')
            ],
            loc='upper right'
        )

        plt.tight_layout()

        if save:
            from config.paths_config import setup_project_paths
            paths = setup_project_paths()
            if directory is None:
                directory = paths["FIGURES_DIR"]
            save_fig(fname=fname, directory=directory, figsize=figsize, show=show, close=not show)
        elif show:
            plt.show()

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
        print(f"❌ Erreur lors de la visualisation UMAP : {str(e)}")
        import traceback
        traceback.print_exc()


def plot_continuous_by_class(df, continuous_cols, output_dir, figsize=(15, 5)):
    fig, axes = plt.subplots(1, len(continuous_cols), figsize=figsize)
    for i, col in enumerate(continuous_cols):
        df_clean = df[[col, 'y']].dropna()
        df_clean['y_label'] = df_clean['y'].map({0: 'Non-pub', 1: 'Pub'})
        sns.violinplot(data=df_clean, x='y_label', y=col, ax=axes[i])
        axes[i].set_title(f'{col} par classe')
    plt.tight_layout()
    save_fig('continuous_by_class.png', directory=output_dir, figsize=figsize)



def plot_binary_sparsity(df, binary_cols, output_dir, sample_size=100):
    sample_vars = np.random.choice(binary_cols, size=min(sample_size, len(binary_cols)), replace=False)
    sample_data = df[sample_vars].head(sample_size)
    plt.figure(figsize=(10, 6))
    plt.imshow(sample_data.values, cmap='binary', aspect='auto')
    plt.colorbar(label='Valeur (0 ou 1)')
    plt.title('Sparsité (100 obs × 100 var binaires)')
    plt.xlabel('Variables')
    plt.ylabel('Observations')
    plt.tight_layout()
    save_fig('sparsity_visualization.png', directory=output_dir, figsize=(10, 6))


def plot_continuous_target_corr(df, continuous_cols, output_dir):
    fig, ax = plt.subplots(figsize=(8, 6))
    corr_data = df[continuous_cols + ['y']].corr()['y'][:-1].to_frame()
    sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0,
                cbar_kws={'label': 'Corrélation'}, fmt='.3f', ax=ax)
    ax.set_title('Corrélations avec la cible')
    plt.tight_layout()
    save_fig('continuous_target_correlation.png', directory=output_dir, figsize=(8, 6))



def plot_eda_summary(df, continuous_cols, binary_cols, target_corr, sparsity, imbalance_ratio, output_dir, presence_series):
    fig = plt.figure(figsize=(14, 8))

    # Distribution de la cible
    ax1 = plt.subplot(2, 3, 1)
    df['y'].map({0: 'noad.', 1: 'ad.'}).value_counts().plot.pie(
        ax=ax1, autopct='%1.1f%%', colors=['#3498db', '#e74c3c'])
    ax1.set_title('Cible')
    ax1.set_ylabel('')

    # Valeurs manquantes (hardcodé)
    ax2 = plt.subplot(2, 3, 2)
    pd.Series({'X1': 27.41, 'X2': 27.37, 'X3': 27.61}).plot.bar(
        ax=ax2, color='coral')
    ax2.set_title('Valeurs manquantes')
    ax2.set_ylabel('%')

    # Sparsité
    ax3 = plt.subplot(2, 3, 3)
    pd.Series({'Zéros': sparsity, 'Uns': 100-sparsity}).plot.pie(
        ax=ax3, autopct='%1.1f%%', colors=['lightgray', 'darkgray'])
    ax3.set_title('Sparsité')

    # Corrélations ↔ cible
    ax4 = plt.subplot(2, 3, 4)
    target_corr.abs().nlargest(10).plot.barh(ax=ax4, color='skyblue')
    ax4.set_title('Top 10 corrélations')

    # Taux de présence
    ax5 = plt.subplot(2, 3, 5)
    presence_series.hist(ax=ax5, bins=30, color='lightgreen', edgecolor='black')
    ax5.axvline(presence_series.mean(), color='red', linestyle='--')
    ax5.set_title('Taux de présence')
    ax5.legend([f'Moy: {presence_series.mean():.1f}%'])

    # Statistiques clés
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    summary = f"""
    Observations: {len(df):,}
    Variables: {df.shape[1]:,}
    Déséquilibre: {imbalance_ratio:.1f}:1
    Sparsité: {sparsity:.1f}%
    Corr. max (y): {abs(target_corr).max():.3f}
    Binaires: {len(binary_cols)} | Continues: {len(continuous_cols)}
    """
    ax6.text(0, 0.5, summary, fontsize=12)

    plt.suptitle("Résumé visuel de l'EDA")
    plt.tight_layout()
    save_fig('eda_summary.png', directory=output_dir, figsize=(14, 8))




def plot_outlier_comparison(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    cols: List[str],
    output_dir: Union[str, Path],
    show: bool = True,
    save: bool = True,
    dpi: int = 100
):
    """
    Affiche les boxplots avant/après suppression des outliers pour chaque variable.

    Args:
        df_before (pd.DataFrame): Données avant suppression.
        df_after (pd.DataFrame): Données après suppression.
        cols (list): Colonnes à analyser.
        output_dir (str or Path): Dossier de sauvegarde des figures.
        show (bool): Affiche les figures si True.
        save (bool): Sauvegarde les figures si True.
        dpi (int): Résolution de l’image sauvegardée.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for col in cols:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

        sns.boxplot(x=df_before[col], ax=axes[0], color='salmon')
        axes[0].set_title(f"{col} - Avant suppression")

        sns.boxplot(x=df_after[col], ax=axes[1], color='mediumseagreen')
        axes[1].set_title(f"{col} - Après suppression")

        plt.tight_layout()

        if save:
            from exploration.visualization import save_fig
            save_fig(
                fname=f"{col}_outliers_comparison.png",
                directory=output_dir,
                figsize=(10, 3.5),
                dpi=dpi,
                show=show,
                close=not show
            )
        elif show:
            plt.show()
        else:
            plt.close()
