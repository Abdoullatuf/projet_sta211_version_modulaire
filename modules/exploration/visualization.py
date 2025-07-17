# modules/exploration/visualization.py

import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def save_fig(
    fname: str,
    directory: Union[str, Path],
    dpi: int = 150,
    figsize: Optional[Tuple[float, float]] = None,
    format: str = "png",
    close: bool = False,
    show: bool = True,
    **kwargs
) -> Path:
    """
    Sauvegarde la figure matplotlib courante.
    """
    if plt.get_fignums() == 0:
        raise ValueError("Aucune figure Matplotlib active à sauvegarder.")

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / fname
    if not path.suffix:
        path = path.with_suffix(f".{format}")

    if figsize:
        plt.gcf().set_size_inches(figsize)

    plt.savefig(path, dpi=dpi, bbox_inches="tight", format=format, **kwargs)
    if show:
        plt.show()
    if close:
        plt.close(plt.gcf())
    print(f"✅ Figure sauvegardée : {path}")
    return path

def visualize_distributions_and_boxplots(df: pd.DataFrame, continuous_cols: List[str], output_dir: Path) -> None:
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
    save_fig(
        fname="continuous_distributions_boxplots.png",
        directory=output_dir,
        dpi=150,
        format="png",
        show=True,
        close=True
    )

def plot_continuous_by_class(df, continuous_cols, output_dir, figsize=(15, 5)):
    fig, axes = plt.subplots(1, len(continuous_cols), figsize=figsize)
    for i, col in enumerate(continuous_cols):
        df_clean = df[[col, 'y']].dropna()
        df_clean['y_label'] = df_clean['y'].map({0: 'Non-pub', 1: 'Pub'})
        sns.violinplot(data=df_clean, x='y_label', y=col, ax=axes[i])
        axes[i].set_title(f'{col} par classe')
    plt.tight_layout()
    save_fig('continuous_by_class.png', directory=output_dir, figsize=figsize)

def plot_continuous_target_corr(df, continuous_cols, output_dir):
    fig, ax = plt.subplots(figsize=(8, 4))
    corr_data = df[continuous_cols + ['y']].corr()['y'][:-1].to_frame()
    sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0,
                cbar_kws={'label': 'Corrélation'}, fmt='.3f', ax=ax)
    ax.set_title('Corrélations avec la cible')
    plt.tight_layout()
    save_fig('continuous_target_correlation.png', directory=output_dir, figsize=(8, 4))

def plot_binary_sparsity(df, binary_cols, output_dir, sample_size=100):
    sample_vars = np.random.choice(binary_cols, size=min(sample_size, len(binary_cols)), replace=False)
    sample_data = df[sample_vars].head(sample_size)
    plt.figure(figsize=(8, 4))
    plt.imshow(sample_data.values, cmap='binary', aspect='auto')
    plt.colorbar(label='Valeur (0 ou 1)')
    plt.title('Sparsité (100 obs × 100 var binaires)')
    plt.xlabel('Variables')
    plt.ylabel('Observations')
    plt.tight_layout()
    save_fig('sparsity_visualization.png', directory=output_dir, figsize=(8, 4))



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
    Affiche les boxplots avant/après traitement des outliers pour chaque variable.

    Args:
        df_before (pd.DataFrame): Données avant traitement.
        df_after (pd.DataFrame): Données après traitement.
        cols (list): Colonnes à analyser.
        output_dir (str or Path): Dossier de sauvegarde des figures.
        show (bool): Affiche les figures si True.
        save (bool): Sauvegarde les figures si True.
        dpi (int): Résolution de l’image sauvegardée.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for col in cols:
        fig, axes = plt.subplots(1, 2, figsize=(8, 3))

        sns.boxplot(x=df_before[col], ax=axes[0], color='salmon')
        axes[0].set_title(f"{col} - Avant traitement")

        sns.boxplot(x=df_after[col], ax=axes[1], color='mediumseagreen')
        axes[1].set_title(f"{col} - Après traitement")

        plt.tight_layout()

        if save:
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
