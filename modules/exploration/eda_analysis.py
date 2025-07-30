#eda_analysis.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import List, Dict, Optional, Tuple, Union

from exploration.visualization import save_fig



def univariate_analysis(data, paths=None, display_figures=True, save_fig=True):
    """Analyse univariée complète avec sauvegarde facultative de la figure."""
    if paths is None:
        from project_setup import setup_project_paths
        paths = setup_project_paths()

    print("\n=== Analyse Univariée ===")

    target_dist = data['outcome'].value_counts()
    print("\n📌 Distribution de la variable cible :")
    print(target_dist)
    print(f"\nPourcentage de la classe majoritaire : {(target_dist.max() / len(data)) * 100:.2f}%")

    bin_cols = [col for col in data.select_dtypes(include='int64').columns if col != 'outcome']
    if bin_cols:
        print(f"\n📊 {len(bin_cols)} variables binaires détectées.")
        print("Extrait :")
        print(bin_cols[:10], "..." if len(bin_cols) > 10 else "")
        for col in bin_cols[:5]:
            print(f"\n- {col}")
            print(data[col].value_counts(normalize=True).round(3))

    cont_cols = data.select_dtypes(include='float64').columns
    if cont_cols.any():
        print(f"\n📈 Statistiques descriptives des variables continues ({len(cont_cols)} colonnes) :")
        print(data[cont_cols].describe().T.round(2))

        if display_figures:
            #plt.figure(figsize=(16, 10))
            data[cont_cols].hist(bins=20, figsize=(16, 10), color='lightcoral', edgecolor='black')
            plt.suptitle("Distribution des variables continues", fontsize=16)
            plt.tight_layout()
            if save_fig:
                fig_path = paths["FIGURES_DIR"] / "distribution_variables_continues.png"
                plt.savefig(fig_path)
                print(f"\n✅ Figure sauvegardée dans : {fig_path}")
            plt.show();

    missing = data.isnull().sum()
    if missing.any():
        print("\n⚠️ Valeurs manquantes détectées :")
        print(missing[missing > 0])



def bivariate_analysis(
    data: pd.DataFrame,
    use_transformed: bool = True,
    display_correlations: bool = True,
    top_n: int = 10,
    show_plot: bool = True
):
    """
    Analyse bivariée complète :
    - Corrélations continues et binaires avec la cible
    - Redondances internes (variables continues & binaires)

    Args:
        data: DataFrame contenant les données
        use_transformed: True = utilise X1_trans, X2_trans, X3_trans
        display_correlations: affiche les résultats dans la console
        top_n: nombre de variables à afficher dans le top
        show_plot: génère un barplot des meilleures variables

    Returns:
        corr_df: DataFrame trié des corrélations avec la cible
        high_corr_pairs: paires de variables continues fortement corrélées
        binary_corr_pairs: paires binaires très redondantes
    """
    print("\n=== Analyse Bivariée ===")

    # 1. Encodage cible binaire
    target_numeric = (data['outcome'] == 'ad.').astype(int)

    # 2. Sélection des variables
    continuous_vars = [col for col in ['X1_trans', 'X2_trans', 'X3_trans'] if col in data.columns] \
        if use_transformed else [col for col in ['X1', 'X2', 'X3'] if col in data.columns]

    binary_vars = [
        col for col in data.columns
        if data[col].dropna().nunique() == 2 and col != 'outcome'
    ]

    # 3. Corrélation avec la cible
    correlations = []
    for col in continuous_vars:
        corr = data[col].corr(target_numeric)
        correlations.append((col, corr))

    for col in binary_vars:
        try:
            corr, _ = pointbiserialr(data[col], target_numeric)
            correlations.append((col, corr))
        except Exception:
            continue

    corr_df = pd.DataFrame(correlations, columns=['feature', 'correlation']).dropna()
    corr_df = corr_df.sort_values('correlation', key=abs, ascending=False)

    if display_correlations:
        print(f"\n🔝 Top {top_n} variables les plus corrélées à la cible :")
        print(corr_df.head(top_n))

    if show_plot:
        plt.figure(figsize=(8, 5))
        sns.barplot(data=corr_df.head(top_n), y='feature', x='correlation', palette='viridis')
        plt.title(f"Top {top_n} variables corrélées à la cible")
        plt.tight_layout()
        plt.show()

    # 4. Corrélation interne (variables continues)
    high_corr_pairs = []
    if continuous_vars:
        corr_matrix = data[continuous_vars].corr()
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                val = corr_matrix.iloc[i, j]
                if abs(val) > 0.9:
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], val))

    # 5. Corrélation interne (variables binaires)
    binary_corr_pairs = []
    if len(binary_vars) > 1:
        bin_corr = data[binary_vars].corr()
        for i in range(len(bin_corr.columns)):
            for j in range(i + 1, len(bin_corr.columns)):
                val = bin_corr.iloc[i, j]
                if abs(val) > 0.95:
                    binary_corr_pairs.append((bin_corr.columns[i], bin_corr.columns[j], val))

    return corr_df, high_corr_pairs, binary_corr_pairs


def multivariate_analysis(
    data: pd.DataFrame,
    target_col: str = "outcome",
    threshold: float = 0.90,
    plot_corr: bool = True,
    return_matrix: bool = False,
    save_fig: bool = False,
    fig_name: str = "correlation_matrix_continues.png"
):
    """
    Analyse multivariée (corrélation entre variables continues uniquement).

    Args:
        data (pd.DataFrame): jeu de données
        target_col (str): nom de la variable cible
        threshold (float): seuil de corrélation forte
        plot_corr (bool): affiche la heatmap
        return_matrix (bool): retourne aussi la matrice complète
        save_fig (bool): sauvegarde la figure dans FIGURES_DIR
        fig_name (str): nom du fichier figure

    Returns:
        list: paires fortement corrélées
        optionnel: matrice de corrélation
    """
    print("\n=== Analyse Multivariée (sur float64 uniquement) ===")

    # 1. Sélection des colonnes continues
    float_cols = data.select_dtypes(include='float64').columns
    float_cols = [col for col in float_cols if col != target_col]

    if not float_cols:
        print("⚠️ Aucune variable continue détectée.")
        return [] if not return_matrix else ([], pd.DataFrame())

    # 2. Matrice de corrélation
    corr_matrix = data[float_cols].corr()

    # 3. Recherche de paires fortement corrélées
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            val = corr_matrix.iloc[i, j]
            if abs(val) > threshold:
                high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], val))

    # 4. Heatmap (optionnelle)
    if plot_corr or save_fig:
        plt.figure(figsize=(6, 4))
        sns.heatmap(corr_matrix, cmap='coolwarm', annot=True, fmt=".2f", square=True)
        plt.title("Corrélation entre variables continues")
        plt.tight_layout()
        if save_fig:
            paths = setup_project_paths()
            fig_path = paths["FIGURES_DIR"] / fig_name
            plt.savefig(fig_path)
            print(f"✅ Figure sauvegardée dans : {fig_path}")
        if plot_corr:
            plt.show()
        else:
            plt.close()

    # 5. Retour
    high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    return (high_corr_pairs, corr_matrix) if return_matrix else high_corr_pairs


def full_correlation_analysis(
    df_study: pd.DataFrame,
    continuous_cols: List[str],
    presence_rates: Dict[str, float],
    FIGURES_DIR: Union[str, Path],
    ROOT_DIR: Union[str, Path],
    top_n_corr_features: int = 20,
    save_json: bool = True,
    figsize_corr_matrix: Tuple[int, int] = (8, 6),
    figsize_binary: Tuple[int, int] = (10, 5)
) -> None:
    from exploration.visualization import save_fig
    import json


    print("🔗 Analyse des corrélations")
    print("=" * 60)

    # Sélection de variables pour l'analyse
    print("\n📊 Sélection des variables pour l’analyse des corrélations...")
    presence_series = pd.Series(presence_rates)
    quartiles = presence_series.quantile([0.25, 0.5, 0.75])

    

    vars_q1 = presence_series[presence_series <= quartiles[0.25]].sample(10, random_state=42).index.tolist()
    vars_q2 = presence_series[(presence_series > quartiles[0.25]) & (presence_series <= quartiles[0.5])].sample(10, random_state=42).index.tolist()
    vars_q3 = presence_series[(presence_series > quartiles[0.5]) & (presence_series <= quartiles[0.75])].sample(10, random_state=42).index.tolist()
    vars_q4 = presence_series[presence_series > quartiles[0.75]].sample(10, random_state=42).index.tolist()

    selected_vars = continuous_cols + vars_q1 + vars_q2 + vars_q3 + vars_q4
    print(f"  - Variables sélectionnées : {len(selected_vars)} (3 continues + 40 binaires)")

    print(f" Analyse sur un sous-échantillon : {len(continuous_cols)} continues + {len(selected_vars) - len(continuous_cols)} binaires")
    print("ℹ️ Cette matrice de corrélation n’inclut qu’un échantillon aléatoire de 40 variables binaires (10 par quartile de taux de présence).")
    print("ℹ️ Pour une analyse complète des redondances, voir la section bivariée ci-dessous.")



    print("\n📊 Calcul de la matrice de corrélation...")
    corr_matrix = df_study[selected_vars + ['y']].corr()

    # Corrélation avec la cible
    target_corr = corr_matrix['y'].drop('y').sort_values(ascending=False)
    print("\n🎯 Top 10 corrélations avec la cible (y) :")
    for var, corr in target_corr.head(10).items():
        print(f"  - {var}: {corr:.4f}")

    print("\n🎯 Bottom 10 corrélations avec la cible (y) :")
    for var, corr in target_corr.tail(10).items():
        print(f"  - {var}: {corr:.4f}")

    # Heatmap de corrélation
    plt.figure(figsize=figsize_corr_matrix)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix, mask=mask, cmap='coolwarm', center=0,
        square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, vmin=-0.5, vmax=0.5
    )
    plt.title('Matrice de corrélation (échantillon)', fontsize=14)
    plt.tight_layout()
    save_fig("correlation_matrix_sample.png", directory=FIGURES_DIR / "eda", figsize=figsize_corr_matrix)

    # Analyse de la corrélation entre features
    print("\n🔍 Analyse des corrélations entre features :")
    upper_triangle = corr_matrix.where(np.triu(np.ones_like(corr_matrix, dtype=bool), k=1))
    high_corr_pairs = [
        {'var1': c1, 'var2': c2, 'corr': upper_triangle.loc[c1, c2]}
        for c1 in upper_triangle.columns
        for c2 in upper_triangle.columns
        if c1 != c2 and not pd.isna(upper_triangle.loc[c1, c2]) and abs(upper_triangle.loc[c1, c2]) > 0.8
    ]
    if high_corr_pairs:
        print(f"⚠️ Paires de variables très corrélées : {len(high_corr_pairs)}")
        for pair in high_corr_pairs[:5]:
            print(f"  - {pair['var1']} vs {pair['var2']}: {pair['corr']:.3f}")
    else:
        print("✅ Aucune paire avec |corr| > 0.8")

    print("\n## 4.5 Analyse approfondie des corrélations <a id='analyse-approfondie-correlations'></a>\n")
    print("🔗 Analyse approfondie des corrélations entre variables...")

    # Analyse bivariée
    df_bivariate = df_study.copy()
    df_bivariate['outcome'] = df_bivariate['y'].map({0: 'noad.', 1: 'ad.'})

    corr_df, high_corr_pairs, binary_corr_pairs = bivariate_analysis(
        data=df_bivariate,
        use_transformed=False,
        display_correlations=True,
        top_n=top_n_corr_features,
        show_plot=False
    )

    # Affichage des corrélations binaires élevées
    print(f"\n🔢 Variables binaires très corrélées (|r| > 0.95):")
    if binary_corr_pairs:
        print(f"  - Nombre total : {len(binary_corr_pairs)}")
        corr_values = [abs(corr) for _, _, corr in binary_corr_pairs]
        print(f"  - Moyenne : {np.mean(corr_values):.3f}")
        print(f"  - Max : {max(corr_values):.3f}")
        print("\n  Exemples :")
        for i, (v1, v2, c) in enumerate(binary_corr_pairs[:5]):
            print(f"    {i+1}. {v1} ↔ {v2}: r = {c:.3f}")

        # Visualisation
        plt.figure(figsize=figsize_binary)
        plt.subplot(1, 2, 1)
        plt.hist(corr_values, bins=20, color='coral', edgecolor='black')
        plt.axvline(0.95, color='red', linestyle='--', label='Seuil 0.95')
        plt.xlabel('|Corrélation|')
        plt.ylabel('Nombre de paires')
        plt.title(f'Distribution ({len(binary_corr_pairs)} paires)')
        plt.legend()

        var_counts = {}
        for v1, v2, _ in binary_corr_pairs:
            var_counts[v1] = var_counts.get(v1, 0) + 1
            var_counts[v2] = var_counts.get(v2, 0) + 1

        top_vars = sorted(var_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        plt.subplot(1, 2, 2)
        plt.barh([v for v, _ in top_vars], [c for _, c in top_vars], color='skyblue')
        plt.xlabel('Nb de corrélations élevées')
        plt.title('Top 10 variables redondantes')
        plt.tight_layout()
        save_fig('binary_correlations_analysis.png', directory=FIGURES_DIR / "eda", figsize=figsize_binary)
    else:
        print("✅ Aucune paire de variables binaires avec |r| > 0.95")

    # Résumé
    print("\n💡 Résumé :")
    print(f"  - Corrélation max avec y : {abs(target_corr).max():.3f}")
    print(f"  - Dataset sparse avec peu de multicolinéarité")
    if binary_corr_pairs:
        print("  - Réduction de dimension recommandée (sélection ou PCA)")

    # Sauvegarde
    if save_json:
        results = {
            'top_correlations_with_target': corr_df.head(top_n_corr_features).to_dict(),
            'n_high_corr_continuous': len(high_corr_pairs),
            'n_high_corr_binary': len(binary_corr_pairs),
            'redundant_variables': list(var_counts.keys()) if binary_corr_pairs else []
        }
        out_path = ROOT_DIR / "results" / "bivariate_analysis_results.json"
        out_path.parent.mkdir(exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n✅ Résultats sauvegardés dans : {out_path}")
