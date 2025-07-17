import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import warnings
from typing import List, Tuple, Dict, Optional
import time

def find_optimal_k_v2(
    df: pd.DataFrame,
    columns_to_impute: List[str],
    k_range: range = range(3, 21, 2),
    cv_folds: int = 5,
    sample_size: Optional[int] = 1000,
    test_size: float = 0.2,
    random_state: int = 42,
    metric: str = 'mse',
    plot_results: bool = True,
    verbose: bool = True,
    figsize: Tuple[int, int] = (10, 6)
) -> Dict:
    """
    Trouve la valeur optimale de K pour l'imputation KNN par validation croisée.
    Version corrigée pour être compatible avec les anciennes versions de scikit-learn.
    """

    # ... (Le début de la fonction reste identique) ...
    start_time = time.time()

    if verbose:
        print("🔍 Recherche de la valeur optimale K pour l'imputation KNN")
        print("=" * 60)
        print(f"📊 Colonnes à évaluer      : {columns_to_impute}")
        print(f"🎯 Plage K à tester        : {list(k_range)}")
        print(f"🔄 Validation croisée      : {cv_folds} folds")
        print(f"📏 Métrique d'évaluation  : {metric.upper()}")
        print("-" * 60)

    if not columns_to_impute:
        raise ValueError("❌ La liste 'columns_to_impute' ne peut pas être vide.")

    for col in columns_to_impute:
        if col not in df.columns:
            raise ValueError(f"❌ Colonne '{col}' introuvable dans le DataFrame.")

    df_work = df[columns_to_impute].copy()
    df_complete = df_work.dropna()

    if len(df_complete) < cv_folds:
        raise ValueError("Moins de lignes complètes que de folds pour la CV.")

    if sample_size is not None and len(df_complete) > sample_size:
        if verbose:
            print(f"Échantillonnage de {sample_size} lignes parmi {len(df_complete)} pour l'optimisation.")
        df_complete = df_complete.sample(n=sample_size, random_state=random_state)

    if verbose:
        print(f"\n🧮 Données utilisées pour le test : {len(df_complete):,} lignes.")

    results = {'k_values': list(k_range), 'scores_mean': [], 'scores_std': []}

    for k in k_range:
        if verbose:
            print(f"\n🔄 Test K={k:2d} ", end="")

        fold_scores = []
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(df_complete)):
            df_train_fold, df_test_fold = df_complete.iloc[train_idx], df_complete.iloc[test_idx]

            df_test_masked = df_test_fold.copy()
            mask = np.random.RandomState(random_state + fold_idx).rand(*df_test_masked.shape) < test_size
            mask[df_test_masked.isnull()] = False

            true_values = df_test_fold.to_numpy()[mask]
            df_test_masked.iloc[mask] = np.nan

            if df_test_masked.isnull().values.sum() == 0:
                continue

            imputer = KNNImputer(n_neighbors=k)
            df_imputed = pd.DataFrame(imputer.fit(df_train_fold).transform(df_test_masked),
                                      columns=df_test_masked.columns, index=df_test_masked.index)

            imputed_values = df_imputed.to_numpy()[mask]

            if len(true_values) > 0 and len(imputed_values) > 0:
                # --- DÉBUT DE LA CORRECTION ---
                # 1. Calculez toujours le MSE
                mse = mean_squared_error(true_values, imputed_values)

                # 2. Adaptez le score en fonction de la métrique demandée
                if metric.lower() == 'rmse':
                    score = np.sqrt(mse)
                elif metric.lower() == 'mae':
                    # Le MAE est une métrique différente, il faut l'importer ou la calculer manuellement
                    from sklearn.metrics import mean_absolute_error
                    score = mean_absolute_error(true_values, imputed_values)
                else: # 'mse' est le cas par défaut
                    score = mse
                # --- FIN DE LA CORRECTION ---

                fold_scores.append(score)

        if fold_scores:
            mean_score, std_score = np.mean(fold_scores), np.std(fold_scores)
            results['scores_mean'].append(mean_score)
            results['scores_std'].append(std_score)
            if verbose:
                print(f"→ {metric.upper()}: {mean_score:.4f} (±{std_score:.4f})")
        else:
            results['scores_mean'].append(np.inf)
            results['scores_std'].append(0)
            if verbose:
                print("→ ❌ Échec (pas de scores valides)")

    valid_scores = [(i, score) for i, score in enumerate(results['scores_mean']) if np.isfinite(score)]
    if not valid_scores:
        raise RuntimeError("❌ Aucune valeur K n'a produit de résultats valides.")

    best_idx, best_score = min(valid_scores, key=lambda x: x[1])
    optimal_k = results['k_values'][best_idx]

    if plot_results and len(valid_scores) > 1:
        plt.figure(figsize=figsize)
        k_vals = [results['k_values'][i] for i, _ in valid_scores]
        scores = [score for _, score in valid_scores]
        stds = [results['scores_std'][i] for i, _ in valid_scores]
        plt.errorbar(k_vals, scores, yerr=stds, marker='o', capsize=5, label='Score moyen par K')
        plt.axvline(x=optimal_k, color='red', linestyle='--', label=f'K optimal = {optimal_k}')
        plt.scatter([optimal_k], [best_score], color='red', s=150, zorder=5, marker='*')
        plt.title(f'Optimisation K pour Imputation KNN ({metric.upper()})', fontsize=14)
        plt.xlabel('Nombre de voisins (K)')
        plt.ylabel(f'Score ({metric.upper()})')
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.show()

    computation_time = time.time() - start_time
    if verbose:
        print("\n" + "=" * 60)
        print("🎯 RÉSULTATS DE L'OPTIMISATION")
        print("=" * 60)
        print(f"🏆 K optimal              : {optimal_k}")
        print(f"📊 Meilleur score ({metric.upper()})  : {best_score:.4f}")
        print(f"⏱️  Temps de calcul        : {computation_time:.2f}s")

    return {
        'optimal_k': optimal_k,
        'best_score': best_score,
        'results_df': pd.DataFrame(results)
    }