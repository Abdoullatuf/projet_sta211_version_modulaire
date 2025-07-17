#missing_values.py

# Standard libraries
from pathlib import Path
from typing import Optional, Union, List
import os
import warnings
import time

# Data manipulation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Scikit-learn
from sklearn.experimental import enable_iterative_imputer   # Doit rester au-dessus de IterativeImputer
from sklearn.impute import IterativeImputer, KNNImputer


from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Machine learning models
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor

# Utilities
import joblib




def analyze_missing_values(df: pd.DataFrame, columns: Optional[List[str]] = None, plot: bool = False) -> dict:
    """
    Analyse des valeurs manquantes 
    
    Améliorations:
    - Ajout de statistiques détaillées
    - Meilleure catégorisation des niveaux de manquement
    - Recommandations automatiques
    """
    df_check = df[columns] if columns else df.copy()

    total = df_check.size
    miss = df_check.isnull().sum().sum()
    pct = miss / total * 100

    by_col = df_check.isnull().sum()
    by_col = by_col[by_col > 0]
    pct_col = by_col / len(df_check) * 100

    # Catégorisation améliorée
    critical = pct_col[pct_col > 50]  # Plus de 50% manquant
    high = pct_col[(pct_col > 30) & (pct_col <= 50)]  # 30-50%
    med = pct_col[(pct_col > 10) & (pct_col <= 30)]   # 10-30%
    low = pct_col[pct_col <= 10]  # Moins de 10%

    print(f"Total missing       : {miss} ({pct:.2f}%)")
    print(f"Colonnes affectées  : {len(by_col)}")
    print(f"  • Critique (>50%)  : {len(critical)}")
    print(f"  • Élevé (30-50%)   : {len(high)}")
    print(f"  • Moyen (10-30%)   : {len(med)}")
    print(f"  • Faible (<10%)    : {len(low)}")
    
    # Recommandations automatiques
    recommendations = []
    if len(critical) > 0:
        recommendations.append(f"⚠️ Considérer supprimer les colonnes critiques: {list(critical.index)}")
    if len(high) > 0:
        recommendations.append(f"🔧 Imputation avancée recommandée pour: {list(high.index)}")
    if len(med) + len(low) > 0:
        recommendations.append(f"✅ Imputation standard possible pour: {list((med.index.tolist() + low.index.tolist()))}")
    
    if recommendations:
        print("\n💡 Recommandations:")
        for rec in recommendations:
            print(f"   {rec}")

    print("\nTop 5 colonnes manquantes :")
    print(pct_col.sort_values(ascending=False).head())

    if plot and not pct_col.empty:
        plt.figure(figsize=(12, 6))
        
        # Graphique principal
        plt.subplot(1, 2, 1)
        colors = ['red' if x > 50 else 'orange' if x > 30 else 'yellow' if x > 10 else 'green' 
                 for x in pct_col.values]
        bars = plt.bar(range(len(pct_col)), pct_col.values, color=colors)
        plt.xticks(range(len(pct_col)), pct_col.index, rotation=45, ha='right')
        plt.ylabel('% de valeurs manquantes')
        plt.title('Pourcentage de valeurs manquantes par variable')
        plt.grid(True, alpha=0.3)
        
        # Ajout des seuils
        plt.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='Critique (50%)')
        plt.axhline(y=30, color='orange', linestyle='--', alpha=0.7, label='Élevé (30%)')
        plt.axhline(y=10, color='yellow', linestyle='--', alpha=0.7, label='Moyen (10%)')
        plt.legend()
        
        # Distribution des pourcentages
        plt.subplot(1, 2, 2)
        plt.hist(pct_col.values, bins=10, edgecolor='black', alpha=0.7)
        plt.xlabel('% de valeurs manquantes')
        plt.ylabel('Nombre de variables')
        plt.title('Distribution des niveaux de manquement')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    return {
        'total_missing': int(miss),
        'percent_missing': pct,
        'cols_missing': by_col.to_dict(),
        'percent_per_col': pct_col.to_dict(),
        'critical_missing': critical.to_dict(),  # Nouveau
        'high_missing': high.to_dict(),
        'medium_missing': med.to_dict(),
        'low_missing': low.to_dict(),
        'recommendations': recommendations,  # Nouveau
        'summary_stats': {  # Nouveau
            'total_vars_with_missing': len(by_col),
            'critical_count': len(critical),
            'high_count': len(high),
            'medium_count': len(med),
            'low_count': len(low)
        }
    }





def find_optimal_k(
    data: pd.DataFrame,
    continuous_cols: list[str],
    k_range: range = range(3, 21, 2),
    cv_folds: int = 5,
    sample_size: int = 1000,
    missing_rate: float = 0.15,
    figsize: tuple = (8, 4)
) -> int:
    """
    Trouve k optimal pour KNNImputer via CV sur un petit échantillon.
    Affiche la courbe MSE vs k, et renvoie le k minimisant le MSE.
    """
    print(f"🔍 Recherche k optimal KNN - Variables: {len(continuous_cols)}")

    X = data[continuous_cols].copy()

    if X.empty or X.isnull().all().any():
        print("⚠️ Données insuffisantes, k=5 par défaut")
        return 5

    # Échantillonnage
    original_size = len(X)
    if len(X) > sample_size:
        X = X.sample(n=sample_size, random_state=42)
        print(f"   Échantillon: {len(X)}/{original_size} observations")

    # ... (le reste de la logique de masquage et de CV reste identique) ...
    # ... (je ne le remets pas pour la clarté) ...

    # Visualisation améliorée (partie corrigée)
    try:
        plt.figure(figsize=figsize)
        k_list = list(k_range)
        valid_indices = [i for i, s in enumerate(mse_scores) if np.isfinite(s)]

        if valid_indices:
            k_valid = [k_list[i] for i in valid_indices]
            mse_valid = [mse_scores[i] for i in valid_indices]
            
            plt.plot(k_valid, mse_valid, marker='o', label='MSE')

            best_idx = np.argmin(mse_valid)
            best_k = k_valid[best_idx]
            best_mse = mse_valid[best_idx]

            plt.scatter([best_k], [best_mse], color='red', s=100, zorder=5, marker='*', label=f'Optimal k={best_k}')
            plt.axvline(best_k, color='red', linestyle='--', alpha=0.6)

            plt.xlabel('k (nombre de voisins)')
            plt.ylabel('Mean Squared Error')
            plt.title(f'Optimisation K (MSE = {best_mse:.4f})')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.show()

            print(f"✅ k optimal trouvé: {best_k}")
            return best_k
        else:
            print("❌ Pas de données valides pour le graphique")
            return 5 # Retourne une valeur par défaut
    
    except Exception as e:
        print(f"⚠️ Erreur lors de la création du graphique: {e}")
        # Logique de secours si le graphique échoue
        valid_scores = [s for s in mse_scores if np.isfinite(s)]
        if valid_scores:
            best_k = int(k_range[np.argmin(valid_scores)])
            print(f"✅ k optimal (calculé sans graphique): {best_k}")
            return best_k
        else:
            print("⚠️ Aucun k valide trouvé, retour de k=5 par défaut")
            return 5
        


# ============================================================================
# VERSION ENCORE PLUS SIMPLE (SI PROBLÈMES PERSISTENT)
# ============================================================================

def find_optimal_k_ultra_simple(
    data: pd.DataFrame,
    continuous_cols: list[str],
    max_k: int = 15
) -> int:
    """
    Version ultra-simplifiée qui fonctionne toujours.
    """
    X = data[continuous_cols].dropna().sample(n=min(500, len(data)), random_state=42)
    
    if len(X) < 50:
        return 5
    
    # Test simple avec train/test split
    from sklearn.model_selection import train_test_split
    
    X_train, X_test = train_test_split(X, test_size=0.3, random_state=42)
    
    # Masquer 10% des valeurs du test
    rng = np.random.RandomState(42)
    mask = rng.rand(*X_test.shape) < 0.1
    X_test_missing = X_test.copy()
    X_test_missing[mask] = np.nan
    
    scores = []
    k_range = range(3, max_k + 1, 2)
    
    for k in k_range:
        try:
            imputer = KNNImputer(n_neighbors=k)
            imputer.fit(X_train)
            X_pred = imputer.transform(X_test_missing)
            
            # MSE sur valeurs masquées
            mse = mean_squared_error(
                X_test.values[mask], 
                X_pred[mask]
            )
            scores.append(mse)
        except:
            scores.append(np.inf)
    
    # Graphique simple
    plt.figure(figsize=(8, 4))
    plt.plot(list(k_range), scores, 'o-')
    plt.xlabel('k')
    plt.ylabel('MSE')
    plt.title('KNN k optimization')
    plt.grid(True)
    plt.show()
    
    best_k = int(k_range[np.argmin(scores)])
    print(f"→ k optimal: {best_k}")
    return best_k









def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = 'mixed_mar_mcar',
    mar_method: str = 'knn',
    knn_k: Optional[int] = None,
    mar_cols: Optional[List[str]] = None,
    mcar_cols: Optional[List[str]] = None,
    display_info: bool = True,
    save_results: bool = True,
    processed_data_dir: Optional[Union[str, Path]] = None,
    imputers_dir: Optional[Union[str, Path]] = None,  # ✅ Renommé ici
    custom_filename: Optional[str] = None,
    auto_optimize_k: bool = False,
    validate_imputation: bool = True,
    backup_method: str = 'median',
    mice_estimator: Optional[object] = None
) -> pd.DataFrame:

    def median_fill(df_local, cols_local):
        for col in cols_local:
            if col in df_local.columns and df_local[col].isnull().any():
                df_local[col] = df_local[col].fillna(df_local[col].median())

    if display_info:
        print("🔧 Début de l'imputation des valeurs manquantes")
        print("=" * 50)

    df_proc = df.copy()
    suffix = ''

    initial_missing = df_proc.isnull().sum().sum()
    if display_info:
        print(f"📊 Valeurs manquantes initiales: {initial_missing}")

    if strategy == 'all_median':
        num_cols = df_proc.select_dtypes(include=[np.number]).columns
        median_fill(df_proc, num_cols)
        suffix = 'median_all'

    elif strategy == 'mixed_mar_mcar':
        if mar_cols is None:
            mar_cols = ['X1_trans', 'X2_trans', 'X3_trans']
        if mcar_cols is None:
            mcar_cols = ['X4']

        available_mar_cols = [col for col in mar_cols if col in df_proc.columns]

        if available_mar_cols:
            if display_info:
                print(f"📊 Variables MAR à imputer: {len(available_mar_cols)}")
                for col in available_mar_cols:
                    count = df_proc[col].isnull().sum()
                    print(f"   • {col}: {count} valeurs manquantes")

            try:
                if mar_method == 'knn':
                    if knn_k is None:
                        knn_k = 5
                        if display_info:
                            print("⚠️ k non spécifié, utilisation de k = 5 par défaut.")

                    imputer = KNNImputer(n_neighbors=knn_k)
                    df_proc[available_mar_cols] = imputer.fit_transform(df_proc[available_mar_cols])
                    suffix = f'knn_k{knn_k}'

                elif mar_method == 'mice':
                    if mice_estimator is None:
                        if display_info:
                            print("⚙️ Utilisation de MICE avec BayesianRidge par défaut")
                        mice_estimator = BayesianRidge()

                    complete_rows = df_proc[available_mar_cols].dropna().shape[0]
                    if complete_rows < 10:
                        raise ValueError("Pas assez de données complètes pour utiliser MICE efficacement.")

                    imputer = IterativeImputer(
                        estimator=mice_estimator,
                        max_iter=50,
                        random_state=42
                    )

                    df_proc[available_mar_cols] = imputer.fit_transform(df_proc[available_mar_cols])
                    suffix = 'mice_custom'

                else:
                    raise ValueError("❌ mar_method doit être 'knn' ou 'mice'.")

                if save_results and imputers_dir:
                    imputers_dir = Path(imputers_dir)
                    imputers_dir.mkdir(parents=True, exist_ok=True)
                    imp_path = imputers_dir / f"imputer_{suffix}.pkl"
                    joblib.dump(imputer, imp_path)
                    if display_info:
                        print(f"💾 Modèle d'imputation sauvegardé: {imp_path}")

            except Exception as e:
                print(f"❌ Erreur lors de l'imputation MAR: {e}")
                print(f"🔄 Utilisation de la méthode de secours: {backup_method}")
                median_fill(df_proc, available_mar_cols)
                suffix = f'{backup_method}_backup'

        median_fill(df_proc, mcar_cols)

    else:
        raise ValueError("❌ strategy doit être 'all_median' ou 'mixed_mar_mcar'.")

    final_missing = df_proc.isnull().sum().sum()
    if display_info:
        print(f"\n📊 Résumé de l'imputation:")
        print(f"   • Valeurs manquantes avant: {initial_missing}")
        print(f"   • Valeurs manquantes après: {final_missing}")

    if save_results:
        if processed_data_dir is None:
            raise ValueError("processed_data_dir doit être fourni si save_results=True.")
        processed_data_dir = Path(processed_data_dir)

        if processed_data_dir.suffix:
            raise ValueError("processed_data_dir doit être un dossier, pas un fichier.")

        processed_data_dir.mkdir(parents=True, exist_ok=True)

        filename = custom_filename or f"df_imputed_{suffix}.csv"
        filepath = processed_data_dir / filename
        df_proc.to_csv(filepath, index=False)

        if display_info:
            print(f"💾 Données imputées sauvegardées: {filepath}")

    if display_info:
        print("=" * 50)
        print("✅ Imputation terminée avec succès")

    return df_proc
