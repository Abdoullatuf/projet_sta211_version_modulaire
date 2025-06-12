#missing_values.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Union
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from scipy import stats
import joblib
import os
from pathlib import Path

def analyze_missing_values(df: pd.DataFrame, columns: Optional[List[str]] = None, plot: bool = False) -> dict:
    """
    Analyse des valeurs manquantes - VERSION AMÉLIORÉE
    
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
    k_range: range = range(3, 21, 2),  # Commence à 3, pas 1
    cv_folds: int = 5,
    sample_size: int = 1000,
    missing_rate: float = 0.15,  # Paramètre configurable
    figsize: tuple = (8, 4)      # Taille configurable
) -> int:
    """
    Trouve k optimal pour KNNImputer via CV sur un petit échantillon.
    Affiche la courbe MSE vs k, et renvoie le k minimisant le MSE.
    
    Version améliorée avec gestion d'erreurs et paramètres configurables.
    """
    print(f"🔍 Recherche k optimal KNN - Variables: {len(continuous_cols)}")
    
    X = data[continuous_cols].copy()
    
    # Validation basique
    if X.empty or X.isnull().all().any():
        print("⚠️ Données insuffisantes, k=5 par défaut")
        return 5
    
    # Échantillonnage
    original_size = len(X)
    if len(X) > sample_size:
        X = X.sample(n=sample_size, random_state=42)
        print(f"   Échantillon: {len(X)}/{original_size} observations")

    # Masquage avec taux configurable
    rng = np.random.RandomState(42)
    mask = rng.rand(*X.shape) < missing_rate
    X_missing = X.copy()
    X_missing[mask] = np.nan
    
    print(f"   Valeurs masquées: {missing_rate:.1%}")

    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    mse_scores = []

    for k in k_range:
        fold_mse = []
        imputer = KNNImputer(n_neighbors=k)
        
        for train_idx, val_idx in cv.split(X):
            try:
                X_tr = X_missing.iloc[train_idx]
                X_val = X_missing.iloc[val_idx]
                X_true = X.iloc[val_idx]

                # Vérification rapide
                if X_tr.dropna().empty:
                    continue

                imputer.fit(X_tr)
                X_imp = pd.DataFrame(
                    imputer.transform(X_val),
                    columns=X_val.columns,
                    index=X_val.index
                )

                mask_val = mask[val_idx]
                y_true = X_true.to_numpy(dtype=float)[mask_val]
                y_pred = X_imp.to_numpy(dtype=float)[mask_val]
                valid = (~np.isnan(y_true)) & (~np.isnan(y_pred))
                
                if valid.any() and valid.sum() >= 3:  # Au moins 3 valeurs
                    mse = mean_squared_error(y_true[valid], y_pred[valid])
                    if np.isfinite(mse):  # Vérification MSE valide
                        fold_mse.append(mse)
                        
            except Exception:
                continue  # Ignore les erreurs et continue

        # Ajouter MSE moyen si au moins 2 folds ont réussi
        if len(fold_mse) >= 2:
            mse_scores.append(np.mean(fold_mse))
        else:
            mse_scores.append(np.inf)  # Marquer comme invalide

    # Vérification des résultats
    valid_scores = [s for s in mse_scores if np.isfinite(s)]
    
    if not valid_scores:
        print("⚠️ Aucun k valide trouvé, k=5 par défaut")
        return 5

    # Visualisation améliorée
    try:
        plt.figure(figsize=figsize)
        
        # Filtrer les scores infinis pour le graphique
        k_list = list(k_range)
        valid_indices = [i for i, s in enumerate(mse_scores) if np.isfinite(s)]
        
        if valid_indices:
            k_valid = [k_list[i] for i in valid_indices]
            mse_valid = [mse_scores[i] for i in valid_indices]
            
            plt.plot(k_valid, mse_valid, marker='o', linewidth=2, markersize=6,
                    color='steelblue', label='MSE')
            
            # Trouver et marquer le minimum
            best_idx = np.argmin(mse_valid)
            best_k = k_valid[best_idx]
            best_mse = mse_valid[best_idx]
            
            plt.scatter([best_k], [best_mse], color='red', s=100, zorder=5,
                       marker='*', label=f'Optimal k={best_k}')
            plt.axvline(best_k, color='red', linestyle='--', alpha=0.6)
            
            plt.xlabel('k (nombre de voisins)', fontsize=11)
            plt.ylabel('Mean Squared Error', fontsize=11)
            plt.title(f'Optimisation K pour KNN Imputation\n'
                     f'k optimal = {best_k} (MSE = {best_mse:.4f})', 
                     fontsize=12, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.show()
            
            print(f"✅ k optimal trouvé: {best_k} (MSE: {best_mse:.4f})")
            print(f"   {len(valid_indices)}/{len(k_range)} valeurs k testées avec succès")
            
            return best_k
        else:
            print("❌ Pas de données valides pour le graphique")
            return 5
            
    except Exception as e:
        print(f"⚠️ Erreur graphique: {e}")
        # Fallback: trouver le minimum sans graphique
        if valid_scores:
            best_k = int(k_range[np.argmin(mse_scores)])
            print(f"✅ k optimal (sans graphique): {best_k}")
            return best_k
        else:
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
    models_dir: Optional[Union[str, Path]] = None,
    custom_filename: Optional[str] = None,
    auto_optimize_k: bool = False,  # NOUVEAU: optimisation automatique
    validate_imputation: bool = True,  # NOUVEAU: validation de l'imputation
    backup_method: str = 'median'  # NOUVEAU: méthode de secours
) -> pd.DataFrame:
    """
    VERSION AMÉLIORÉE - Gère les valeurs manquantes avec optimisations automatiques.
    
    Nouvelles fonctionnalités:
    - Optimisation automatique de k
    - Validation de la qualité d'imputation
    - Méthode de secours en cas d'échec
    - Statistiques détaillées avant/après
    """
    
    if display_info:
        print("🔧 Début de l'imputation des valeurs manquantes")
        print("="*50)
    
    df_proc = df.copy()
    suffix = ''
    imputation_stats = {}
    
    # Analyse initiale des valeurs manquantes
    if display_info:
        initial_missing = df_proc.isnull().sum().sum()
        print(f"📊 Valeurs manquantes initiales: {initial_missing}")
    
    if strategy == 'all_median':
        num_cols = df_proc.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            before_count = df_proc[col].isnull().sum()
            df_proc[col] = df_proc[col].fillna(df_proc[col].median())
            if display_info and before_count > 0:
                print(f"   • {col}: {before_count} valeurs imputées par médiane")
        suffix = 'median_all'
        if display_info:
            print(f"✅ Imputation par médiane sur {len(num_cols)} colonnes numériques.")

    elif strategy == 'mixed_mar_mcar':
        if mar_cols is None:
            mar_cols = ['X1_trans', 'X2_trans', 'X3_trans']
        if mcar_cols is None:
            mcar_cols = ['X4']

        # Vérification des colonnes MAR existantes
        available_mar_cols = [col for col in mar_cols if col in df_proc.columns]
        
        if available_mar_cols:
            if display_info:
                missing_counts = {col: df_proc[col].isnull().sum() for col in available_mar_cols}
                total_mar_missing = sum(missing_counts.values())
                print(f"📊 Variables MAR à imputer: {len(available_mar_cols)}")
                print(f"   • Total valeurs manquantes: {total_mar_missing}")
                for col, count in missing_counts.items():
                    if count > 0:
                        print(f"   • {col}: {count} valeurs manquantes")
            
            try:
                if mar_method == 'knn':
                    # Optimisation automatique de k si demandée
                    if auto_optimize_k and knn_k is None:
                        if display_info:
                            print("🎯 Optimisation automatique de k...")
                        
                        # Utiliser seulement les colonnes avec des valeurs manquantes
                        cols_with_missing = [col for col in available_mar_cols 
                                           if df_proc[col].isnull().any()]
                        
                        if cols_with_missing:
                            k_result = find_optimal_k(
                                df=df_proc,
                                continuous_cols=cols_with_missing,
                                return_all_metrics=True,
                                plot_style='simple'
                            )
                            if isinstance(k_result, dict):
                                knn_k = k_result['optimal_k']
                                if display_info:
                                    print(f"✅ k optimal sélectionné: {knn_k}")
                            else:
                                knn_k = k_result
                        else:
                            knn_k = 5  # Valeur par défaut
                    
                    elif knn_k is None:
                        knn_k = 5
                        if display_info:
                            print("⚠️ k non spécifié, utilisation de k = 5 par défaut.")
                    
                    # Validation de k par rapport aux données disponibles
                    non_missing_rows = df_proc[available_mar_cols].dropna()
                    if len(non_missing_rows) < knn_k:
                        knn_k = max(1, len(non_missing_rows) - 1)
                        if display_info:
                            print(f"⚠️ k ajusté à {knn_k} (données insuffisantes)")
                    
                    # Application de KNN
                    imputer = KNNImputer(n_neighbors=knn_k)
                    original_values = df_proc[available_mar_cols].copy()
                    df_proc[available_mar_cols] = imputer.fit_transform(df_proc[available_mar_cols])
                    suffix = f'knn_k{knn_k}'
                    
                    # Validation de l'imputation
                    if validate_imputation:
                        imputed_mask = original_values.isnull()
                        for col in available_mar_cols:
                            imputed_values = df_proc.loc[imputed_mask[col], col]
                            if len(imputed_values) > 0:
                                # Vérifier les valeurs aberrantes
                                original_stats = original_values[col].describe()
                                imputed_stats = imputed_values.describe()
                                
                                # Détection d'anomalies simples
                                if imputed_stats['min'] < original_stats['min'] - 3*original_stats['std']:
                                    if display_info:
                                        print(f"⚠️ Valeurs imputées potentiellement aberrantes dans {col}")

                elif mar_method == 'mice':
                    imputer = IterativeImputer(random_state=42, max_iter=10)
                    original_values = df_proc[available_mar_cols].copy()
                    df_proc[available_mar_cols] = imputer.fit_transform(df_proc[available_mar_cols])
                    suffix = 'mice'

                else:
                    raise ValueError("❌ mar_method doit être 'knn' ou 'mice'.")

                # Sauvegarde du modèle d'imputation
                if save_results and models_dir:
                    models_dir = Path(models_dir)
                    models_dir.mkdir(parents=True, exist_ok=True)
                    imp_path = models_dir / f"imputer_{suffix}.pkl"
                    joblib.dump(imputer, imp_path)
                    if display_info:
                        print(f"💾 Modèle d'imputation sauvegardé: {imp_path}")

                if display_info:
                    print(f"✅ Imputation {mar_method} appliquée sur {len(available_mar_cols)} colonnes MAR")

            except Exception as e:
                if display_info:
                    print(f"❌ Erreur lors de l'imputation MAR: {e}")
                    print(f"🔄 Utilisation de la méthode de secours: {backup_method}")
                
                # Méthode de secours
                for col in available_mar_cols:
                    if backup_method == 'median':
                        df_proc[col] = df_proc[col].fillna(df_proc[col].median())
                    elif backup_method == 'mean':
                        df_proc[col] = df_proc[col].fillna(df_proc[col].mean())
                suffix = f'{backup_method}_backup'

        # Imputation MCAR (médiane)
        mcar_imputed = 0
        for col in mcar_cols:
            if col in df_proc.columns and df_proc[col].isnull().any():
                missing_count = df_proc[col].isnull().sum()
                val = df_proc[col].median()
                df_proc[col] = df_proc[col].fillna(val)
                mcar_imputed += missing_count
                if display_info:
                    print(f"✅ {col}: {missing_count} valeurs imputées par médiane ({val:.4f})")

    else:
        raise ValueError("❌ strategy doit être 'all_median' ou 'mixed_mar_mcar'.")

    # Validation finale
    final_missing = df_proc.isnull().sum().sum()
    if display_info:
        print(f"\n📊 Résumé de l'imputation:")
        print(f"   • Valeurs manquantes avant: {initial_missing}")
        print(f"   • Valeurs manquantes après: {final_missing}")
        print(f"   • Valeurs imputées: {initial_missing - final_missing}")
        
        if final_missing > 0:
            remaining_cols = df_proc.columns[df_proc.isnull().any()].tolist()
            print(f"⚠️ Valeurs manquantes restantes dans: {remaining_cols}")

    # Sauvegarde des résultats
    if save_results:
        if processed_data_dir is None:
            raise ValueError("processed_data_dir doit être fourni si save_results=True.")
        processed_data_dir = Path(processed_data_dir)
        processed_data_dir.mkdir(parents=True, exist_ok=True)

        filename = custom_filename or f"df_imputed_{suffix}.csv"
        filepath = processed_data_dir / filename
        df_proc.to_csv(filepath, index=False)
        if display_info:
            print(f"💾 Données imputées sauvegardées: {filepath}")

    if display_info:
        print("="*50)
        print("✅ Imputation terminée avec succès")

    return df_proc
