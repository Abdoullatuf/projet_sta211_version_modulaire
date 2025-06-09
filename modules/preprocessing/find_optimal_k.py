#modules/preprocessing/find_optimal_k.py


import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import warnings
from typing import List, Tuple, Optional, Dict
import time

def find_optimal_k(
    df: pd.DataFrame,
    columns_to_impute: List[str],
    k_range: range = range(3, 21, 2),
    cv_folds: int = 5,
    test_size: float = 0.2,
    random_state: int = 42,
    metric: str = 'mse',
    plot_results: bool = True,
    verbose: bool = True,
    figsize: Tuple[int, int] = (10, 6)
) -> Dict:
    """
    Trouve la valeur optimale de K pour l'imputation KNN par validation croisée.
    
    Méthodologie :
    1. Masque artificiellement des valeurs connues
    2. Teste différentes valeurs de K
    3. Évalue la qualité de reconstruction via validation croisée
    4. Sélectionne le K avec le meilleur score moyen
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset contenant les données à imputer
    columns_to_impute : list
        Liste des colonnes à considérer pour l'optimisation
    k_range : range
        Plage des valeurs K à tester (défaut: 3 à 19 par pas de 2)
    cv_folds : int
        Nombre de plis pour la validation croisée
    test_size : float
        Proportion de valeurs à masquer pour le test
    random_state : int
        Graine aléatoire pour la reproductibilité
    metric : str
        Métrique d'évaluation ('mse', 'mae', 'rmse')
    plot_results : bool
        Affichage du graphique des résultats
    verbose : bool
        Affichage détaillé du processus
    figsize : tuple
        Taille de la figure (largeur, hauteur)
        
    Returns:
    --------
    dict : Dictionnaire contenant :
        - 'optimal_k' : Valeur K optimale
        - 'best_score' : Meilleur score obtenu
        - 'scores_mean' : Scores moyens par K
        - 'scores_std' : Écarts-types par K
        - 'detailed_results' : Résultats détaillés par fold
        - 'computation_time' : Temps de calcul total
    """
    
    start_time = time.time()
    
    if verbose:
        print("🔍 Recherche de la valeur optimale K pour l'imputation KNN")
        print("=" * 60)
        print(f"📊 Colonnes à évaluer    : {columns_to_impute}")
        print(f"🎯 Plage K à tester      : {list(k_range)}")
        print(f"🔄 Validation croisée    : {cv_folds} folds")
        print(f"📏 Métrique d'évaluation : {metric.upper()}")
        print(f"🎲 Random state          : {random_state}")
        print("-" * 60)
    
    # Validation des paramètres
    if not columns_to_impute:
        raise ValueError("❌ La liste columns_to_impute ne peut pas être vide")
    
    for col in columns_to_impute:
        if col not in df.columns:
            raise ValueError(f"❌ Colonne '{col}' introuvable dans le DataFrame")
    
    # Préparation des données
    df_work = df[columns_to_impute].copy()
    
    # Vérification des valeurs manquantes
    missing_info = df_work.isnull().sum()
    if verbose:
        print(f"📋 Valeurs manquantes par colonne :")
        for col, missing in missing_info.items():
            pct = (missing / len(df_work)) * 100
            print(f"   • {col:<15} : {missing:>4} ({pct:>5.1f}%)")
    
    # Ne garder que les lignes complètes pour le test
    df_complete = df_work.dropna()
    
    if len(df_complete) < 50:
        warnings.warn(f"⚠️ Seulement {len(df_complete)} lignes complètes disponibles")
    
    if verbose:
        print(f"\n🧮 Données disponibles   : {len(df_complete):,} lignes complètes")
        print(f"📊 Taille échantillon test: {int(len(df_complete) * test_size):,}")
    
    # Stockage des résultats
    results = {
        'k_values': list(k_range),
        'scores_mean': [],
        'scores_std': [],
        'detailed_results': {}
    }
    
    # Configuration de la validation croisée
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    # Test de chaque valeur K
    for k in k_range:
        if verbose:
            print(f"\n🔄 Test K={k:2d} ", end="")
        
        fold_scores = []
        results['detailed_results'][k] = []
        
        # Validation croisée
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(df_complete)):
            # Division train/test
            train_data = df_complete.iloc[train_idx].copy()
            test_data = df_complete.iloc[test_idx].copy()
            
            # Masquage artificiel de valeurs dans le set de test
            n_to_mask = max(1, int(len(test_data) * test_size))
            
            test_data_masked = test_data.copy()
            true_values = []
            
            # Masquage aléatoire pour chaque colonne
            np.random.seed(random_state + fold_idx)
            for col in columns_to_impute:
                if len(test_data[col]) > 0:
                    mask_indices = np.random.choice(
                        test_data.index, 
                        size=min(n_to_mask, len(test_data)), 
                        replace=False
                    )
                    true_values.extend(test_data.loc[mask_indices, col].values)
                    test_data_masked.loc[mask_indices, col] = np.nan
            
            # Combinaison train + test masqué
            combined_data = pd.concat([train_data, test_data_masked], ignore_index=True)
            
            # Imputation KNN
            try:
                imputer = KNNImputer(n_neighbors=k)
                imputed_data = imputer.fit_transform(combined_data)
                imputed_df = pd.DataFrame(imputed_data, columns=combined_data.columns)
                
                # Récupération des valeurs imputées
                test_start_idx = len(train_data)
                imputed_test = imputed_df.iloc[test_start_idx:].reset_index(drop=True)
                
                # Extraction des valeurs prédites
                predicted_values = []
                val_idx = 0
                for col in columns_to_impute:
                    if len(test_data[col]) > 0:
                        mask_indices = np.random.choice(
                            range(len(test_data)), 
                            size=min(n_to_mask, len(test_data)), 
                            replace=False
                        )
                        for idx in mask_indices:
                            if val_idx < len(true_values):
                                predicted_values.append(imputed_test.iloc[idx][col])
                                val_idx += 1
                
                # Calcul de la métrique
                if len(true_values) > 0 and len(predicted_values) > 0:
                    if metric.lower() == 'mse':
                        score = mean_squared_error(true_values[:len(predicted_values)], predicted_values)
                    elif metric.lower() == 'rmse':
                        score = np.sqrt(mean_squared_error(true_values[:len(predicted_values)], predicted_values))
                    elif metric.lower() == 'mae':
                        score = np.mean(np.abs(np.array(true_values[:len(predicted_values)]) - np.array(predicted_values)))
                    else:
                        score = mean_squared_error(true_values[:len(predicted_values)], predicted_values)
                    
                    fold_scores.append(score)
                    results['detailed_results'][k].append(score)
                
            except Exception as e:
                if verbose:
                    print(f"❌ Erreur K={k}, fold {fold_idx}: {e}")
                continue
            
            if verbose:
                print(".", end="")
        
        # Calcul des statistiques pour ce K
        if fold_scores:
            mean_score = np.mean(fold_scores)
            std_score = np.std(fold_scores)
            results['scores_mean'].append(mean_score)
            results['scores_std'].append(std_score)
            
            if verbose:
                print(f" → {metric.upper()}: {mean_score:.4f} (±{std_score:.4f})")
        else:
            results['scores_mean'].append(np.inf)
            results['scores_std'].append(0)
            if verbose:
                print(f" → ❌ Échec")
    
    # Identification du K optimal
    valid_scores = [(i, score) for i, score in enumerate(results['scores_mean']) if not np.isinf(score)]
    
    if not valid_scores:
        raise RuntimeError("❌ Aucune valeur K n'a produit de résultats valides")
    
    best_idx, best_score = min(valid_scores, key=lambda x: x[1])
    optimal_k = results['k_values'][best_idx]
    
    computation_time = time.time() - start_time
    
    # Affichage des résultats
    if verbose:
        print("\n" + "=" * 60)
        print("🎯 RÉSULTATS DE L'OPTIMISATION")
        print("=" * 60)
        print(f"🏆 K optimal             : {optimal_k}")
        print(f"📊 Meilleur score        : {best_score:.4f}")
        print(f"⏱️  Temps de calcul       : {computation_time:.2f}s")
        
        # Top 3 des meilleurs K
        sorted_results = sorted(valid_scores, key=lambda x: x[1])[:3]
        print(f"\n🥇 Top 3 des valeurs K :")
        for rank, (idx, score) in enumerate(sorted_results, 1):
            k_val = results['k_values'][idx]
            std_val = results['scores_std'][idx]
            print(f"   {rank}. K={k_val:2d} → {metric.upper()}: {score:.4f} (±{std_val:.4f})")
    
    # Visualisation
    if plot_results and len(valid_scores) > 1:
        fig, ax = plt.subplots(figsize=figsize)
        
        k_vals = [results['k_values'][i] for i, _ in valid_scores]
        scores = [score for _, score in valid_scores]
        stds = [results['scores_std'][i] for i, _ in valid_scores]
        
        ax.errorbar(k_vals, scores, yerr=stds, marker='o', capsize=5, capthick=2)
        ax.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7, 
                  label=f'K optimal = {optimal_k}')
        ax.scatter([optimal_k], [best_score], color='red', s=100, zorder=5)
        
        ax.set_xlabel('Nombre de voisins (K)', fontsize=12)
        ax.set_ylabel(f'{metric.upper()} Score', fontsize=12)
        ax.set_title(f'Optimisation K pour imputation KNN\n(Validation croisée {cv_folds} folds)', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.show()
    
    # Retour des résultats
    return {
        'optimal_k': optimal_k,
        'best_score': best_score,
        'scores_mean': results['scores_mean'],
        'scores_std': results['scores_std'],
        'k_values': results['k_values'],
        'detailed_results': results['detailed_results'],
        'computation_time': computation_time,
        'method': f'CV-{cv_folds}folds',
        'metric': metric,
        'n_complete_samples': len(df_complete)
    }
