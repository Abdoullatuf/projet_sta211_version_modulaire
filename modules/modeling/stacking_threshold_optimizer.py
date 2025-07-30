# =============================================================================
# MODULE : OPTIMISATION DES SEUILS POUR MODÈLES DE STACKING
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import logging
from modules.config.env_setup import init_project 
init_result = init_project()
paths = init_result["paths"]

# Import des modules du projet
from .stacking_models_creator import create_stacking_models

log = logging.getLogger(__name__)

def optimize_stacking_thresholds(X_train_knn, X_val_knn, y_train_knn, y_val_knn,
                                X_train_mice, X_val_mice, y_train_mice, y_val_mice,
                                optimization_method="f1", verbose=True):
    """
    Optimise les seuils pour les modèles de stacking KNN et MICE.
    
    Parameters:
    -----------
    X_train_knn, X_val_knn, y_train_knn, y_val_knn : données KNN
    X_train_mice, X_val_mice, y_train_mice, y_val_mice : données MICE
    optimization_method : str, "f1", "precision", ou "recall"
    verbose : bool, affichage détaillé
    
    Returns:
    --------
    dict : Résultats d'optimisation avec seuils optimaux
    """
    
    if verbose:
        print("🎯 OPTIMISATION DES SEUILS POUR MODÈLES DE STACKING")
        print("=" * 70)
    
    # Créer les modèles de stacking
    if verbose:
        print("\n📊 Création des modèles de stacking...")
    
    models = create_stacking_models(imputation_method="both", verbose=False)
    stacking_knn = models['stacking_classifier_knn']
    stacking_mice = models['stacking_classifier_mice']
    
    # Entraîner les modèles
    if verbose:
        print("\n🔄 Entraînement des modèles...")
    
    stacking_knn.fit(X_train_knn, y_train_knn)
    stacking_mice.fit(X_train_mice, y_train_mice)
    
    # Optimiser les seuils
    results = {}
    
    # Optimisation pour KNN
    if verbose:
        print("\n📊 Optimisation seuil KNN...")
    
    y_proba_knn = stacking_knn.predict_proba(X_val_knn)[:, 1]
    optimal_threshold_knn, f1_knn_optimal, f1_knn_default = _optimize_threshold(
        y_proba_knn, y_val_knn, optimization_method, verbose
    )
    
    results['KNN'] = {
        'optimal_threshold': optimal_threshold_knn,
        'default_f1': f1_knn_default,
        'optimized_f1': f1_knn_optimal,
        'improvement': f1_knn_optimal - f1_knn_default,
        'improvement_pct': ((f1_knn_optimal - f1_knn_default) / f1_knn_default * 100) if f1_knn_default > 0 else 0,
        'probabilities': y_proba_knn,
        'predictions_default': (y_proba_knn >= 0.5).astype(int),
        'predictions_optimal': (y_proba_knn >= optimal_threshold_knn).astype(int)
    }
    
    # Optimisation pour MICE
    if verbose:
        print("\n📊 Optimisation seuil MICE...")
    
    y_proba_mice = stacking_mice.predict_proba(X_val_mice)[:, 1]
    optimal_threshold_mice, f1_mice_optimal, f1_mice_default = _optimize_threshold(
        y_proba_mice, y_val_mice, optimization_method, verbose
    )
    
    results['MICE'] = {
        'optimal_threshold': optimal_threshold_mice,
        'default_f1': f1_mice_default,
        'optimized_f1': f1_mice_optimal,
        'improvement': f1_mice_optimal - f1_mice_default,
        'improvement_pct': ((f1_mice_optimal - f1_mice_default) / f1_mice_default * 100) if f1_mice_default > 0 else 0,
        'probabilities': y_proba_mice,
        'predictions_default': (y_proba_mice >= 0.5).astype(int),
        'predictions_optimal': (y_proba_mice >= optimal_threshold_mice).astype(int)
    }
    
    # Résumé des résultats
    if verbose:
        _print_optimization_summary(results)
    
    return results

def optimize_stacking_thresholds_with_models(stacking_knn, stacking_mice,
                                           X_train_knn, X_val_knn, y_train_knn, y_val_knn,
                                           X_train_mice, X_val_mice, y_train_mice, y_val_mice,
                                           optimization_method="f1", verbose=True):
    """
    Optimise les seuils pour les modèles de stacking KNN et MICE déjà créés.
    
    Parameters:
    -----------
    stacking_knn, stacking_mice : modèles de stacking déjà créés
    X_train_knn, X_val_knn, y_train_knn, y_val_knn : données KNN
    X_train_mice, X_val_mice, y_train_mice, y_val_mice : données MICE
    optimization_method : str, "f1", "precision", ou "recall"
    verbose : bool, affichage détaillé
    
    Returns:
    --------
    dict : Résultats d'optimisation avec seuils optimaux
    """
    
    if verbose:
        print("🎯 OPTIMISATION DES SEUILS POUR MODÈLES DE STACKING (MODÈLES EXISTANTS)")
        print("=" * 80)
    
    # Entraîner les modèles (si pas déjà fait)
    if verbose:
        print("\n🔄 Entraînement des modèles...")
    
    stacking_knn.fit(X_train_knn, y_train_knn)
    stacking_mice.fit(X_train_mice, y_train_mice)
    
    # Optimiser les seuils
    results = {}
    
    # Optimisation pour KNN
    if verbose:
        print("\n📊 Optimisation seuil KNN...")
    
    y_proba_knn = stacking_knn.predict_proba(X_val_knn)[:, 1]
    optimal_threshold_knn, f1_knn_optimal, f1_knn_default = _optimize_threshold(
        y_proba_knn, y_val_knn, optimization_method, verbose
    )
    
    results['KNN'] = {
        'optimal_threshold': optimal_threshold_knn,
        'default_f1': f1_knn_default,
        'optimized_f1': f1_knn_optimal,
        'improvement': f1_knn_optimal - f1_knn_default,
        'improvement_pct': ((f1_knn_optimal - f1_knn_default) / f1_knn_default * 100) if f1_knn_default > 0 else 0,
        'probabilities': y_proba_knn,
        'predictions_default': (y_proba_knn >= 0.5).astype(int),
        'predictions_optimal': (y_proba_knn >= optimal_threshold_knn).astype(int)
    }
    
    # Optimisation pour MICE
    if verbose:
        print("\n📊 Optimisation seuil MICE...")
    
    y_proba_mice = stacking_mice.predict_proba(X_val_mice)[:, 1]
    optimal_threshold_mice, f1_mice_optimal, f1_mice_default = _optimize_threshold(
        y_proba_mice, y_val_mice, optimization_method, verbose
    )
    
    results['MICE'] = {
        'optimal_threshold': optimal_threshold_mice,
        'default_f1': f1_mice_default,
        'optimized_f1': f1_mice_optimal,
        'improvement': f1_mice_optimal - f1_mice_default,
        'improvement_pct': ((f1_mice_optimal - f1_mice_default) / f1_mice_default * 100) if f1_mice_default > 0 else 0,
        'probabilities': y_proba_mice,
        'predictions_default': (y_proba_mice >= 0.5).astype(int),
        'predictions_optimal': (y_proba_mice >= optimal_threshold_mice).astype(int)
    }
    
    # Résumé des résultats
    if verbose:
        _print_optimization_summary(results)
    
    return results

def optimize_stacking_thresholds_with_trained_models(stacking_knn, stacking_mice,
                                                   X_val_knn, y_val_knn,
                                                   X_val_mice, y_val_mice,
                                                   optimization_method="f1", verbose=True):
    """
    Optimise les seuils pour les modèles de stacking KNN et MICE déjà entraînés.
    
    Parameters:
    -----------
    stacking_knn, stacking_mice : modèles de stacking déjà entraînés
    X_val_knn, y_val_knn : données de validation KNN
    X_val_mice, y_val_mice : données de validation MICE
    optimization_method : str, "f1", "precision", ou "recall"
    verbose : bool, affichage détaillé
    
    Returns:
    --------
    dict : Résultats d'optimisation avec seuils optimaux
    """
    
    if verbose:
        print("🎯 OPTIMISATION DES SEUILS POUR MODÈLES DE STACKING (MODÈLES DÉJÀ ENTRÂINÉS)")
        print("=" * 80)
    
    # Optimiser les seuils (sans réentraîner les modèles)
    results = {}
    
    # Optimisation pour KNN
    if verbose:
        print("\n📊 Optimisation seuil KNN...")
    
    y_proba_knn = stacking_knn.predict_proba(X_val_knn)[:, 1]
    optimal_threshold_knn, f1_knn_optimal, f1_knn_default = _optimize_threshold(
        y_proba_knn, y_val_knn, optimization_method, verbose
    )
    
    results['KNN'] = {
        'optimal_threshold': optimal_threshold_knn,
        'default_f1': f1_knn_default,
        'optimized_f1': f1_knn_optimal,
        'improvement': f1_knn_optimal - f1_knn_default,
        'improvement_pct': ((f1_knn_optimal - f1_knn_default) / f1_knn_default * 100) if f1_knn_default > 0 else 0,
        'probabilities': y_proba_knn,
        'predictions_default': (y_proba_knn >= 0.5).astype(int),
        'predictions_optimal': (y_proba_knn >= optimal_threshold_knn).astype(int)
    }
    
    # Optimisation pour MICE
    if verbose:
        print("\n📊 Optimisation seuil MICE...")
    
    y_proba_mice = stacking_mice.predict_proba(X_val_mice)[:, 1]
    optimal_threshold_mice, f1_mice_optimal, f1_mice_default = _optimize_threshold(
        y_proba_mice, y_val_mice, optimization_method, verbose
    )
    
    results['MICE'] = {
        'optimal_threshold': optimal_threshold_mice,
        'default_f1': f1_mice_default,
        'optimized_f1': f1_mice_optimal,
        'improvement': f1_mice_optimal - f1_mice_default,
        'improvement_pct': ((f1_mice_optimal - f1_mice_default) / f1_mice_default * 100) if f1_mice_default > 0 else 0,
        'probabilities': y_proba_mice,
        'predictions_default': (y_proba_mice >= 0.5).astype(int),
        'predictions_optimal': (y_proba_mice >= optimal_threshold_mice).astype(int)
    }
    
    # Résumé des résultats
    if verbose:
        _print_optimization_summary(results)
    
    return results

def _optimize_threshold(y_proba, y_true, optimization_method="f1", verbose=True):
    """
    Optimise le seuil de décision pour un ensemble de probabilités.
    
    Parameters:
    -----------
    y_proba : array, probabilités prédites
    y_true : array, vraies étiquettes
    optimization_method : str, métrique à optimiser
    verbose : bool, affichage détaillé
    
    Returns:
    --------
    tuple : (seuil_optimal, f1_optimal, f1_default)
    """
    
    # Seuil par défaut
    y_pred_default = (y_proba >= 0.5).astype(int)
    f1_default = f1_score(y_true, y_pred_default)
    
    # Test de différents seuils
    thresholds = np.linspace(0.01, 0.99, 99)
    scores = []
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        
        if optimization_method == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        elif optimization_method == "precision":
            from sklearn.metrics import precision_score
            score = precision_score(y_true, y_pred, zero_division=0)
        elif optimization_method == "recall":
            from sklearn.metrics import recall_score
            score = recall_score(y_true, y_pred, zero_division=0)
        else:
            score = f1_score(y_true, y_pred, zero_division=0)
        
        scores.append(score)
    
    # Trouver le meilleur seuil
    best_idx = np.argmax(scores)
    optimal_threshold = thresholds[best_idx]
    f1_optimal = scores[best_idx]
    
    if verbose:
        print(f"   Seuil par défaut (0.5): F1 = {f1_default:.4f}")
        print(f"   Seuil optimal ({optimal_threshold:.3f}): F1 = {f1_optimal:.4f}")
        improvement = f1_optimal - f1_default
        if improvement > 0:
            print(f"   🚀 Amélioration: +{improvement:.4f}")
        else:
            print(f"   📊 Variation: {improvement:+.4f}")
    
    return optimal_threshold, f1_optimal, f1_default

def _print_optimization_summary(results):
    """Affiche un résumé des résultats d'optimisation."""
    
    print("\n" + "=" * 70)
    print("📊 RÉSUMÉ DE L'OPTIMISATION DES SEUILS")
    print("=" * 70)
    
    for method, result in results.items():
        print(f"\n🔹 {method.upper()}:")
        print(f"   Seuil par défaut (0.5): F1 = {result['default_f1']:.4f}")
        print(f"   Seuil optimal ({result['optimal_threshold']:.3f}): F1 = {result['optimized_f1']:.4f}")
        print(f"   Amélioration: {result['improvement']:+.4f} ({result['improvement_pct']:+.1f}%)")
    
    # Identifier le meilleur modèle
    best_method = max(results.keys(), key=lambda x: results[x]['optimized_f1'])
    best_result = results[best_method]
    
    print(f"\n🏆 MEILLEUR MODÈLE:")
    print(f"   {best_method.upper()} - F1: {best_result['optimized_f1']:.4f}")
    print(f"   Seuil optimal: {best_result['optimal_threshold']:.3f}")

def plot_threshold_optimization(results, y_val_knn=None, y_val_mice=None, save_path=None):
    """
    Visualise les résultats d'optimisation des seuils.
    
    Parameters:
    -----------
    results : dict, résultats d'optimisation
    y_val_knn : array, étiquettes de validation KNN (optionnel)
    y_val_mice : array, étiquettes de validation MICE (optionnel)
    save_path : str, chemin de sauvegarde (optionnel)
    """
    
    # Debug: vérifier les paramètres reçus
    print(f"🔍 Debug - y_val_knn type: {type(y_val_knn)}, length: {len(y_val_knn) if y_val_knn is not None else 'None'}")
    print(f"🔍 Debug - y_val_mice type: {type(y_val_mice)}, length: {len(y_val_mice) if y_val_mice is not None else 'None'}")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('🎯 Optimisation des Seuils - Modèles de Stacking', fontsize=16, fontweight='bold')
    
    # 1. Comparaison F1-score avant/après
    ax1 = axes[0, 0]
    methods = list(results.keys())
    default_f1s = [results[m]['default_f1'] for m in methods]
    optimized_f1s = [results[m]['optimized_f1'] for m in methods]
    
    x = np.arange(len(methods))
    width = 0.35
    
    ax1.bar(x - width/2, default_f1s, width, label='Seuil 0.5', alpha=0.7, color='lightcoral')
    ax1.bar(x + width/2, optimized_f1s, width, label='Seuil optimal', alpha=0.7, color='lightgreen')
    
    ax1.set_title('📊 F1-score: Avant vs Après optimisation')
    ax1.set_xlabel('Méthode d\'imputation')
    ax1.set_ylabel('F1-score')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Seuils optimaux
    ax2 = axes[0, 1]
    thresholds = [results[m]['optimal_threshold'] for m in methods]
    colors = ['skyblue' if t < 0.5 else 'orange' for t in thresholds]
    
    bars = ax2.bar(methods, thresholds, color=colors, alpha=0.7)
    ax2.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='Seuil par défaut (0.5)')
    ax2.set_title('📈 Seuils optimaux')
    ax2.set_ylabel('Seuil optimal')
    ax2.legend()
    
    # Ajouter les valeurs sur les barres
    for bar, threshold in zip(bars, thresholds):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{threshold:.3f}', ha='center', va='bottom')
    
    # 3. Améliorations
    ax3 = axes[1, 0]
    improvements = [results[m]['improvement'] for m in methods]
    colors = ['green' if imp > 0 else 'red' if imp < 0 else 'gray' for imp in improvements]
    
    bars = ax3.bar(methods, improvements, color=colors, alpha=0.7)
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.7)
    ax3.set_title('🚀 Améliorations F1-score')
    ax3.set_ylabel('Amélioration')
    
    # Ajouter les valeurs sur les barres
    for bar, imp in zip(bars, improvements):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + (0.001 if imp >= 0 else -0.001),
                f'{imp:+.4f}', ha='center', va='bottom' if imp >= 0 else 'top')
    
    # 4. Courbes ROC (seulement si les étiquettes sont disponibles)
    ax4 = axes[1, 1]
    if y_val_knn is not None and y_val_mice is not None:
        print("✅ Courbes ROC disponibles - création en cours...")
        for method, result in results.items():
            y_val = y_val_knn if method == 'KNN' else y_val_mice
            fpr, tpr, _ = roc_curve(y_val, result['probabilities'])
            auc = np.trapz(tpr, fpr)
            ax4.plot(fpr, tpr, label=f'{method} (AUC = {auc:.3f})')
        
        ax4.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax4.set_title('📊 Courbes ROC')
        ax4.set_xlabel('Taux de faux positifs')
        ax4.set_ylabel('Taux de vrais positifs')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        # Si les étiquettes ne sont pas disponibles, afficher un message
        print("❌ Courbes ROC non disponibles - étiquettes manquantes")
        ax4.text(0.5, 0.5, 'Courbes ROC non disponibles\n(étiquettes manquantes)', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.set_title('📊 Courbes ROC')
        ax4.set_xlabel('Taux de faux positifs')
        ax4.set_ylabel('Taux de vrais positifs')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Graphique sauvegardé: {save_path}")
    
    plt.show()

def save_optimization_results(results, save_dir=paths["OUTPUTS_DIR"]/ "modeling/thresholds"):
    """
    Sauvegarde les résultats d'optimisation.
    
    Parameters:
    -----------
    results : dict, résultats d'optimisation
    save_dir : str, répertoire de sauvegarde
    """
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Sauvegarde JSON
    json_file = save_dir / "stacking_threshold_optimization.json"
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Sauvegarde CSV
    csv_data = []
    for method, result in results.items():
        csv_data.append({
            'method': method,
            'optimal_threshold': result['optimal_threshold'],
            'default_f1': result['default_f1'],
            'optimized_f1': result['optimized_f1'],
            'improvement': result['improvement'],
            'improvement_pct': result['improvement_pct']
        })
    
    df = pd.DataFrame(csv_data)
    csv_file = save_dir / "stacking_threshold_optimization.csv"
    df.to_csv(csv_file, index=False)
    
    print(f"\n💾 Résultats sauvegardés:")
    print(f"   JSON: {json_file}")
    print(f"   CSV:  {csv_file}")
    
    return json_file, csv_file

def get_optimal_predictions(results, X_test_knn, X_test_mice, stacking_knn, stacking_mice):
    """
    Génère les prédictions avec les seuils optimaux.
    
    Parameters:
    -----------
    results : dict, résultats d'optimisation
    X_test_knn, X_test_mice : données de test
    stacking_knn, stacking_mice : modèles entraînés
    
    Returns:
    --------
    dict : Prédictions avec seuils optimaux
    """
    
    predictions = {}
    
    # Prédictions KNN avec seuil optimal
    if 'KNN' in results:
        y_proba_knn = stacking_knn.predict_proba(X_test_knn)[:, 1]
        optimal_threshold_knn = results['KNN']['optimal_threshold']
        y_pred_knn_optimal = (y_proba_knn >= optimal_threshold_knn).astype(int)
        
        predictions['KNN'] = {
            'probabilities': y_proba_knn,
            'predictions': y_pred_knn_optimal,
            'threshold': optimal_threshold_knn
        }
    
    # Prédictions MICE avec seuil optimal
    if 'MICE' in results:
        y_proba_mice = stacking_mice.predict_proba(X_test_mice)[:, 1]
        optimal_threshold_mice = results['MICE']['optimal_threshold']
        y_pred_mice_optimal = (y_proba_mice >= optimal_threshold_mice).astype(int)
        
        predictions['MICE'] = {
            'probabilities': y_proba_mice,
            'predictions': y_pred_mice_optimal,
            'threshold': optimal_threshold_mice
        }
    
    return predictions

def get_optimal_predictions_with_refit_models(results, X_test_knn, X_test_mice):
    """
    Génère les prédictions avec les seuils optimaux en utilisant les modèles with_refit.
    
    Parameters:
    -----------
    results : dict, résultats d'optimisation
    X_test_knn, X_test_mice : données de test
    
    Returns:
    --------
    dict : Prédictions avec seuils optimaux
    """
    
    # Import des modèles with_refit depuis le contexte global
    import sys
    import inspect
    
    # Récupérer le frame du notebook pour accéder aux variables globales
    frame = inspect.currentframe()
    while frame:
        if 'stacking_knn_with_refit' in frame.f_globals:
            stacking_knn_with_refit = frame.f_globals['stacking_knn_with_refit']
            stacking_mice_with_refit = frame.f_globals['stacking_mice_with_refit']
            break
        frame = frame.f_back
    else:
        raise ValueError("Les modèles stacking_knn_with_refit et stacking_mice_with_refit ne sont pas disponibles")
    
    predictions = {}
    
    # Prédictions KNN avec seuil optimal
    if 'KNN' in results:
        y_proba_knn = stacking_knn_with_refit.predict_proba(X_test_knn)[:, 1]
        optimal_threshold_knn = results['KNN']['optimal_threshold']
        y_pred_knn_optimal = (y_proba_knn >= optimal_threshold_knn).astype(int)
        
        predictions['KNN'] = {
            'probabilities': y_proba_knn,
            'predictions': y_pred_knn_optimal,
            'threshold': optimal_threshold_knn
        }
    
    # Prédictions MICE avec seuil optimal
    if 'MICE' in results:
        y_proba_mice = stacking_mice_with_refit.predict_proba(X_test_mice)[:, 1]
        optimal_threshold_mice = results['MICE']['optimal_threshold']
        y_pred_mice_optimal = (y_proba_mice >= optimal_threshold_mice).astype(int)
        
        predictions['MICE'] = {
            'probabilities': y_proba_mice,
            'predictions': y_pred_mice_optimal,
            'threshold': optimal_threshold_mice
        }
    
    return predictions

# =============================================================================
# EXEMPLE D'UTILISATION
# =============================================================================

if __name__ == "__main__":
    print("🔧 Test du module d'optimisation des seuils...")
    print("Ce module doit être importé dans un notebook avec les données disponibles.") 