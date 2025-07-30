# =============================================================================
# UTILISATION DU MODULE D'OPTIMISATION DES SEUILS AVEC MODÈLES EXISTANTS
# =============================================================================

# Import du module d'optimisation
from modules.modeling.stacking_threshold_optimizer import (
    optimize_stacking_thresholds_with_models, 
    plot_threshold_optimization, 
    save_optimization_results,
    get_optimal_predictions
)

# =============================================================================
# ÉTAPE 1 : OPTIMISATION DES SEUILS AVEC MODÈLES EXISTANTS
# =============================================================================

print("🎯 ÉTAPE 1 : Optimisation des seuils avec modèles existants")
print("=" * 70)

# Utiliser les modèles déjà créés (stacking_knn et stacking_mice)
# Optimiser les seuils pour KNN et MICE
results = optimize_stacking_thresholds_with_models(
    stacking_knn=stacking_knn,  # Modèle déjà créé
    stacking_mice=stacking_mice, # Modèle déjà créé
    X_train_knn=X_train_knn, 
    X_val_knn=X_val_knn, 
    y_train_knn=y_train_knn, 
    y_val_knn=y_val_knn,
    X_train_mice=X_train_mice, 
    X_val_mice=X_val_mice, 
    y_train_mice=y_train_mice, 
    y_val_mice=y_val_mice,
    optimization_method="f1",  # Optimiser sur F1-score
    verbose=True
)

# =============================================================================
# ÉTAPE 2 : VISUALISATION DES RÉSULTATS
# =============================================================================

print("\n🎯 ÉTAPE 2 : Visualisation des résultats")
print("=" * 70)

# Créer les visualisations (graphiques plus petits maintenant)
plot_threshold_optimization(
    results, 
    y_val_knn=y_val_knn,
    y_val_mice=y_val_mice,
    save_path="outputs/figures/figures_notebook3/stacking_threshold_optimization.png"
)

# =============================================================================
# ÉTAPE 3 : SAUVEGARDE DES RÉSULTATS
# =============================================================================

print("\n🎯 ÉTAPE 3 : Sauvegarde des résultats")
print("=" * 70)

# Sauvegarder les résultats
json_file, csv_file = save_optimization_results(results)

# =============================================================================
# ÉTAPE 4 : UTILISATION DES SEUILS OPTIMAUX
# =============================================================================

print("\n🎯 ÉTAPE 4 : Utilisation des seuils optimaux")
print("=" * 70)

# Générer les prédictions avec seuils optimaux
predictions = get_optimal_predictions(
    results, 
    X_test_knn, 
    X_test_mice, 
    stacking_knn, 
    stacking_mice
)

# =============================================================================
# ÉTAPE 5 : ÉVALUATION DES PERFORMANCES
# =============================================================================

print("\n🎯 ÉTAPE 5 : Évaluation des performances")
print("=" * 70)

from sklearn.metrics import f1_score, classification_report

# Évaluer les performances avec seuils optimaux
for method, pred_data in predictions.items():
    print(f"\n📊 PERFORMANCES {method.upper()} AVEC SEUIL OPTIMAL:")
    print(f"   Seuil optimal: {pred_data['threshold']:.3f}")
    
    # Calculer F1-score (si y_test disponible)
    if f'y_test_{method.lower()}' in globals():
        y_test = globals()[f'y_test_{method.lower()}']
        f1_optimal = f1_score(y_test, pred_data['predictions'])
        print(f"   F1-score: {f1_optimal:.4f}")
        
        print(f"\n   📋 Rapport de classification:")
        print(classification_report(y_test, pred_data['predictions']))

# =============================================================================
# RÉSUMÉ FINAL
# =============================================================================

print("\n" + "=" * 70)
print("🎯 RÉSUMÉ DE L'OPTIMISATION DES SEUILS")
print("=" * 70)

print("\n📊 SEUILS OPTIMAUX TROUVÉS:")
for method, result in results.items():
    print(f"   🔹 {method}: {result['optimal_threshold']:.3f}")
    print(f"      F1 par défaut: {result['default_f1']:.4f}")
    print(f"      F1 optimisé: {result['optimized_f1']:.4f}")
    print(f"      Amélioration: {result['improvement']:+.4f} ({result['improvement_pct']:+.1f}%)")

# Identifier le meilleur modèle
best_method = max(results.keys(), key=lambda x: results[x]['optimized_f1'])
best_result = results[best_method]

print(f"\n🏆 MEILLEUR MODÈLE:")
print(f"   {best_method.upper()} - F1: {best_result['optimized_f1']:.4f}")
print(f"   Seuil optimal: {best_result['optimal_threshold']:.3f}")

print("\n✅ OPTIMISATION TERMINÉE !")
print("=" * 70)

print("\n📋 Variables disponibles :")
print("   - results : Résultats d'optimisation")
print("   - predictions : Prédictions avec seuils optimaux")
print("   - stacking_knn, stacking_mice : Modèles entraînés")

print("\n🎯 Prochaines étapes :")
print("   1. Utiliser les prédictions optimales pour la soumission")
print("   2. Sauvegarder les modèles avec leurs seuils optimaux")
print("   3. Comparer avec d'autres méthodes") 