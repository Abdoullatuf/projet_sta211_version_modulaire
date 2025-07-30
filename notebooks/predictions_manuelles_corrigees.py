# =============================================================================
# PRÉDICTIONS MANUELLES AVEC SEUILS OPTIMAUX (CORRIGÉES)
# =============================================================================

print("🎯 PRÉDICTIONS MANUELLES AVEC SEUILS OPTIMAUX (MODÈLES WITH_REFIT)")
print("=" * 70)

# =============================================================================
# ÉTAPE 1 : RÉCUPÉRATION DES SEUILS OPTIMAUX
# =============================================================================

print("\n🎯 ÉTAPE 1 : Récupération des seuils optimaux")
print("=" * 50)

# Seuils optimaux trouvés lors de l'optimisation
optimal_threshold_knn = 0.380
optimal_threshold_mice = 0.300

print(f"📊 Seuils optimaux:")
print(f"   KNN: {optimal_threshold_knn}")
print(f"   MICE: {optimal_threshold_mice}")

# =============================================================================
# ÉTAPE 2 : VÉRIFICATION DES MODÈLES CORRECTS
# =============================================================================

print("\n🎯 ÉTAPE 2 : Vérification des modèles")
print("=" * 50)

# Vérifier que les bons modèles sont disponibles
print("📋 Modèles utilisés:")
print(f"   stacking_knn_with_refit: {type(stacking_knn_with_refit).__name__ if 'stacking_knn_with_refit' in globals() else 'Non défini'}")
print(f"   stacking_mice_with_refit: {type(stacking_mice_with_refit).__name__ if 'stacking_mice_with_refit' in globals() else 'Non défini'}")

# =============================================================================
# ÉTAPE 3 : PRÉDICTIONS KNN (WITH_REFIT)
# =============================================================================

print("\n🎯 ÉTAPE 3 : Prédictions KNN (with_refit)")
print("=" * 50)

# Prédictions KNN sur données de test avec le bon modèle
print("🔧 Génération des prédictions KNN...")
y_proba_knn = stacking_knn_with_refit.predict_proba(X_test_knn)[:, 1]
predictions_knn = (y_proba_knn >= optimal_threshold_knn).astype(int)

print(f"📊 Résultats KNN:")
print(f"   Forme probabilités: {y_proba_knn.shape}")
print(f"   Forme prédictions: {predictions_knn.shape}")
print(f"   Seuil utilisé: {optimal_threshold_knn}")
print(f"   Nombre de prédictions positives: {predictions_knn.sum()}")
print(f"   Nombre de prédictions négatives: {(predictions_knn == 0).sum()}")

# =============================================================================
# ÉTAPE 4 : PRÉDICTIONS MICE (WITH_REFIT)
# =============================================================================

print("\n🎯 ÉTAPE 4 : Prédictions MICE (with_refit)")
print("=" * 50)

# Prédictions MICE sur données de test avec le bon modèle
print("🔧 Génération des prédictions MICE...")
y_proba_mice = stacking_mice_with_refit.predict_proba(X_test_mice)[:, 1]
predictions_mice = (y_proba_mice >= optimal_threshold_mice).astype(int)

print(f"📊 Résultats MICE:")
print(f"   Forme probabilités: {y_proba_mice.shape}")
print(f"   Forme prédictions: {predictions_mice.shape}")
print(f"   Seuil utilisé: {optimal_threshold_mice}")
print(f"   Nombre de prédictions positives: {predictions_mice.sum()}")
print(f"   Nombre de prédictions négatives: {(predictions_mice == 0).sum()}")

# =============================================================================
# ÉTAPE 5 : ÉVALUATION DES PERFORMANCES
# =============================================================================

print("\n🎯 ÉTAPE 5 : Évaluation des performances")
print("=" * 50)

from sklearn.metrics import f1_score, classification_report, accuracy_score

# Évaluation KNN (si y_test_knn disponible)
if 'y_test_knn' in globals() and y_test_knn is not None:
    print("📊 PERFORMANCES KNN (WITH_REFIT):")
    f1_knn = f1_score(y_test_knn, predictions_knn)
    accuracy_knn = accuracy_score(y_test_knn, predictions_knn)
    
    print(f"   F1-score: {f1_knn:.4f}")
    print(f"   Accuracy: {accuracy_knn:.4f}")
    print(f"\n   📋 Rapport de classification:")
    print(classification_report(y_test_knn, predictions_knn))
else:
    print("⚠️  y_test_knn non disponible pour l'évaluation KNN")

# Évaluation MICE (si y_test_mice disponible)
if 'y_test_mice' in globals() and y_test_mice is not None:
    print("\n📊 PERFORMANCES MICE (WITH_REFIT):")
    f1_mice = f1_score(y_test_mice, predictions_mice)
    accuracy_mice = accuracy_score(y_test_mice, predictions_mice)
    
    print(f"   F1-score: {f1_mice:.4f}")
    print(f"   Accuracy: {accuracy_mice:.4f}")
    print(f"\n   📋 Rapport de classification:")
    print(classification_report(y_test_mice, predictions_mice))
else:
    print("⚠️  y_test_mice non disponible pour l'évaluation MICE")

# =============================================================================
# ÉTAPE 6 : SAUVEGARDE DES PRÉDICTIONS
# =============================================================================

print("\n🎯 ÉTAPE 6 : Sauvegarde des prédictions")
print("=" * 50)

import pandas as pd
from pathlib import Path

# Créer un DataFrame avec les prédictions
predictions_df = pd.DataFrame({
    'id': range(len(predictions_knn)),
    'prediction_knn_with_refit': predictions_knn,
    'prediction_mice_with_refit': predictions_mice,
    'probability_knn_with_refit': y_proba_knn,
    'probability_mice_with_refit': y_proba_mice
})

# Sauvegarder les prédictions
output_dir = Path("outputs/predictions")
output_dir.mkdir(parents=True, exist_ok=True)

# Sauvegarder en CSV
csv_path = output_dir / "predictions_manuelles_with_refit_avec_seuils_optimaux.csv"
predictions_df.to_csv(csv_path, index=False)
print(f"💾 Prédictions sauvegardées: {csv_path}")

# =============================================================================
# ÉTAPE 7 : PRÉPARATION POUR SOUMISSION
# =============================================================================

print("\n🎯 ÉTAPE 7 : Préparation pour soumission")
print("=" * 50)

# Créer les fichiers de soumission au format requis
def create_submission_file(predictions, method, output_dir):
    """Créer un fichier de soumission au format requis"""
    
    # Convertir les prédictions en format requis (ad./noad.)
    submission_predictions = []
    for pred in predictions:
        if pred == 1:
            submission_predictions.append("ad.")
        else:
            submission_predictions.append("noad.")
    
    # Sauvegarder sans header (format requis)
    submission_path = output_dir / f"predictions_finales_{method}_with_refit_submission.csv"
    with open(submission_path, 'w') as f:
        for pred in submission_predictions:
            f.write(pred + '\n')
    
    print(f"📁 Fichier de soumission {method} (with_refit): {submission_path}")
    print(f"   Nombre de prédictions: {len(submission_predictions)}")
    print(f"   Prédictions positives: {sum(1 for p in submission_predictions if p == 'ad.')}")
    print(f"   Prédictions négatives: {sum(1 for p in submission_predictions if p == 'noad.')}")
    
    return submission_path

# Créer les fichiers de soumission
submission_knn = create_submission_file(predictions_knn, "knn", output_dir)
submission_mice = create_submission_file(predictions_mice, "mice", output_dir)

# =============================================================================
# RÉSUMÉ FINAL
# =============================================================================

print("\n" + "=" * 70)
print("🎯 RÉSUMÉ DES PRÉDICTIONS MANUELLES (WITH_REFIT)")
print("=" * 70)

print("\n📊 SEUILS OPTIMAUX UTILISÉS:")
print(f"   KNN: {optimal_threshold_knn}")
print(f"   MICE: {optimal_threshold_mice}")

print("\n📊 PRÉDICTIONS GÉNÉRÉES:")
print(f"   KNN (with_refit): {len(predictions_knn)} prédictions")
print(f"   MICE (with_refit): {len(predictions_mice)} prédictions")

print("\n📁 FICHIERS CRÉÉS:")
print(f"   Prédictions détaillées: {csv_path}")
print(f"   Soumission KNN (with_refit): {submission_knn}")
print(f"   Soumission MICE (with_refit): {submission_mice}")

print("\n✅ PRÉDICTIONS MANUELLES TERMINÉES !")
print("=" * 70)

print("\n💡 Variables disponibles:")
print("   - predictions_knn : Prédictions KNN (0/1)")
print("   - predictions_mice : Prédictions MICE (0/1)")
print("   - y_proba_knn : Probabilités KNN")
print("   - y_proba_mice : Probabilités MICE")
print("   - optimal_threshold_knn : Seuil optimal KNN")
print("   - optimal_threshold_mice : Seuil optimal MICE")

print("\n🎯 Prochaines étapes:")
print("   1. Vérifier les fichiers de soumission")
print("   2. Tester les performances sur les données de validation")
print("   3. Choisir le meilleur modèle pour la soumission finale") 