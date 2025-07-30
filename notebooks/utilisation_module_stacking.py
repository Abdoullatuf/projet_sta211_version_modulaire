# =============================================================================
# EXEMPLE D'UTILISATION DU MODULE DE STACKING
# =============================================================================

# Import du module
from modules.modeling.stacking_models_creator import create_stacking_models, get_stacking_models_info

# =============================================================================
# OPTION 1 : CRÉER LES MODÈLES POUR KNN ET MICE
# =============================================================================

print("🎯 OPTION 1 : Création des modèles pour KNN et MICE")
print("=" * 60)

# Créer tous les modèles (KNN + MICE)
models = create_stacking_models(imputation_method="both", verbose=True)

# Afficher les informations sur les modèles créés
get_stacking_models_info(models)

# =============================================================================
# OPTION 2 : CRÉER SEULEMENT LES MODÈLES KNN
# =============================================================================

print("\n" + "=" * 60)
print("🎯 OPTION 2 : Création des modèles KNN seulement")
print("=" * 60)

# Créer seulement les modèles KNN
models_knn = create_stacking_models(imputation_method="knn", verbose=True)

# =============================================================================
# OPTION 3 : CRÉER SEULEMENT LES MODÈLES MICE
# =============================================================================

print("\n" + "=" * 60)
print("🎯 OPTION 3 : Création des modèles MICE seulement")
print("=" * 60)

# Créer seulement les modèles MICE
models_mice = create_stacking_models(imputation_method="mice", verbose=True)

# =============================================================================
# UTILISATION DES MODÈLES CRÉÉS
# =============================================================================

print("\n" + "=" * 60)
print("🎯 UTILISATION DES MODÈLES")
print("=" * 60)

# Exemple d'utilisation avec les modèles créés
if "stacking_classifier_knn" in models:
    print("\n📊 Utilisation du modèle KNN :")
    print("   # Entraîner le modèle")
    print("   models['stacking_classifier_knn'].fit(X_train_knn, y_train_knn)")
    print("   ")
    print("   # Faire des prédictions")
    print("   y_pred_knn = models['stacking_classifier_knn'].predict(X_val_knn)")
    print("   ")
    print("   # Évaluer les performances")
    print("   from sklearn.metrics import f1_score")
    print("   f1_knn = f1_score(y_val_knn, y_pred_knn)")
    print("   print(f'F1 KNN: {f1_knn:.4f}')")

if "stacking_classifier_mice" in models:
    print("\n📊 Utilisation du modèle MICE :")
    print("   # Entraîner le modèle")
    print("   models['stacking_classifier_mice'].fit(X_train_mice, y_train_mice)")
    print("   ")
    print("   # Faire des prédictions")
    print("   y_pred_mice = models['stacking_classifier_mice'].predict(X_val_mice)")
    print("   ")
    print("   # Évaluer les performances")
    print("   f1_mice = f1_score(y_val_mice, y_pred_mice)")
    print("   print(f'F1 MICE: {f1_mice:.4f}')")

print("\n" + "=" * 60)
print("✅ MODULE PRÊT POUR L'UTILISATION !")
print("=" * 60)

print("\n📋 Variables disponibles dans 'models' :")
for key in models.keys():
    if 'stacking_classifier' in key:
        print(f"   🔹 {key} : Modèle de stacking prêt à l'entraînement")
    elif 'params' in key:
        print(f"   🔹 {key} : Paramètres optimisés")
    else:
        print(f"   🔹 {key} : Modèle de base")

print("\n🎯 Pour utiliser dans votre notebook :")
print("   from modules.modeling.stacking_models_creator import create_stacking_models")
print("   models = create_stacking_models(imputation_method='both')")
print("   stacking_knn = models['stacking_classifier_knn']")
print("   stacking_mice = models['stacking_classifier_mice']") 