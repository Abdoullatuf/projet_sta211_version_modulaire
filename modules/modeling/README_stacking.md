# Stacking Sans Refit - Documentation

Ce dossier contient tous les scripts pour implémenter et évaluer le stacking sans refit sur les données KNN et MICE.

## 📁 Fichiers Disponibles

### Scripts Principaux

1. **`stacking_no_refit_knn.py`**
   - Stacking sans refit pour les données KNN
   - Optimisation du seuil pour maximiser le F1-score
   - Sauvegarde complète des modèles et métriques

2. **`stacking_no_refit_mice.py`**
   - Stacking sans refit pour les données MICE
   - Optimisation du seuil pour maximiser le F1-score
   - Sauvegarde complète des modèles et métriques

3. **`load_and_use_stacking_no_refit.py`**
   - Fonctions pour charger et utiliser les modèles de stacking
   - Prédictions sur de nouvelles données

4. **`compare_stacking_no_refit.py`**
   - Comparaison des résultats entre KNN et MICE
   - Graphiques de comparaison
   - Sauvegarde des comparaisons

5. **`final_predictions_stacking.py`**
   - Prédictions finales avec le meilleur modèle
   - Sélection automatique du meilleur modèle (KNN ou MICE)
   - Sauvegarde des prédictions et informations

6. **`stacking_summary.py`**
   - Création de résumés complets
   - Documentation en JSON et Markdown
   - Statistiques finales

## 🚀 Utilisation

### 1. Exécuter le Stacking

```python
# Pour KNN
exec(open('modules/modeling/stacking_no_refit_knn.py').read())

# Pour MICE
exec(open('modules/modeling/stacking_no_refit_mice.py').read())
```

### 2. Comparer les Résultats

```python
exec(open('modules/modeling/compare_stacking_no_refit.py').read())
```

### 3. Faire des Prédictions Finales

```python
from modules.modeling.final_predictions_stacking import make_final_predictions

# Charger vos données de test
X_test = load_test_data()

# Faire les prédictions
results_df, model_info = make_final_predictions(X_test, stacking_dir, output_dir)
```

### 4. Créer un Résumé

```python
exec(open('modules/modeling/stacking_summary.py').read())
```

## 📊 Métriques Sauvegardées

Pour chaque méthode d'imputation (KNN et MICE) :

- **F1-Score** : Métrique principale optimisée
- **Précision** : Précision des prédictions positives
- **Rappel** : Sensibilité du modèle
- **Seuil optimal** : Seuil qui maximise le F1-score

## 📁 Fichiers de Sortie

### Modèles
- `stack_no_refit_knn.joblib` : Modèle de stacking KNN
- `stack_no_refit_mice.joblib` : Modèle de stacking MICE

### Seuils Optimaux
- `best_thr_stack_no_refit_knn.json` : Seuil optimal KNN
- `best_thr_stack_no_refit_mice.json` : Seuil optimal MICE

### Métriques de Performance
- `stack_no_refit_knn_performance.json` : Métriques KNN
- `stack_no_refit_mice_performance.json` : Métriques MICE

### Comparaisons
- `comparison_stacking_no_refit.csv` : Tableau de comparaison
- `comparison_stacking_no_refit.png` : Graphiques de comparaison

### Résumés
- `stacking_summary.json` : Résumé complet en JSON
- `stacking_summary.md` : Documentation en Markdown

## 🔧 Méthodologie

### Stacking Sans Refit
1. **Chargement** : Charger les pipelines déjà entraînés
2. **Prédictions** : Générer les probabilités pour chaque modèle
3. **Moyenne** : Calculer la moyenne des probabilités
4. **Optimisation** : Trouver le seuil optimal (0.2-0.8, 61 étapes)
5. **Évaluation** : Calculer F1, précision, rappel
6. **Sauvegarde** : Sauvegarder modèles, seuils et métriques

### Modèles de Base
- Random Forest
- XGBoost
- Logistic Regression
- SVM
- MLP

### Optimisation
- **Métrique** : F1-score
- **Plage** : 0.2 à 0.8
- **Étapes** : 61 seuils testés

## 📈 Interprétation des Résultats

### F1-Score
- Métrique principale pour évaluer les performances
- Équilibrée entre précision et rappel
- Optimisée pour le seuil de classification

### Seuil Optimal
- Détermine la frontière de décision
- Optimisé pour maximiser le F1-score
- Différent pour chaque méthode d'imputation

### Comparaison KNN vs MICE
- Permet de choisir la meilleure méthode d'imputation
- Basée sur les performances F1-score
- Inclut graphiques et tableaux de comparaison

## 🎯 Recommandations

1. **Exécuter les deux méthodes** : KNN et MICE pour comparaison
2. **Utiliser le meilleur modèle** : Automatiquement sélectionné
3. **Documenter les résultats** : Créer des résumés complets
4. **Sauvegarder tout** : Modèles, seuils, métriques et comparaisons

## 🔍 Dépannage

### Erreurs Courantes
- **Fichiers manquants** : Vérifier que les pipelines sont sauvegardés
- **Chemins incorrects** : Vérifier les variables `stacking_dir`, `MODELS_NB2_DIR`
- **Données manquantes** : S'assurer que `X_test_knn`, `y_test_knn`, etc. sont définis

### Vérifications
- Les pipelines supportent `predict_proba`
- Les données de test sont dans le bon format
- Les dossiers de sauvegarde existent

## 📞 Support

Pour toute question ou problème :
1. Vérifier les chemins et variables
2. S'assurer que tous les pipelines sont sauvegardés
3. Contrôler les formats de données
4. Consulter les logs d'erreur 