## Résumé du processus de stacking sans refit

import json
import pandas as pd
from pathlib import Path

def create_stacking_summary(stacking_dir, output_dir):
    """
    Crée un résumé complet du processus de stacking sans refit
    """
    print("📋 Création du résumé du stacking sans refit...")
    
    # Charger les performances
    with open(stacking_dir / "stack_no_refit_knn_performance.json", "r") as f:
        perf_knn = json.load(f)
    
    with open(stacking_dir / "stack_no_refit_mice_performance.json", "r") as f:
        perf_mice = json.load(f)
    
    # Créer le résumé
    summary = {
        "methodology": {
            "approach": "Stacking sans refit (moyenne des probabilités)",
            "base_models": ["Random Forest", "XGBoost", "Logistic Regression", "SVM", "MLP"],
            "imputation_methods": ["KNN", "MICE"],
            "threshold_optimization": "Optimisation du seuil pour maximiser le F1-score",
            "threshold_range": [0.2, 0.8],
            "threshold_steps": 61
        },
        "results": {
            "knn": {
                "f1_score": perf_knn["f1_score"],
                "precision": perf_knn["precision"],
                "recall": perf_knn["recall"],
                "threshold": perf_knn["threshold"]
            },
            "mice": {
                "f1_score": perf_mice["f1_score"],
                "precision": perf_mice["precision"],
                "recall": perf_mice["recall"],
                "threshold": perf_mice["threshold"]
            }
        },
        "files_saved": {
            "models": [
                "stack_no_refit_knn.joblib",
                "stack_no_refit_mice.joblib"
            ],
            "thresholds": [
                "best_thr_stack_no_refit_knn.json",
                "best_thr_stack_no_refit_mice.json"
            ],
            "performance": [
                "stack_no_refit_knn_performance.json",
                "stack_no_refit_mice_performance.json"
            ]
        }
    }
    
    # Déterminer le meilleur modèle
    if perf_knn["f1_score"] >= perf_mice["f1_score"]:
        summary["best_model"] = {
            "method": "knn",
            "f1_score": perf_knn["f1_score"],
            "reason": "F1-score plus élevé"
        }
    else:
        summary["best_model"] = {
            "method": "mice",
            "f1_score": perf_mice["f1_score"],
            "reason": "F1-score plus élevé"
        }
    
    # Sauvegarder le résumé
    summary_file = output_dir / "stacking_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    # Créer un résumé en markdown
    markdown_summary = f"""
# Résumé du Stacking Sans Refit

## Méthodologie

**Approche :** Stacking sans refit utilisant la moyenne des probabilités des modèles de base.

**Modèles de base :**
- Random Forest
- XGBoost  
- Logistic Regression
- SVM
- MLP

**Méthodes d'imputation :**
- KNN
- MICE

**Optimisation :** Seuil optimisé pour maximiser le F1-score (plage : 0.2-0.8, 61 étapes)

## Résultats

### KNN Imputation
- **F1-Score :** {perf_knn['f1_score']:.4f}
- **Précision :** {perf_knn['precision']:.4f}
- **Rappel :** {perf_knn['recall']:.4f}
- **Seuil optimal :** {perf_knn['threshold']:.3f}

### MICE Imputation
- **F1-Score :** {perf_mice['f1_score']:.4f}
- **Précision :** {perf_mice['precision']:.4f}
- **Rappel :** {perf_mice['recall']:.4f}
- **Seuil optimal :** {perf_mice['threshold']:.3f}

## Modèle Final

**Meilleur modèle :** {summary['best_model']['method'].upper()}
**F1-Score :** {summary['best_model']['f1_score']:.4f}
**Raison :** {summary['best_model']['reason']}

## Fichiers Sauvegardés

### Modèles
- `stack_no_refit_knn.joblib`
- `stack_no_refit_mice.joblib`

### Seuils Optimaux
- `best_thr_stack_no_refit_knn.json`
- `best_thr_stack_no_refit_mice.json`

### Métriques de Performance
- `stack_no_refit_knn_performance.json`
- `stack_no_refit_mice_performance.json`

## Utilisation

Pour faire des prédictions avec le meilleur modèle :

```python
from modules.modeling.final_predictions_stacking import make_final_predictions

# Charger les données de test
X_test = load_test_data()

# Faire les prédictions
results_df, model_info = make_final_predictions(X_test, stacking_dir, output_dir)
```
"""
    
    # Sauvegarder le markdown
    markdown_file = output_dir / "stacking_summary.md"
    with open(markdown_file, "w", encoding="utf-8") as f:
        f.write(markdown_summary)
    
    print(f"✅ Résumé sauvegardé dans : {summary_file}")
    print(f"📝 Documentation markdown dans : {markdown_file}")
    
    # Afficher un résumé concis
    print("\n" + "="*60)
    print("RÉSUMÉ DU STACKING SANS REFIT")
    print("="*60)
    print(f"🏆 Meilleur modèle : {summary['best_model']['method'].upper()}")
    print(f"📊 F1-Score : {summary['best_model']['f1_score']:.4f}")
    print(f"📁 Fichiers sauvegardés dans : {stacking_dir}")
    print("="*60)
    
    return summary

# Exécution
if __name__ == "__main__":
    summary = create_stacking_summary(stacking_dir, output_dir) 