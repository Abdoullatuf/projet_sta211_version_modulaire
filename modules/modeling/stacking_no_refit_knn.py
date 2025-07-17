## Stacking sans refit et optimisation du seuil sur les données KNN

import numpy as np
import joblib
import json
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 1. Charge les pipelines déjà fit (KNN)
print("Chargement des pipelines...")
rf_pipe_knn  = joblib.load(MODELS_NB2_DIR / "randforest" / "pipeline_randforest_knn.joblib")
xgb_pipe_knn = joblib.load(MODELS_NB2_DIR / "xgboost" / "pipeline_xgboost_knn.joblib")
logreg_pipe_knn = joblib.load(MODELS_NB2_DIR / "logreg" / "pipeline_logreg_knn.joblib")
svm_pipe_knn = joblib.load(MODELS_NB2_DIR / "svm" / "pipeline_svm_knn.joblib")
mlp_pipe_knn = joblib.load(MODELS_NB2_DIR / "mlp" / "pipeline_mlp_knn.joblib")

# 2. Prédictions de proba sur X_test_knn
print("Génération des prédictions de probabilité...")
proba_rf     = rf_pipe_knn.predict_proba(X_test_knn)[:, 1]
proba_xgb    = xgb_pipe_knn.predict_proba(X_test_knn)[:, 1]
proba_logreg = logreg_pipe_knn.predict_proba(X_test_knn)[:, 1]
proba_svm    = svm_pipe_knn.predict_proba(X_test_knn)[:, 1]
proba_mlp    = mlp_pipe_knn.predict_proba(X_test_knn)[:, 1]

# 3. Moyenne des probabilités (tous les modèles)
proba_mean = np.mean([proba_rf, proba_xgb, proba_logreg, proba_svm, proba_mlp], axis=0)

# 4. Optimisation du seuil pour maximiser le F1-score
print("Optimisation du seuil...")
thresholds = np.linspace(0.2, 0.8, 61)
best_f1 = 0
best_thr_stack_no_refit_knn = 0.5

for thr in thresholds:
    y_pred = (proba_mean >= thr).astype(int)
    f1 = f1_score(y_test_knn, y_pred)
    if f1 > best_f1:
        best_f1 = f1
        best_thr_stack_no_refit_knn = thr

# 5. Prédiction finale au seuil optimal
y_pred_opt = (proba_mean >= best_thr_stack_no_refit_knn).astype(int)

# 6. Analyse détaillée
f1 = f1_score(y_test_knn, y_pred_opt)
precision = precision_score(y_test_knn, y_pred_opt)
recall = recall_score(y_test_knn, y_pred_opt)
cm = confusion_matrix(y_test_knn, y_pred_opt)

print(f"Meilleur F1 stacking KNN (seuil optimisé): {f1:.4f} (seuil={best_thr_stack_no_refit_knn:.3f})")
print(f"Précision : {precision:.4f}")
print(f"Rappel    : {recall:.4f}")
print("Matrice de confusion :\n", cm)

# Affichage graphique de la matrice de confusion
plt.figure(figsize=(4, 3))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Prédit: Non-ad", "Prédit: Ad"],
            yticklabels=["Réel: Non-ad", "Réel: Ad"])
plt.xlabel("Prédit")
plt.ylabel("Réel")
plt.title(f"Matrice de confusion\nStacking KNN (F1={f1:.3f}, seuil={best_thr_stack_no_refit_knn:.2f})")
plt.tight_layout()
plt.show()

# 7. Sauvegarde complète
print("Sauvegarde des résultats...")

# Créer le dictionnaire de stacking
stack_no_refit_knn = {
    "pipelines": {
        "rf": rf_pipe_knn,
        "xgb": xgb_pipe_knn,
        "logreg": logreg_pipe_knn,
        "svm": svm_pipe_knn,
        "mlp": mlp_pipe_knn
    },
    "threshold": float(best_thr_stack_no_refit_knn),
    "performance": {
        "f1_score": float(f1),
        "precision": float(precision),
        "recall": float(recall)
    }
}

# Sauvegarder le modèle de stacking
joblib.dump(stack_no_refit_knn, stacking_dir / "stack_no_refit_knn.joblib")

# Sauvegarder le seuil optimal séparément
with open(stacking_dir / "best_thr_stack_no_refit_knn.json", "w") as f:
    json.dump({"best_thr_stack_no_refit_knn": float(best_thr_stack_no_refit_knn)}, f, indent=2)

# Sauvegarder les métriques de performance
performance_metrics = {
    "f1_score": float(f1),
    "precision": float(precision),
    "recall": float(recall),
    "threshold": float(best_thr_stack_no_refit_knn),
    "confusion_matrix": cm.tolist()
}

with open(stacking_dir / "stack_no_refit_knn_performance.json", "w") as f:
    json.dump(performance_metrics, f, indent=2)

print("Sauvegarde terminée !")
print(f"Fichiers sauvegardés dans : {stacking_dir}") 