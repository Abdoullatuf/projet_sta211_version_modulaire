
#/modules/modeling/retraining.py


from sklearn.base import clone
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import pandas as pd
import joblib
from pathlib import Path

def retrain_model_with_selected_features(model, X_train, y_train, X_test, y_test,
                                         selected_features, model_name, dataset_name,
                                         save_dir, apply_smote=True):
    """
    Réentraîne un modèle avec les variables sélectionnées et sauvegarde le modèle final.

    Args:
        model: modèle entraîné initialement (sera cloné)
        X_train, y_train: données d'entraînement
        X_test, y_test: données de test (pour évaluation)
        selected_features: liste des noms de variables à utiliser
        model_name: nom du modèle (ex. "Stacking")
        dataset_name: nom du dataset (ex. "MICE")
        save_dir: répertoire de sauvegarde
        apply_smote: booléen pour activer ou non SMOTE dans le pipeline

    Returns:
        model_fitted: le pipeline entraîné
        f1: F1-score obtenu sur le test set
        model_path: chemin du modèle sauvegardé
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Sélection des variables
    X_train_sel = X_train[selected_features]
    X_test_sel = X_test[selected_features]

    # Pipeline
    steps = [('scaler', StandardScaler())]
    if apply_smote:
        steps.append(('smote', SMOTE(random_state=42)))
    steps.append(('classifier', clone(model)))

    pipeline = ImbPipeline(steps)

    # Entraînement
    pipeline.fit(X_train_sel, y_train)
    y_pred = pipeline.predict(X_test_sel)
    f1 = f1_score(y_test, y_pred)

    # Sauvegarde
    file_name = f"{model_name.lower()}_{dataset_name.lower()}_reduced.joblib".replace(" ", "_")
    model_path = save_dir / file_name
    joblib.dump(pipeline, model_path)

    return pipeline, f1, model_path
