
#/modules/modeling/retraining.py

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.base import clone
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import pandas as pd
import joblib
from pathlib import Path

def retrain_model_with_selected_features(
    model, X_train, y_train, X_test, y_test,
    selected_features, model_name, dataset_name,
    save_dir, apply_smote=True,
    param_grid: dict = None,
    search_type: str = "grid",  # ➕ nouveau paramètre : 'grid' ou 'random'
    n_iter: int = 30,           # ➕ applicable pour RandomizedSearch
    cv: int = 5
):
    """
    Réentraîne un modèle avec les variables sélectionnées, avec possibilité
    d'optimisation des hyperparamètres par GridSearch ou RandomizedSearch.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    X_train_sel = X_train[selected_features]
    X_test_sel = X_test[selected_features]

    steps = [('scaler', StandardScaler())]
    if apply_smote:
        steps.append(('smote', SMOTE(random_state=42)))
    steps.append(('classifier', clone(model)))

    pipeline = ImbPipeline(steps)

    if param_grid:
        if search_type == "grid":
            search = GridSearchCV(
                pipeline,
                param_grid=param_grid,
                scoring='f1',
                cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
                n_jobs=-1,
                verbose=1
            )
        elif search_type == "random":
            search = RandomizedSearchCV(
                pipeline,
                param_distributions=param_grid,
                n_iter=n_iter,
                scoring='f1',
                cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
                n_jobs=-1,
                verbose=1,
                random_state=42
            )
        else:
            raise ValueError("search_type must be 'grid' or 'random'")

        search.fit(X_train_sel, y_train)
        pipeline = search.best_estimator_
    else:
        pipeline.fit(X_train_sel, y_train)

    y_pred = pipeline.predict(X_test_sel)
    f1 = f1_score(y_test, y_pred)

    file_name = f"{model_name.lower()}_{dataset_name.lower()}_reduced.joblib".replace(" ", "_")
    model_path = save_dir / file_name
    joblib.dump(pipeline, model_path)

    return pipeline, f1, model_path

