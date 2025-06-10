#modules/modeling/stacking_model_selection.py

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import joblib
import pandas as pd
from pathlib import Path

def train_and_select_best_stacking_model(
    X_train, y_train, X_test, y_test, save_dir: Path, prefix: str = "model_final"
):
    """
    Entraîne un modèle de stacking avec GridSearchCV et RandomizedSearchCV,
    compare les deux et enregistre le meilleur modèle avec son F1-score.

    Args:
        X_train: Données d'entraînement (features)
        y_train: Cible d'entraînement
        X_test: Données de test (features)
        y_test: Cible de test
        save_dir (Path): dossier de sauvegarde
        prefix (str): préfixe pour nommer les fichiers sauvegardés

    Returns:
        best_model: Le meilleur modèle de stacking
        comparison_df: DataFrame contenant la comparaison des performances
    """
    base_estimators = [
        ('lr', LogisticRegression(max_iter=1000, solver='liblinear', penalty='l2', random_state=42)),
        ('dt', DecisionTreeClassifier(random_state=42)),
        ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
    ]

    final_est = RandomForestClassifier(
        random_state=42, n_jobs=-1, class_weight='balanced'
    )

    stack_model = StackingClassifier(
        estimators=base_estimators,
        final_estimator=final_est,
        cv=5,
        n_jobs=-1
    )

    param_grid = {
        'lr__C': [0.01, 0.1, 1],
        'lr__penalty': ['l1', 'l2'],
        'dt__max_depth': [3, None],
        'xgb__n_estimators': [100],
        'xgb__max_depth': [3],
        'xgb__learning_rate': [0.05, 0.1],
        'final_estimator__n_estimators': [100],
        'final_estimator__max_depth': [5, None]
    }

    param_distributions = {
        'lr__C': [0.01, 0.1, 1],
        'lr__penalty': ['l1', 'l2'],
        'dt__max_depth': [3, None],
        'dt__min_samples_split': [2, 10],
        'xgb__n_estimators': [100, 200],
        'xgb__max_depth': [3, 5],
        'xgb__learning_rate': [0.05, 0.1],
        'final_estimator__n_estimators': [100, 200],
        'final_estimator__max_depth': [5, None]
    }

    print("\U0001f535 GridSearchCV en cours...")
    grid_search = GridSearchCV(stack_model, param_grid, scoring='f1', cv=5, n_jobs=-1, refit=True)
    grid_search.fit(X_train, y_train)
    f1_grid = f1_score(y_test, grid_search.predict(X_test))

    print("\n\U0001f7e0 RandomizedSearchCV en cours...")
    random_search = RandomizedSearchCV(
        stack_model, param_distributions, scoring='f1',
        cv=5, n_jobs=-1, n_iter=50, verbose=1, random_state=42, refit=True
    )
    random_search.fit(X_train, y_train)
    f1_random = f1_score(y_test, random_search.predict(X_test))

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    sns.heatmap(confusion_matrix(y_test, grid_search.predict(X_test)), annot=True, fmt="d", cmap="Blues")
    plt.title("GridSearch Stacking")
    plt.xlabel("Prédit"); plt.ylabel("Réel")

    plt.subplot(1, 2, 2)
    sns.heatmap(confusion_matrix(y_test, random_search.predict(X_test)), annot=True, fmt="d", cmap="Oranges")
    plt.title("RandomizedSearch Stacking")
    plt.xlabel("Prédit"); plt.ylabel("Réel")
    plt.tight_layout()
    plt.savefig(save_dir / f"{prefix}_stacking_confusions.png", dpi=300)
    plt.close()

    comparison_df = pd.DataFrame({
        "Méthode de recherche": ["GridSearch", "RandomizedSearch"],
        "F1-score": [f1_grid, f1_random],
        "Best Params": [grid_search.best_params_, random_search.best_params_]
    })
    comparison_df.to_csv(save_dir / f"{prefix}_stacking_comparison.csv", index=False)

    if f1_grid >= f1_random:
        best_model = grid_search.best_estimator_
        best_method = "GridSearch"
    else:
        best_model = random_search.best_estimator_
        best_method = "RandomizedSearch"

    joblib.dump(best_model, save_dir / f"{prefix}_stacking_best_model.joblib")

    print(f"\n\U0001f3c6 Meilleur modèle : {best_method} (F1 = {max(f1_grid, f1_random):.4f})")
    print(classification_report(y_test, best_model.predict(X_test)))

    return best_model, comparison_df
