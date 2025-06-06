# modules/optuna_pipeline.py

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_curve, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import StackingClassifier, VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import optuna
from optuna.samplers import TPESampler
import joblib
import matplotlib.pyplot as plt





def optimize_threshold_for_f1(y_true, y_proba, plot=False):
    thresholds = np.linspace(0.01, 0.99, 999)
    f1_scores = [f1_score(y_true, (y_proba >= t).astype(int)) for t in thresholds]
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx], f1_scores[best_idx]


def define_estimators(trial):
    estimators = []

    lgb_params = {
        'n_estimators': trial.suggest_int('lgb_n_estimators', 100, 800),
        'max_depth': trial.suggest_int('lgb_max_depth', 3, 12),
        'learning_rate': trial.suggest_float('lgb_lr', 0.01, 0.3),
        'num_leaves': trial.suggest_int('lgb_leaves', 20, 150)
    }
    estimators.append(('lgb', LGBMClassifier(**lgb_params, objective='binary', random_state=42, verbose=-1)))

    xgb_params = {
        'n_estimators': trial.suggest_int('xgb_n_estimators', 100, 800),
        'max_depth': trial.suggest_int('xgb_max_depth', 3, 10),
        'learning_rate': trial.suggest_float('xgb_lr', 0.01, 0.3),
        'subsample': trial.suggest_float('xgb_subsample', 0.7, 1.0)
    }
    estimators.append(('xgb', XGBClassifier(**xgb_params, use_label_encoder=False, eval_metric='logloss', random_state=42)))

    cat_params = {
        'iterations': trial.suggest_int('cat_iters', 100, 800),
        'depth': trial.suggest_int('cat_depth', 4, 10),
        'learning_rate': trial.suggest_float('cat_lr', 0.01, 0.3)
    }
    estimators.append(('cat', CatBoostClassifier(**cat_params, verbose=False, random_state=42)))

    return estimators


def optimize_pipeline(X_train, y_train, X_test, y_test, imput_name, models_dir, n_trials=100):
    def objective(trial):
        estimators = define_estimators(trial)

        meta_model = LogisticRegression(C=trial.suggest_float("meta_C", 0.1, 10), max_iter=1000, random_state=42)
        model = StackingClassifier(
            estimators=estimators,
            final_estimator=meta_model,
            cv=5,
            n_jobs=-1
        )

        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)[:, 1]
        threshold, f1 = optimize_threshold_for_f1(y_test, y_proba)
        return f1

    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)

    # === Reconstruction et entraînement final
    best_trial = study.best_trial
    estimators = define_estimators(best_trial)
    meta_model = LogisticRegression(C=best_trial.params["meta_C"], max_iter=1000, random_state=42)

    final_model = StackingClassifier(estimators=estimators, final_estimator=meta_model, cv=5, n_jobs=-1)
    final_model.fit(X_train, y_train)
    y_proba = final_model.predict_proba(X_test)[:, 1]
    threshold, f1 = optimize_threshold_for_f1(y_test, y_proba, plot=True)
    y_pred = (y_proba >= threshold).astype(int)

    print(f"\n📌 Seuil optimal : {threshold:.4f}")
    print(f"🏁 F1-score optimisé : {f1:.4f}")
    print("\n📋 Rapport de classification :")
    print(classification_report(y_test, y_pred))

    # Sauvegarde
    joblib.dump(final_model, models_dir / f"stacking_{imput_name}.joblib")
    with open(models_dir / f"threshold_{imput_name}.json", "w") as f:
        json.dump({
            "threshold": round(threshold, 4),
            "f1_score": round(f1, 4),
            "params": best_trial.params
        }, f, indent=4)

    print(f"💾 Modèle et seuil optimisés sauvegardés pour : {imput_name}")

    return final_model, threshold, f1