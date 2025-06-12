

from project_setup import setup_project_paths
paths = setup_project_paths()

from data_preprocessing import load_data

import pandas as pd
import numpy as np
import joblib
import os

# On importe la fonction de transformation Yeo-Johnson de ton pipeline final, si besoin
from final_preprocessing import apply_yeojohnson

def generate_submission(
    test_file="data_test.csv",
    imputer_file=None,
    scaler_file=None,
    model_file="best_final_modele.joblib",
    columns_file="optimal_features_stacking_best.joblib",
    save_path="my_pred.csv",
    apply_threshold=True,
    threshold=0.605  # Met ici ton seuil optimal stacking
):
    paths = setup_project_paths()
    RAW_DATA_DIR = paths["RAW_DATA_DIR"]
    MODELS_DIR = paths["MODELS_DIR"]
    ROOT_DIR = paths["ROOT_DIR"]

    print("🔵 Chargement des données de test...")
    df = load_data(RAW_DATA_DIR / test_file, require_outcome=False)

    # Étape d'imputation éventuelle
    if imputer_file:
        print("🔵 Imputation des variables X1-X4...")
        imputer = joblib.load(MODELS_DIR / imputer_file)
        df[['X1', 'X2', 'X3']] = imputer.transform(df[['X1', 'X2', 'X3']])
        df['X4'] = df['X4'].fillna(df['X4'].median())

    # Transformation Yeo-Johnson sur X1, X2, X3 (conforme au pipeline)
    print("🔵 Transformation Yeo-Johnson sur X1, X2, X3...")
    # Si tu as sauvé le transformer, recharge-le pour appliquer exactement la même transfo que sur le train
    yj_path = MODELS_DIR / "yeojohnson.pkl"
    if yj_path.exists():
        transformer = joblib.load(yj_path)
        X_num = transformer.transform(df[['X1', 'X2', 'X3']])
        df[["X1_trans", "X2_trans", "X3_trans"]] = X_num
        df.drop(columns=["X1", "X2", "X3"], inplace=True)
    else:
        # Applique la transformation directe si pas de modèle sauvegardé (moins conseillé)
        df, _ = apply_yeojohnson(df, columns=["X1", "X2", "X3"], standardize=False)

    # Sélection des variables optimales
    print("🔵 Chargement des colonnes utilisées...")
    selected_features = joblib.load(MODELS_DIR / columns_file)
    df = df[selected_features]

    # Scaling éventuel (rare si tout est dans pipeline)
    if scaler_file:
        print("🔵 Scaling des données de test...")
        scaler = joblib.load(MODELS_DIR / scaler_file)
        df = pd.DataFrame(scaler.transform(df), columns=selected_features)

    print("🔵 Chargement du modèle final...")
    model = joblib.load(MODELS_DIR / model_file)

    print("🛠 Génération des prédictions...")
    if apply_threshold and hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(df)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)
    else:
        y_pred = model.predict(df)

    labels = np.where(y_pred == 1, "ad.", "noad.")

    # Nom automatique du fichier
    if save_path == "my_pred.csv":
        suffix = model_file.replace("best_", "").replace("final_model_", "").replace(".joblib", "")
        if apply_threshold and threshold is not None:
            suffix += f"_thresh{int(threshold * 1000):03d}"
        save_path = f"my_pred_{suffix}.csv"

    pred_path = ROOT_DIR / save_path
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pd.Series(labels).to_csv(pred_path, index=False, header=False)

    print(f"✅ Fichier de prédiction généré : {pred_path}")





