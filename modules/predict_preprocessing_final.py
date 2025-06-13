

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





def generate_submission_mfa(
    test_file="data_test.csv",
    imputer_file=None,
    scaler_file="scaler_knn.pkl",
    model_file="best_model_mfa.joblib",
    mfa_file="mfa_model.pkl",
    save_path="my_pred_mfa.csv"
):
    paths = get_project_paths()
    RAW_DATA_DIR = paths["RAW_DATA_DIR"]
    MODELS_DIR = paths["MODELS_DIR"]
    ROOT_DIR = paths["ROOT_DIR"]
    df = load_data(RAW_DATA_DIR / test_file, require_outcome=False)
    if imputer_file is None:
        imputer_file = next(f for f in os.listdir(MODELS_DIR) if f.startswith("imputer_knn_k") and f.endswith(".pkl"))
    imputer = joblib.load(MODELS_DIR / imputer_file)
    required_cols = ['X1', 'X2', 'X3']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(f"❌ Colonnes manquantes : {missing}")
    df[required_cols] = imputer.transform(df[required_cols])
    df['X4'] = df['X4'].fillna(df['X4'].median())
    df = transform_selected_variables(df, safe_mode=True, verbose=True)
    mfa_model = joblib.load(MODELS_DIR / mfa_file)
    quantitative_vars = ['X1_log', 'X2_boxcox', 'X3_boxcox', 'X4']
    binary_vars = [col for col in df.columns if col not in quantitative_vars and df[col].nunique() == 2]
    mfa_input = df[quantitative_vars + binary_vars].copy()
    mfa_input.columns = pd.MultiIndex.from_tuples(
        [('Quantitatives', col) if col in quantitative_vars else ('Binaires', col) for col in mfa_input.columns]
    )
    df_mfa = mfa_model.transform(mfa_input)
    df_mfa.columns = [f"AFM_{i+1}" for i in range(df_mfa.shape[1])]
    df_scaled = scaler.transform(df_mfa)
    model = joblib.load(MODELS_DIR / model_file)
    y_pred = model.predict(df_scaled)
    labels = np.where(y_pred == 1, "ad.", "noad.")
    pred_path = ROOT_DIR / save_path
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pd.Series(labels).to_csv(pred_path, index=False, header=False)
    print(f"✅ Fichier de prédiction (MFA) généré : {pred_path}")
