from project_setup import setup_project_paths
from data_preprocessing import load_data
import pandas as pd
import numpy as np
import joblib
import os

# Fonction alternative de génération de prédictions via MFA

def generate_submission_mfa(
    test_file="data_test.csv",
    imputer_file=None,
    scaler_file="scaler_knn.pkl",
    model_file="best_model_mfa.joblib",
    mfa_file="mfa_model.pkl",
    save_path="my_pred_mfa.csv",
):
    paths = setup_project_paths()
    RAW_DATA_DIR = paths["RAW_DATA_DIR"]
    MODELS_DIR = paths["MODELS_DIR"]
    ROOT_DIR = paths["ROOT_DIR"]

    df = load_data(RAW_DATA_DIR / test_file, require_outcome=False)
    if imputer_file is None:
        imputer_file = next(
            f for f in os.listdir(MODELS_DIR) if f.startswith("imputer_knn_k") and f.endswith(".pkl")
        )
    imputer = joblib.load(MODELS_DIR / imputer_file)

    required_cols = ["X1", "X2", "X3"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(f"❌ Colonnes manquantes : {missing}")
    df[required_cols] = imputer.transform(df[required_cols])
    df["X4"] = df["X4"].fillna(df["X4"].median())

    df = transform_selected_variables(df, safe_mode=True, verbose=True)
    mfa_model = joblib.load(MODELS_DIR / mfa_file)

    quantitative_vars = ["X1_log", "X2_boxcox", "X3_boxcox", "X4"]
    binary_vars = [col for col in df.columns if col not in quantitative_vars and df[col].nunique() == 2]
    mfa_input = df[quantitative_vars + binary_vars].copy()
    mfa_input.columns = pd.MultiIndex.from_tuples(
        [("Quantitatives", col) if col in quantitative_vars else ("Binaires", col) for col in mfa_input.columns]
    )
    df_mfa = mfa_model.transform(mfa_input)
    df_mfa.columns = [f"AFM_{i+1}" for i in range(df_mfa.shape[1])]

    scaler = joblib.load(MODELS_DIR / scaler_file)
    df_scaled = scaler.transform(df_mfa)

    model = joblib.load(MODELS_DIR / model_file)
    y_pred = model.predict(df_scaled)
    labels = np.where(y_pred == 1, "ad.", "noad.")

    pred_path = ROOT_DIR / save_path
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pd.Series(labels).to_csv(pred_path, index=False, header=False)
    print(f"✅ Fichier de prédiction (MFA) généré : {pred_path}")
