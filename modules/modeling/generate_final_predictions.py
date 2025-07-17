import pandas as pd
import joblib
import json
from pathlib import Path

def load_columns(model_dir, imputation, outliers):
    outlier_tag = "" if outliers else "_no_outliers"
    col_path = model_dir / f"columns_{imputation}{outlier_tag}.pkl"
    return joblib.load(col_path)

def load_pipeline_and_threshold(model_dir, thresholds_dir, model_type, imputation, outliers):
    outlier_tag = "" if outliers else "_no_outliers"
    pipeline_path = model_dir / f"pipeline_{model_type.lower()}_{imputation}{outlier_tag}_champion.joblib"
    threshold_path = thresholds_dir / f"threshold_{model_type.lower()}_{imputation}{outlier_tag}.json"
    pipeline = joblib.load(pipeline_path)
    if threshold_path.exists():
        with open(threshold_path, "r") as f:
            threshold = json.load(f).get("threshold", 0.5)
    else:
        threshold = 0.5
    return pipeline, threshold

def generate_final_predictions(
    data_test_path,
    model_dir,
    thresholds_dir,
    submission_path,
    imputation="knn",                # "knn" ou "mice"
    outliers=True,                   # True = avec outliers, False = sans
    model_type="randomforest",       # "randomforest", "xgboost", etc.
    threshold=None                   # Si None, utilise le seuil optimisé sauvegardé
):
    # 1. Chargement des données test
    df_test = pd.read_csv(data_test_path)
    # 2. Chargement des colonnes utilisées à l'entraînement
    columns = load_columns(model_dir, imputation, outliers)
    # 3. Filtrage et réordonnancement des colonnes du test
    X_test = df_test[columns].copy()
    # 4. Chargement du pipeline et du seuil optimisé
    pipeline, threshold_opt = load_pipeline_and_threshold(model_dir, thresholds_dir, model_type, imputation, outliers)
    if threshold is None:
        threshold = threshold_opt
    # 5. Application du pipeline (imputation, standardisation, etc.)
    X_test_transformed = pipeline.transform(X_test)
    # 6. Prédiction des probabilités
    y_proba = pipeline.named_steps['clf'].predict_proba(X_test_transformed)[:, 1]
    # 7. Application du seuil optimisé
    y_pred = (y_proba >= threshold)
    # 8. Mapping en labels attendus
    y_pred_labels = pd.Series(y_pred).map({True: "ad.", False: "noad."})
    # 9. Vérification du format
    assert y_pred_labels.isin(["ad.", "noad."]).all(), "Valeurs de prédiction incorrectes"
    assert len(y_pred_labels) == len(df_test), "Le nombre de prédictions est incorrect"
    # 10. Sauvegarde au format attendu (pas d'index, pas d'en-tête)
    y_pred_labels.to_csv(submission_path, index=False, header=False)
    print(f"✅ Fichier de soumission généré : {submission_path}")
    return y_pred_labels

# Exemple d'utilisation (à adapter dans un notebook ou un script)
if __name__ == "__main__":
    DATA_TEST_PATH = Path("data/raw/data_test.csv")
    MODEL_DIR = Path("models")
    THRESHOLDS_DIR = Path("outputs/modeling/thresholds")
    SUBMISSION_PATH = Path("outputs/predictions_finales.csv")
    # Ces paramètres peuvent être modifiés dynamiquement
    generate_final_predictions(
        data_test_path=DATA_TEST_PATH,
        model_dir=MODEL_DIR,
        thresholds_dir=THRESHOLDS_DIR,
        submission_path=SUBMISSION_PATH,
        imputation="knn",           # ou "mice"
        outliers=True,              # ou False
        model_type="randomforest"   # ou "xgboost", etc.
    ) 