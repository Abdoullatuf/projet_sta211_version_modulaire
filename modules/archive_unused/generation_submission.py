#modules/modeling/generation_submission.py

import pandas as pd
from pathlib import Path
import joblib

from preprocessing.final_preprocessing import prepare_final_dataset
from config.paths_config import setup_project_paths

from sklearn.ensemble import RandomForestClassifier  # ou mon modèle final réel

# === 1. Configurations ===
paths = setup_project_paths()
MODEL_PATH = paths["MODELS_DIR"] / "final_model.pkl"          # Mon modèle entraîné
TRANSFORMER_PATH = paths["MODELS_DIR"] / "yeojohnson.pkl"     # Transformateur sauvegardé
TEST_PATH = paths["RAW_DATA_DIR"] / "data_test.csv"           # Données test
SUBMISSION_PATH = paths["OUTPUTS_DIR"] / "submission.csv"     # Fichier à soumettre

# === 2. Prétraitement des données test ===
df_test = prepare_final_dataset(
    file_path=TEST_PATH,
    strategy="mixed_mar_mcar",
    mar_method="knn",  # ou "mice" selon ton modèle
    knn_k=5,
    drop_outliers=False,  # Jamais sur test
    correlation_threshold=0.95,
    save_transformer=False,
    processed_data_dir=None,
    models_dir=paths["MODELS_DIR"],
    display_info=True,
    require_outcome=False
)

# === 3. Chargement du modèle
model = joblib.load(MODEL_PATH)
print(f"✅ Modèle chargé : {MODEL_PATH}")

# === 4. Prédictions
X_test = df_test.drop(columns=["id"], errors="ignore")  # s'il y a une colonne `id`
y_pred = model.predict(X_test)

# === 5. Format de soumission
submission = pd.DataFrame({
    "id": df_test["id"] if "id" in df_test.columns else df_test.index,
    "prediction": y_pred
})

# === 6. Sauvegarde
SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)
submission.to_csv(SUBMISSION_PATH, index=False)
print(f"📤 Soumission sauvegardée : {SUBMISSION_PATH}")
