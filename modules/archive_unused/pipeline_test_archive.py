import sys
from pathlib import Path

# Ajout du dossier modules au PYTHONPATH
modules_dir = Path(__file__).parent.parent / "modules"
sys.path.insert(0, str(modules_dir.resolve()))
print("modules_dir ajouté à sys.path :", modules_dir.resolve())

from final_preprocessing import prepare_final_dataset
from project_setup import setup_project_paths

import pandas as pd
import numpy as np

# Choix du mode de test ("simple" = RandomForest, "complet" = pipeline Optuna)
mode = "simple"
if len(sys.argv) > 1 and sys.argv[1] == "complet":
    mode = "complet"

print(f"\n=== Lancement du test pipeline en mode : {mode.upper()} ===")

# Génération du dummy dataset
N = 500 if mode == "complet" else 200  # Pour accélérer en simple
data = pd.DataFrame({
    'X1': np.random.normal(0, 1, N),
    'X2': np.random.exponential(1, N),
    'X3': np.random.normal(5, 2, N),
    'X4': np.random.choice([1, 2, np.nan], N),
    'bin1': np.random.randint(0, 2, N),
    'bin2': np.random.randint(0, 2, N),
    'bin3': np.random.randint(0, 2, N),
    'outcome': np.random.choice(['ad.', 'noad.'], N)
})
dummy_path = Path(__file__).parent / "dummy_data.csv"
data.to_csv(dummy_path, index=False)
print("Dummy data CSV created at:", dummy_path.resolve())

# Initialisation des chemins
paths = setup_project_paths()

# Prétraitement complet
df_final = prepare_final_dataset(
    file_path=dummy_path,
    strategy="mixed_mar_mcar",
    mar_method="knn",
    knn_k=3,
    drop_outliers=False,
    processed_data_dir=paths["DATA_PROCESSED"],
    models_dir=paths["MODELS_DIR"],
    display_info=True,
    raw_data_dir=None,
    require_outcome=True
)

X = df_final.drop(columns=["outcome"])
y = (df_final["outcome"] == "ad.").astype(int)

# Split train/test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

if mode == "complet":
    from modeling import optimize_pipeline
    print("=== Test pipeline Optuna + stacking complet ===")
    # Essayez n_trials=2 pour un test rapide (montez à 10+ pour un vrai tuning)
    try:
        model, threshold, f1 = optimize_pipeline(
            X_train, y_train, X_test, y_test,
            imput_name="dummytest",
            models_dir=paths["MODELS_DIR"],
            n_trials=2
        )
        print(f"\nF1-score pipeline stacking : {f1:.3f}")
    except Exception as e:
        print("Erreur lors du pipeline complet :", e)
        sys.exit(1)
else:
    print("=== Test simple RandomForestClassifier ===")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score
    rf = RandomForestClassifier(n_estimators=30, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    print(f"\nF1-score RandomForest sur dummy : {f1:.3f}")

print("✅ Test pipeline exécuté avec succès !")
print("Fin du script de test")
