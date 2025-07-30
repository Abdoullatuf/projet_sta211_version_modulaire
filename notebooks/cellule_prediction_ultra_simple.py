# =============================================================================
# CELLULE DE PRÉDICTION ULTRA-SIMPLE - UTILISE LE SCRIPT QUI FONCTIONNE
# =============================================================================

print("🚀 GÉNÉRATION DES PRÉDICTIONS FINALES")
print("=" * 50)

from pathlib import Path

# -----------------------------------------------------------------------------
# CONFIGURATION DES CHEMINS
# -----------------------------------------------------------------------------

try:
    # Importer la configuration des chemins
    from modules.config.paths_config import setup_project_paths
    
    # Configurer les chemins du projet
    paths = setup_project_paths()
    RAW_DATA_DIR = paths["RAW_DATA_DIR"]
    MODELS_DIR = paths["MODELS_DIR"]
    OUTPUTS_DIR = paths["OUTPUTS_DIR"]
    ROOT_DIR = paths["ROOT_DIR"]
    
    print(f"📂 ROOT_DIR : {ROOT_DIR}")
    print(f"📂 RAW_DATA_DIR : {RAW_DATA_DIR}")
    print(f"📂 MODELS_DIR : {MODELS_DIR}")
    print(f"📂 OUTPUTS_DIR : {OUTPUTS_DIR}")
    
    # Vérifier que le fichier de test existe
    test_data_path = RAW_DATA_DIR / "data_test.csv"
    if test_data_path.exists():
        print(f"✅ Fichier de test trouvé : {test_data_path}")
    else:
        print(f"❌ Fichier de test non trouvé : {test_data_path}")
        raise FileNotFoundError(f"Fichier de test non trouvé : {test_data_path}")
    
except Exception as e:
    print(f"❌ Erreur configuration : {e}")
    # Fallback avec chemins par défaut
    ROOT_DIR = Path.cwd()
    RAW_DATA_DIR = Path("data/raw")
    MODELS_DIR = Path("models")
    OUTPUTS_DIR = Path("outputs")

# -----------------------------------------------------------------------------
# TROUVER LE SCRIPT PREDICTION_SUBMISSION.PY
# -----------------------------------------------------------------------------

# Essayer différents chemins possibles
possible_script_paths = [
    "prediction_submission.py",                                    # Répertoire courant
    ROOT_DIR / "prediction_submission.py",                        # Racine du projet
    Path.cwd().parent / "prediction_submission.py",               # Répertoire parent
    "/content/prediction_submission.py",                          # Colab
    "/content/drive/MyDrive/projet_sta211/prediction_submission.py"  # Google Drive
]

script_path = None
for path in possible_script_paths:
    if Path(path).exists():
        script_path = path
        break

if script_path:
    print(f"✅ Script trouvé : {script_path}")
else:
    print("❌ Script prediction_submission.py non trouvé")
    print("🔧 Chemins essayés :")
    for path in possible_script_paths:
        print(f"   - {path}")
    raise FileNotFoundError("Script prediction_submission.py non trouvé")

# -----------------------------------------------------------------------------
# MODIFIER ET EXÉCUTER LE SCRIPT
# -----------------------------------------------------------------------------

try:
    # Lire le contenu du script
    script_content = open(script_path, 'r').read()
    
    # Remplacer les chemins hardcodés par les chemins corrects
    print("🔧 Modification des chemins dans le script...")
    
    # Remplacer le chemin des données de test
    script_content = script_content.replace(
        "test_data_path=Path(\"data/raw/data_test.csv\")",
        f"test_data_path=Path(\"{RAW_DATA_DIR}/data_test.csv\")"
    )
    
    # Remplacer le chemin des modèles
    script_content = script_content.replace(
        "models_dir=Path(\"models\")",
        f"models_dir=Path(\"{MODELS_DIR}\")"
    )
    
    # Remplacer le chemin de sortie
    script_content = script_content.replace(
        "output_dir=Path(\"outputs/predictions\")",
        f"output_dir=Path(\"{OUTPUTS_DIR}/predictions\")"
    )
    
    print("✅ Chemins modifiés dans le script")
    
    # Exécuter le script modifié
    exec(script_content)
    
    print("✅ Prédictions générées avec succès !")
    
    # Vérification rapide
    submission_path = OUTPUTS_DIR / "predictions/predictions_finales_stacking_knn_submission.csv"
    
    if submission_path.exists():
        with open(submission_path, 'r') as f:
            lines = f.readlines()
        print(f"📊 Nombre de lignes : {len(lines)}")
        print(f"📁 Fichier : {submission_path}")
        print("✅ Prêt pour la soumission !")
        
        # Afficher les premières lignes
        print("\n📝 PREMIÈRES LIGNES :")
        print("-" * 20)
        for i, line in enumerate(lines[:5]):
            print(f"{i+1}: {line.strip()}")
        if len(lines) > 5:
            print(f"... et {len(lines)-5} lignes supplémentaires")
    else:
        print("❌ Fichier de soumission non trouvé")
    
except Exception as e:
    print(f"❌ Erreur : {e}")
    
    # Fallback : essayer avec les modules
    try:
        print("\n🔧 Tentative avec les modules...")
        from modules.modeling.generate_final_predictions import generate_final_predictions
        
        y_pred_labels = generate_final_predictions(
            data_test_path=RAW_DATA_DIR / "data_test.csv",
            model_dir=MODELS_DIR,
            thresholds_dir=OUTPUTS_DIR / "modeling/thresholds",
            submission_path=OUTPUTS_DIR / "predictions/predictions_finales_stacking_knn_submission.csv",
            imputation="knn",
            outliers=True,
            model_type="randomforest"
        )
        print("✅ Prédictions générées avec les modules !")
        
    except Exception as e2:
        print(f"❌ Erreur modules : {e2}")

print("\n" + "=" * 50)
print("🚀 FIN")
print("=" * 50) 