# modules/data_management/artefact_loader.py
import joblib
import json
import pandas as pd
from pathlib import Path
import logging

log = logging.getLogger(__name__)

def load_stacking_artefacts(paths):
    """
    Recharge tous les artefacts nécessaires pour le début du Notebook 03 (Stacking).

    Args:
        paths (dict): Un dictionnaire contenant les chemins vers les répertoires du projet.

    Returns:
        tuple: Un tuple contenant les artefacts chargés :
               (splits, all_optimized_pipelines, all_thresholds, df_all_thr, feature_cols)
               Chaque élément peut être None si le chargement a échoué.
    """
    print("🔄 Rechargement de tous les artefacts pour le Notebook 03 (Stacking)...")
    print("=" * 60)

    # Initialize variables to None
    splits = None
    all_optimized_pipelines = {}
    all_thresholds = {}
    df_all_thr = None
    feature_cols = None

    # 1. Charger les données splitées (nécessaire pour générer les OOF predictions)
    print("💾 Chargement des données splitées (train, val, test)...")
    try:
        knn_split_dir = paths["MODELS_DIR"] / "notebook2" / "knn"
        mice_split_dir = paths["MODELS_DIR"] / "notebook2" / "mice"

        if not knn_split_dir.exists() or not mice_split_dir.exists():
             raise FileNotFoundError("Les dossiers de split KNN ou MICE n'existent pas. Exécutez le Notebook 02 jusqu'au bout.")

        splits = {
            "knn": {
                "X_train": joblib.load(knn_split_dir / "knn_train.pkl")["X"],
                "y_train": joblib.load(knn_split_dir / "knn_train.pkl")["y"],
                "X_val": joblib.load(knn_split_dir / "knn_val.pkl")["X"],
                "y_val": joblib.load(knn_split_dir / "knn_val.pkl")["y"],
                "X_test": joblib.load(knn_split_dir / "knn_test.pkl")["X"],
                "y_test": joblib.load(knn_split_dir / "knn_test.pkl")["y"],
            },
            "mice": {
                "X_train": joblib.load(mice_split_dir / "mice_train.pkl")["X"],
                "y_train": joblib.load(mice_split_dir / "mice_train.pkl")["y"],
                "X_val": joblib.load(mice_split_dir / "mice_val.pkl")["X"],
                "y_val": joblib.load(mice_split_dir / "mice_val.pkl")["y"],
                "X_test": joblib.load(mice_split_dir / "mice_test.pkl")["X"],
                "y_test": joblib.load(mice_split_dir / "mice_test.pkl")["y"],
            }
        }
        print("✅ Données splitées chargées.")

    except FileNotFoundError as e:
        print(f"❌ Erreur : {e}")
        print("Impossible de charger les données splitées. Assurez-vous d'avoir exécuté le Notebook 02.")
    except Exception as e:
        print(f"❌ Une erreur inattendue s'est produite lors du chargement des splits : {e}")


    # 2. Charger tous les pipelines optimisés (stockés par modèle/imputation)
    print("💾 Chargement de tous les pipelines optimisés...")
    models_notebook2_dir = paths["MODELS_DIR"] / "notebook2"

    model_dirs = [d for d in models_notebook2_dir.iterdir() if d.is_dir()]

    if not model_dirs:
        print("⚠️ Aucun dossier de modèle trouvé dans MODELS_DIR/notebook2. Exécutez le Notebook 02.")
    else:
        for model_dir in model_dirs:
            model_name = model_dir.name

            for imp in ["knn", "mice"]:
                pipeline_file = next(model_dir.glob(f"pipeline_{model_name}_*{imp}.joblib"), None)
                threshold_file = next(model_dir.glob(f"threshold_{model_name}_*{imp}.json"), None)

                if pipeline_file and threshold_file:
                    try:
                        dict_key = f"{model_name}_{imp}"
                        all_optimized_pipelines[dict_key] = joblib.load(pipeline_file)
                        with open(threshold_file, "r") as f:
                             all_thresholds[dict_key] = json.load(f)

                        print(f"✅ Pipeline et seuil chargés pour {model_name.capitalize()} ({imp.upper()}).")
                    except Exception as e:
                        print(f"❌ Erreur lors du chargement de {model_name.capitalize()} ({imp.upper()}) : {e}")
                elif pipeline_file:
                     print(f"⚠️ Pipeline trouvé pour {model_name.capitalize()} ({imp.upper()}) mais fichier de seuil manquant : {threshold_file}")
                elif threshold_file:
                     print(f"⚠️ Fichier de seuil trouvé pour {model_name.capitalize()} ({imp.upper()}) mais pipeline manquant : {pipeline_file}")


    print(f"✅ {len(all_optimized_pipelines)} pipelines optimisés et leurs seuils chargés.")
    if not all_optimized_pipelines:
        print("⚠️ Aucun pipeline n'a pu être chargé. Vérifiez le dossier MODELS_DIR/notebook2 et les noms de fichiers.")


    # 3. Charger le tableau récapitulatif des seuils optimaux
    print("💾 Chargement du tableau des seuils optimaux...")
    thresholds_csv_path = models_notebook2_dir / "df_all_thresholds.csv"
    try:
        df_all_thr = pd.read_csv(thresholds_csv_path)
        print("✅ Tableau des seuils optimaux chargé.")
    except FileNotFoundError:
        print(f"❌ Erreur : Fichier non trouvé : {thresholds_csv_path}")
        print("Impossible de charger le tableau des seuils. Assurez-vous d'avoir exécuté l'étape 6.4 du Notebook 02.")
    except Exception as e:
        print(f"❌ Une erreur inattendue s'est produite lors du chargement de {thresholds_csv_path} : {e}")


    # 4. Charger les feature columns utilisées pour l'entraînement
    print("💾 Chargement des feature columns...")
    knn_cols_path = models_notebook2_dir / "knn" / "columns_knn.pkl"
    mice_cols_path = models_notebook2_dir / "mice" / "columns_mice.pkl"

    try:
        feature_cols_knn = joblib.load(knn_cols_path)
        feature_cols_mice = joblib.load(mice_cols_path)

        if feature_cols_knn != feature_cols_mice:
            print("ℹ️ INFO : Les feature columns chargées pour KNN et MICE sont différentes !")
            print(f"  KNN ({len(feature_cols_knn)} cols) : {feature_cols_knn[:5]}...")
            print(f"  MICE ({len(feature_cols_mice)} cols) : {feature_cols_mice[:5]}...")
            print("  ✅ Conservation des deux ensembles de colonnes.")
        else:
            print("✅ Les feature columns KNN et MICE sont identiques.")
            
        # Garder les deux ensembles de colonnes
        feature_cols = {
            "knn": feature_cols_knn,
            "mice": feature_cols_mice
        }
        print(f"✅ Feature columns chargées : KNN ({len(feature_cols_knn)} cols), MICE ({len(feature_cols_mice)} cols).")

    except FileNotFoundError:
        print(f"❌ Erreur : Fichiers columns.pkl non trouvés dans {models_notebook2_dir}/knn ou {models_notebook2_dir}/mice.")
        print("Impossible de charger les feature columns. Assurez-vous d'avoir exécuté l'étape 2 du Notebook 02.")
    except Exception as e:
        print(f"❌ Une erreur inattendue s'est produite lors du chargement des feature columns : {e}")


    # Résumé des artefacts chargés
    print("✨ Résumé des artefacts chargés pour Notebook 03 :")
    print(f"   - Données splitées ('splits') : {'✅ Chargées' if splits is not None else '❌ Non disponibles'}")
    print(f"   - Pipelines optimisés ('all_optimized_pipelines') : ✅ {len(all_optimized_pipelines)} chargés")
    print(f"   - Seuils optimaux ('all_thresholds') : ✅ {len(all_thresholds)} chargés")
    print(f"   - Tableau récapitulatif des seuils ('df_all_thr') : {'✅ Chargé' if df_all_thr is not None else '❌ Non disponible'}")
    print(f"   - Feature columns ('feature_cols') : {'✅ Chargées (KNN: ' + str(len(feature_cols['knn'])) + ', MICE: ' + str(len(feature_cols['mice'])) + ')' if feature_cols is not None else '❌ Non disponibles'}")

    print("🚀 Le Notebook 03 est prêt à utiliser ces artefacts.")

    return splits, all_optimized_pipelines, all_thresholds, df_all_thr, feature_cols

