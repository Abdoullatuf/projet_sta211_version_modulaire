import json
from pathlib import Path

def save_model_metrics(
    model_name: str,
    model_type: str,
    f1_score: float,
    precision: float,
    recall: float,
    threshold: float,
    save_path: Path,
    source: str = "csv"
):
    """
    Sauvegarde les métriques d’un modèle dans un fichier JSON.

    :param model_name: Nom du fichier JSON (sans extension)
    :param model_type: Type de modèle (ex. "Stacking", "XGBoost", etc.)
    :param f1_score: F1-score calculé sur le jeu de test
    :param precision: Précision sur le jeu de test
    :param recall: Rappel sur le jeu de test
    :param threshold: Seuil de classification optimal utilisé
    :param save_path: Répertoire dans lequel sauvegarder le fichier
    :param source: Source des données (ex: "csv", "parquet", etc.)
    """
    save_path.mkdir(parents=True, exist_ok=True)
    file_path = save_path / f"{model_name}.json"

    data = {
        "Model": model_name,
        "Type": model_type,
        "F1": round(f1_score, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "Threshold": round(threshold, 4),
        "Source": source
    }

    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"✅ Fichier sauvegardé : {file_path}")
