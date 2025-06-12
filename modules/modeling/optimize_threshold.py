# modules/modeling/optimize_threshold.py

from sklearn.metrics import precision_recall_curve, classification_report
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

def sanitize_name(model_name: str, dataset_name: str) -> str:
    """
    Construit un nom propre à partir du modèle et du dataset,
    évite les doublons comme stacking_mice_mice → stacking_mice.
    """
    model_clean = model_name.strip().lower().replace(" ", "_")
    dataset_clean = dataset_name.strip().lower().replace(" ", "_")

    if dataset_clean in model_clean:
        return model_clean
    return f"{model_clean}_{dataset_clean}"

def optimize_threshold(model, X_test, y_test, model_name, dataset_name, figures_dir, json_dir):
    """
    Optimise le seuil de décision pour maximiser le F1-score.
    Sauvegarde la courbe F1/seuil et un fichier JSON contenant le seuil optimal.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, classification_report
    import json
    from pathlib import Path

    # Nom propre
    def sanitize_name(model_name, dataset_name):
        model_clean = model_name.strip().lower().replace(" ", "_")
        dataset_clean = dataset_name.strip().lower().replace(" ", "_")
        return f"{model_clean}_{dataset_clean}"

    clean_name = sanitize_name(model_name, dataset_name)

    # Prédiction des probabilités
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calcul des scores
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)
    f1_scores = np.where((precisions + recalls) == 0, 0, 2 * (precisions * recalls) / (precisions + recalls))
    optimal_index = np.argmax(f1_scores[:-1])
    optimal_threshold = thresholds[optimal_index]
    optimal_f1 = f1_scores[optimal_index]

    # === Figure
    fig_path = figures_dir / f"optim_seuil_{clean_name}.png"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 3.5))
    plt.plot(thresholds, f1_scores[:-1], label="F1-score", color="lightcoral")
    plt.axvline(optimal_threshold, color='red', linestyle='--', label=f"Seuil optimal = {optimal_threshold:.2f}")
    plt.title(f"Optimisation du seuil - {model_name} ({dataset_name})")
    plt.xlabel("Seuil de décision")
    plt.ylabel("F1-score")
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.show()

    # === JSON
    json_dir.mkdir(parents=True, exist_ok=True)
    json_path = json_dir / f"threshold_{clean_name}.json"
    with open(json_path, "w") as f:
        json.dump({
            "model": model_name,
            "dataset": dataset_name,
            "optimal_threshold": round(float(optimal_threshold), 4),
            "optimal_f1": round(float(optimal_f1), 4)
        }, f, indent=4)

    # Affichage résumé
    y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
    print(f"\n📌 Résultats pour {model_name} ({dataset_name})")
    print(f"Seuil optimal : {optimal_threshold:.4f}")
    print(f"F1-score optimal : {optimal_f1:.4f}")
    print("\nRapport de classification :")
    print(classification_report(y_test, y_pred_optimal))

    return optimal_threshold, optimal_f1, y_pred_optimal

def load_optimal_threshold(model_name, dataset_name, threshold_dir, silent=False):
    """
    Recharge un seuil optimal sauvegardé depuis un fichier JSON.

    Args:
        model_name (str): Nom du modèle (ex: "Stacking")
        dataset_name (str): Nom du dataset (ex: "MICE")
        threshold_dir (Path): Dossier contenant les fichiers JSON
        silent (bool): Si True, désactive les messages

    Returns:
        float | None: Seuil optimal, ou None si le fichier est introuvable
    """
    import json

    # Nettoyage des noms pour éviter les doublons comme stacking_mice_mice
    def sanitize_name(model_name, dataset_name):
        model_clean = model_name.strip().lower().replace(" ", "_")
        dataset_clean = dataset_name.strip().lower().replace(" ", "_")
        return f"{model_clean}_{dataset_clean}" if dataset_clean not in model_clean else model_clean

    clean_name = sanitize_name(model_name, dataset_name)
    json_path = threshold_dir / f"threshold_{clean_name}.json"

    if not json_path.exists():
        if not silent:
            print(f"❌ Fichier de seuil introuvable : {json_path}")
        return None

    with open(json_path, "r") as f:
        data = json.load(f)

    if not silent:
        print(f"✅ Seuil rechargé depuis : {json_path}")
        print(f"• Seuil : {data['optimal_threshold']}")
        print(f"• F1-score : {data['optimal_f1']}")

    return data["optimal_threshold"]

