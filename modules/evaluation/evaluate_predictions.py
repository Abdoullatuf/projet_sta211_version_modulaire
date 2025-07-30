#modules/evaluation/avaluate_predictions


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report, confusion_matrix

def evaluate_predictions(y_true, y_pred, label=""):
    """
    Affiche les métriques principales, le rapport de classification et une jolie matrice de confusion.
    
    Paramètres :
    ------------
    y_true : array-like
        Vraies étiquettes.
    y_pred : array-like
        Prédictions binaires.
    label : str
        Titre personnalisé pour l'affichage (ex: "VAL KNN", "TEST MICE").
    """
    # Calcul des métriques
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    print(f"\n=== ÉVALUATION : {label.upper()} ===")
    print(f"F1-score   : {f1:.4f}")
    print(f"Précision  : {precision:.4f}")
    print(f"Rappel     : {recall:.4f}")
    print("\n--- Rapport détaillé ---")
    print(classification_report(y_true, y_pred, digits=4))

    # Matrice de confusion
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(3.5, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Prédit: No-AD", "Prédit: AD"],
                yticklabels=["Réel: No-AD", "Réel: AD"])
    plt.title(f"Matrice de Confusion – {label.upper()}", fontsize=11)
    plt.xlabel("Valeurs Prédites")
    plt.ylabel("Valeurs Réelles")
    plt.tight_layout()
    plt.grid(False)
    plt.show()


def evaluate_from_probabilities(y_true, y_proba, threshold=0.5, label=""):
    """
    Évalue les performances à partir des probabilités prédites avec un seuil donné.
    
    Paramètres :
    ------------
    y_true : array-like
        Vraies étiquettes.
    y_proba : array-like
        Probabilités prédites pour la classe positive.
    threshold : float
        Seuil de décision à appliquer sur y_proba.
    label : str
        Titre de l’évaluation (ex: "TEST KNN", "VALIDATION MICE").
    """
    y_pred = (np.array(y_proba) >= threshold).astype(int)

    # 📊 Métriques
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    print(f"\n=== ÉVALUATION : {label.upper()} ===")
    print(f"Seuil appliqué : {threshold:.3f}")
    print(f"F1-score       : {f1:.4f}")
    print(f"Précision      : {precision:.4f}")
    print(f"Rappel         : {recall:.4f}")
    print("\n--- Rapport détaillé ---")
    print(classification_report(y_true, y_pred, digits=4))

    # 🔍 Matrice de confusion
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(3.5, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Prédit: No-AD", "Prédit: AD"],
                yticklabels=["Réel: No-AD", "Réel: AD"])
    plt.title(f"Matrice de Confusion – {label.upper()}", fontsize=11)
    plt.xlabel("Valeurs Prédites")
    plt.ylabel("Valeurs Réelles")
    plt.tight_layout()
    plt.grid(False)
    plt.show()

    # Optionnel : retourner les scores pour logs
    return {
        "threshold": threshold,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }


def analyze_model_performance(result_dict, X_test, y_test, method_name):
    """
    Analyse les performances d'un modèle à partir du dictionnaire de résultats.
    """
    if result_dict is None:
        print(f"❌ Aucun résultat disponible pour {method_name}.")
        return

    model = result_dict['model']
    threshold = result_dict['threshold']
    
    if model is None:
        print(f"❌ Aucun modèle disponible dans les résultats pour {method_name}.")
        return

    # a. Prédictions
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    # b. Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n--- Matrice de Confusion - {method_name} ---")
    print(cm)

    # c. Classification Report
    print(f"\n--- Rapport de Classification - {method_name} ---")
    print(classification_report(y_test, y_pred))

    # d. Visualisation de la matrice de confusion (optionnel)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["Non-admis", "Admis"], 
                yticklabels=["Non-admis (Réel)", "Admis (Réel)"])
    plt.title(f'Matrice de Confusion - {method_name}\n(Seuil: {threshold:.3f}, F1-test: {result_dict["metrics"]["f1_score_test"]:.4f})')
    plt.xlabel('Prédit')
    plt.ylabel('Réel')
    plt.tight_layout()
    plt.show()

    # e. Afficher les métriques déjà calculées
    print(f"\n--- Métriques Test (Seuil Optimal {threshold:.3f}) - {method_name} ---")
    metrics = result_dict['metrics']
    print(f"F1-score (test): {metrics['f1_score_test']:.4f}")
    print(f"Précision (test): {metrics['precision_test']:.4f}")
    print(f"Rappel (test): {metrics['recall_test']:.4f}")