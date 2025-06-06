import os

IGNORER = {"__pycache__", ".ipynb_checkpoints", ".git", "data", ".venv", ".idea"}
EXTENSIONS_UTILES = {".py", ".ipynb", ".md", ".txt", ".csv"}

def afficher_structure(dossier, niveau=0):
    contenu = []
    for nom in sorted(os.listdir(dossier)):
        if nom in IGNORER:
            continue
        chemin = os.path.join(dossier, nom)
        if os.path.isdir(chemin):
            contenu.append("  " * niveau + f"📁 {nom}/")
            contenu.extend(afficher_structure(chemin, niveau + 1))
        elif os.path.isfile(chemin):
            ext = os.path.splitext(nom)[1]
            if ext in EXTENSIONS_UTILES:
                contenu.append("  " * niveau + f"📄 {nom}")
    return contenu

# Mets ici le chemin de ton projet
chemin_projet = "G:/Mon Drive/projet_sta211"
structure = afficher_structure(chemin_projet)
print("\n".join(structure))
