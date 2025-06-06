# 🧠 Projet STA211 — Détection de Publicités dans les Images

Ce projet a été réalisé dans le cadre du cours **STA211** du **CNAM**, dont l'objectif est de mettre en œuvre des méthodes exploratoires, de prétraitement et de modélisation supervisée pour résoudre un problème de classification binaire : déterminer si une image est une publicité (`ad.`) ou non (`noad.`).

Ce dépôt contient l'ensemble du code permettant de tester différentes approches de prétraitement et de modélisation sur le jeu de données **Internet Advertisements Dataset** provenant de l'[UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/internet+advertisements).

## Table des matières
- [Installation](#installation)
- [Jeu de données](#jeu-de-données)
- [Structure du dépôt](#structure-du-dépôt)
- [Utilisation basique](#utilisation-basique)
- [Lancer les tests](#lancer-les-tests)
- [Licence](#licence)
- [Crédits](#crédits)

## Installation

1. Utiliser **Python 3.11** ou plus récent.
2. Créer un environnement virtuel :
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
3. Installer les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

## Jeu de données

Le projet s'appuie sur le **Internet Advertisements Dataset** disponible sur le [UCI Repository](https://archive.ics.uci.edu/ml/datasets/internet+advertisements). Téléchargez les fichiers `ad.data` et `ad.names` depuis la page UCI puis placez-les dans un dossier de votre choix avant d'exécuter les scripts de prétraitement.

## Structure du dépôt

```
modules/    # Modules Python pour le prétraitement et la modélisation
notebooks/  # Notebooks d'exploration
tests/      # Scripts de tests dont le pipeline complet
outputs/    # Figures et artefacts générés
```

Les dossiers `config/` et `results/` contiennent respectivement les fichiers de configuration et certains résultats sauvegardés.

## Utilisation basique

Pour lancer la préparation des données et tester la chaîne de modélisation, exécutez :

```bash
python tests/test_pipeline.py
```

L'appel sans argument effectue une démonstration rapide sur un jeu de données factice. En passant l'argument `complet`, le pipeline Optuna complet est exécuté sur ces données factices.

## Lancer les tests

Les tests unitaires peuvent être lancés avec [pytest](https://pytest.readthedocs.io/) :

```bash
pytest
```

## Licence

Ce projet est distribué sous licence [MIT](LICENSE).

## Crédits

Projet développé dans le cadre du cours STA211 du CNAM par @ellatuf.
