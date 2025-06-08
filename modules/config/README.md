# 📁 Modules de Configuration - STA211

Ce dossier contient tous les modules de configuration pour le projet STA211. Ces modules permettent de simplifier drastiquement la configuration de l'environnement et du projet.

## 📚 Structure des modules

```
modules/config/
├── __init__.py              # Package principal
├── environment.py           # Configuration complète de l'environnement
├── paths_config.py         # Gestion des chemins et structure projet
├── project_config.py       # Configuration centralisée du projet
├── display_config.py       # Configuration affichage et visualisations
└── README.md               # Documentation (ce fichier)
```

## 🚀 Utilisation rapide

### Configuration complète en une ligne

```python
from modules.config.environment import setup_environment

# Informations du projet
project_info = {
    "name": "Projet STA 211: Internet Advertisements Classification",
    "author": "Abdoullatuf",
    "version": "1.0", 
    "target_metric": "F1-Score"
}

# Configuration complète
config, paths, metadata = setup_environment(
    project_info=project_info,
    random_state=42,
    install_packages=True
)

# ✅ Tout est configuré !
```

## 📋 Modules détaillés

### 1. `environment.py` - Configuration environnement

**Fonctionnalités principales :**
- Détection automatique Colab/Local
- Installation des packages requis
- Import des bibliothèques (core + optionnelles)
- Configuration graines aléatoires
- Configuration complète en une fonction

**Fonctions clés :**
```python
setup_environment()          # Configuration complète
detect_environment()         # Détection Colab/Local
install_required_packages()  # Installation packages
import_core_libraries()      # Import bibliothèques principales
setup_random_seeds()         # Configuration reproductibilité
```

### 2. `paths_config.py` - Gestion des chemins

**Fonctionnalités principales :**
- Configuration automatique des chemins selon l'environnement
- Création de la structure des dossiers
- Validation de l'existence des chemins
- Gestion des chemins de fichiers spécifiques

**Fonctions clés :**
```python
setup_project_paths()        # Configuration principale
create_project_structure()   # Création dossiers
get_data_files_paths()      # Chemins fichiers données
validate_paths()            # Validation existence
display_paths_summary()     # Affichage résumé
```

### 3. `project_config.py` - Configuration projet

**Fonctionnalités principales :**
- Configuration centralisée de tous les paramètres
- Gestion des métadonnées du projet
- Sauvegarde/chargement configuration
- Accès et modification faciles des paramètres

**Classe principale :**
```python
class ProjectConfig:
    def get(key)              # Accès valeur avec notation pointée
    def update(key, value)    # Mise à jour paramètre
    def display_config()      # Affichage configuration
    def save_config(path)     # Sauvegarde JSON
    def load_config(path)     # Chargement JSON
```

**Sections de configuration :**
- `PROJECT_CONFIG`: Paramètres généraux
- `COLUMN_CONFIG`: Configuration colonnes/variables
- `VIZ_CONFIG`: Paramètres visualisation
- `MODEL_CONFIG`: Configuration modèles ML
- `PIPELINE_CONFIG`: Configuration pipelines
- `F1_OPTIMIZATION`: Optimisation score F1

### 4. `display_config.py` - Configuration affichage

**Fonctionnalités principales :**
- Configuration matplotlib et seaborn
- Paramètres pandas optimisés
- Gestion warnings
- Palettes de couleurs personnalisées
- Styles Jupyter

**Fonctions clés :**
```python
setup_display_config()      # Configuration complète affichage
get_color_palette()         # Palettes couleurs projet
create_custom_style()       # Style personnalisé
setup_jupyter_display()     # Configuration Jupyter
```

## 🎯 Avantages de cette approche

### ✅ **Notebook simplifié**
- 3 cellules de configuration → 1 seule cellule
- Code technique caché dans les modules
- Focus sur l'analyse plutôt que la configuration

### ✅ **Reproductibilité**
- Configuration centralisée et versionnée
- Graines aléatoires gérées automatiquement
- Sauvegarde automatique des paramètres

### ✅ **Maintenabilité**
- Modifications centralisées dans les modules
- Code réutilisable pour d'autres projets
- Documentation intégrée

### ✅ **Robustesse**
- Gestion d'erreurs complète
- Validation automatique
- Fallbacks pour les imports optionnels

## 🛠️ Exemples d'usage

### Configuration de base
```python
from modules.config import setup_environment

# Configuration rapide
config, paths, metadata = setup_environment({
    "name": "Mon Projet",
    "author": "Mon Nom",
    "version": "1.0"
})
```

### Configuration personnalisée
```python
from modules.config.project_config import create_config
from modules.config.paths_config import setup_project_paths

# Chemins personnalisés
paths = setup_project_paths(env="local", create_structure=True)

# Configuration projet
config = create_config("Mon Projet", "1.0", "Auteur", paths)

# Personnalisation
config.update("PROJECT_CONFIG.SCORING", "accuracy")
config.update("VIZ_CONFIG.figure_size", (10, 6))
```

### Accès aux paramètres
```python
# Lecture
test_size = config.get("PROJECT_CONFIG.TEST_SIZE")
colors = config.get("VIZ_CONFIG.colors.primary")

# Modification
config.update("PROJECT_CONFIG.CV_FOLDS", 10)
config.update("VIZ_CONFIG.style", "darkgrid")

# Sauvegarde
config.save_config("ma_config.json")
```

## 🧪 Tests et validation

### Exécution des tests
```bash
# Test complet des modules
python test_config.py

# Test depuis notebook
from modules.config.environment import validate_environment_setup
is_valid = validate_environment_setup(config, paths, metadata)
```

### Validation manuelle
```python
# Vérification de la configuration
config.display_config()

# Vérification des chemins
from modules.config.paths_config import display_paths_summary
display_paths_summary(paths)

# Vérification des bibliothèques
from modules.config.environment import verify_environment
env_info = verify_environment()
```

## 🔧 Dépannage

### Problèmes courants

**1. Modules non trouvés**
```python
# Vérifier le PYTHONPATH
import sys
print(sys.path)

# Ajouter manuellement si nécessaire
sys.path.insert(0, "chemin/vers/modules")
```

**2. Erreurs d'import optionnels**
```python
# Les imports optionnels ne sont pas critiques
# Vérifier la disponibilité
from modules.config.environment import import_optional_libraries
opt_libs = import_optional_libraries()
print(opt_libs)
```

**3. Chemins non trouvés**
```python
# Forcer la création de la structure
from modules.config.paths_config import setup_project_paths
paths = setup_project_paths(create_structure=True)
```

### Configuration de débogage
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Configuration avec logs détaillés
config, paths, metadata = setup_environment(project_info, install_packages=True)
```

## 📄 Configuration par défaut

Le fichier `config/default_config.json` contient tous les paramètres par défaut du projet. Il peut servir de :
- Référence pour les paramètres disponibles
- Base pour créer des configurations personnalisées
- Sauvegarde de configuration

## 🔄 Migration depuis l'ancien code

### Avant (3 cellules)
```python
# Cellule 1: Configuration environnement (50+ lignes)
# Cellule 2: Import bibliothèques (40+ lignes)  
# Cellule 3: Configuration projet (30+ lignes)
```

### Après (1 cellule)
```python
from modules.config.environment import setup_environment

config, paths, metadata = setup_environment({
    "name": "Projet STA 211",
    "author": "Abdoullatuf",
    "version": "1.0",
    "target_metric": "F1-Score"
})
```

**Réduction : 120+ lignes → 10 lignes** 🎉

## 🤝 Contribution

Pour ajouter de nouvelles fonctionnalités :

1. **Modifier la configuration** : Ajuster `project_config.py`
2. **Ajouter des fonctions** : Étendre les modules existants
3. **Tester** : Utiliser `test_config.py`
4. **Documenter** : Mettre à jour ce README

---

*Ce système de configuration modulaire a été conçu pour simplifier et professionnaliser le workflow du projet STA211.*