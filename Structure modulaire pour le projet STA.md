# Structure modulaire pour le projet STA211

## 📁 Organisation des modules

```
modules/
├── __init__.py
├── config/
│   ├── __init__.py
│   ├── paths_config.py          # Configuration des chemins
│   ├── project_config.py        # Configuration principale
│   └── display_config.py        # Configuration d'affichage
├── data/
│   ├── __init__.py
│   ├── loader.py               # Chargement des données
│   ├── validator.py            # Validation des données
│   └── preprocessor.py         # Prétraitement de base
├── eda/
│   ├── __init__.py
│   ├── exploratory.py          # Analyse exploratoire
│   ├── missing_values.py       # Analyse des valeurs manquantes
│   ├── outliers.py             # Détection des outliers
│   └── correlations.py         # Analyse des corrélations
├── preprocessing/
│   ├── __init__.py
│   ├── imputation.py           # Méthodes d'imputation (MICE, KNN)
│   ├── scaling.py              # Normalisation et standardisation
│   ├── feature_engineering.py  # Création de features
│   └── pipeline.py             # Pipelines de prétraitement
├── visualization/
│   ├── __init__.py
│   ├── base_plots.py           # Graphiques de base
│   ├── eda_plots.py            # Visualisations EDA
│   ├── model_plots.py          # Visualisations modèles
│   └── utils.py                # Utilitaires de visualisation
└── utils/
    ├── __init__.py
    ├── helpers.py              # Fonctions utilitaires
    ├── logging_config.py       # Configuration du logging
    └── performance.py          # Monitoring des performances
```

## 🔧 Modules prioritaires à créer

### 1. **modules/config/environment.py**
- Détection d'environnement (Colab/Local)
- Configuration des chemins
- Import des bibliothèques
- Configuration globale

### 2. **modules/data/loader.py**
- Chargement des datasets
- Validation des formats
- Aperçu des données

### 3. **modules/eda/analyzer.py**
- Analyse exploratoire complète
- Statistiques descriptives
- Distribution des variables

### 4. **modules/preprocessing/imputer.py**
- Imputation MICE et KNN
- Gestion des outliers
- Transformation des variables

### 5. **modules/visualization/plotter.py**
- Visualisations standardisées
- Graphiques EDA
- Exports des figures



## 🚀 Prochaines étapes

1. **Créer la structure des dossiers**
2. **Migrer le code existant vers les modules**
3. **Simplifier les cellules du notebook**
4. **Ajouter les imports modulaires**
5. **Documenter chaque module**



# =============================================================================
# 📊 STA211 - EDA & Prétraitement des Données
# =============================================================================

"""
Notebook principal pour l'analyse exploratoire et le prétraitement 
du dataset Internet Advertisements (Challenge STA211).

Objectif : Prédire si une image est une publicité (ad./noad.) 
Métrique cible : Score F1
"""

# Métadonnées du projet
PROJECT_INFO = {
    "name": "Projet STA 211: Internet Advertisements Classification",
    "author": "Abdoullatuf", 
    "version": "1.0",
    "target_metric": "F1-Score"
}

print(f"📋 {PROJECT_INFO['name']}")
print(f"👤 Auteur: {PROJECT_INFO['author']} | Version: {PROJECT_INFO['version']}")
print(f"🎯 Métrique cible: {PROJECT_INFO['target_metric']}")

# =============================================================================
# 🔧 Configuration de l'environnement
# =============================================================================

# Import du module de configuration
from modules.config.environment import setup_environment

# Configuration complète en une ligne
config, paths, metadata = setup_environment(
    project_info=PROJECT_INFO,
    random_state=42,
    install_packages=True
)

print("✅ Environnement configuré avec succès")

# =============================================================================
# 📦 Import des modules du projet
# =============================================================================

# Modules de données
from modules.data.loader import DataLoader
from modules.data.validator import DataValidator

# Modules d'analyse
from modules.eda.analyzer import EDAAnalyzer
from modules.eda.missing_values import MissingValueAnalyzer

# Modules de prétraitement  
from modules.preprocessing.imputer import AdvancedImputer
from modules.preprocessing.pipeline import PreprocessingPipeline

# Modules de visualisation
from modules.visualization.plotter import EDAPlotter

print("📦 Modules du projet importés")

# =============================================================================
# 📊 Chargement et aperçu des données
# =============================================================================

# Chargement des données
loader = DataLoader(paths['RAW_DATA_DIR'])
train_data, test_data = loader.load_datasets()

# Validation des données
validator = DataValidator()
validation_report = validator.validate_datasets(train_data, test_data)

print(f"📊 Données chargées: {train_data.shape[0]} train, {test_data.shape[0]} test")
print(f"🔍 Validation: {validation_report['status']}")

# =============================================================================
# 🔍 Analyse exploratoire des données
# =============================================================================

# Analyseur EDA
eda = EDAAnalyzer(config)
eda_results = eda.analyze(train_data)

# Analyse des valeurs manquantes
missing_analyzer = MissingValueAnalyzer()
missing_report = missing_analyzer.analyze(train_data)

# Visualisations EDA
plotter = EDAPlotter(config['VIZ_CONFIG'])
plotter.create_overview_plots(train_data, save_dir=paths['FIGURES_DIR'])

print("🔍 Analyse exploratoire terminée")

# =============================================================================
# ⚙️ Prétraitement des données
# =============================================================================

# Pipeline de prétraitement
preprocessor = PreprocessingPipeline(config['PIPELINE_CONFIG'])

# Application du prétraitement
processed_data = preprocessor.fit_transform(
    train_data, 
    test_data,
    save_intermediates=True,
    output_dir=paths['DATA_PROCESSED']
)

print("⚙️ Prétraitement terminé")

# =============================================================================
# 💾 Sauvegarde des résultats
# =============================================================================

# Sauvegarde des datasets finaux
final_datasets = {
    'train_mice': processed_data['mice']['train'],
    'test_mice': processed_data['mice']['test'],
    'train_knn': processed_data['knn']['train'], 
    'test_knn': processed_data['knn']['test']
}

for name, dataset in final_datasets.items():
    save_path = paths['DATA_PROCESSED'] / f'{name}_final.csv'
    dataset.to_csv(save_path, index=False)
    print(f"💾 {name}: {save_path.name}")

print("\n🎉 Notebook EDA & Prétraitement terminé avec succès!")
