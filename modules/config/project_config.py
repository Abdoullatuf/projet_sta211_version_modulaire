# project_config.py
"""
Module de configuration centralisé pour le projet STA211.
Contient tous les paramètres et configurations du projet.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

# Import des utilitaires nécessaires
try:
    from sklearn.impute import KNNImputer, IterativeImputer
    from sklearn.preprocessing import StandardScaler
except ImportError:
    print("⚠️ Scikit-learn non disponible pour la configuration des pipelines")

class ProjectConfig:
    """Classe de configuration centralisée pour le projet STA211."""
    
    def __init__(self, project_name: str, version: str, author: str, random_state: int = 42):
        """
        Initialise la configuration du projet.
        
        Args:
            project_name: Nom du projet
            version: Version du projet
            author: Auteur du projet
            random_state: Seed pour la reproductibilité
        """
        self.project_name = project_name
        self.version = version
        self.author = author
        self.random_state = random_state
        
        # Initialisation des configurations
        self._init_project_config()
        self._init_column_config()
        self._init_pipeline_config()
        self._init_viz_config()
        self._init_model_config()
        
    def _init_project_config(self):
        """Initialise les paramètres généraux du projet."""
        self.PROJECT_CONFIG = {
            # Informations du projet
            "PROJECT_NAME": self.project_name,
            "VERSION": self.version,
            "AUTHOR": self.author,
            
            # Paramètres de données
            "TEST_SIZE": 0.2,
            "VALIDATION_SIZE": 0.2,
            "STRATIFY": True,
            
            # Paramètres de prétraitement
            "MISSING_THRESHOLD": 0.5,  # Seuil pour supprimer les colonnes
            "OUTLIER_THRESHOLD": 3,    # Z-score pour la détection des outliers
            "CORRELATION_THRESHOLD": 0.95,  # Seuil de corrélation
            
            # Paramètres d'imputation
            "IMPUTATION_METHODS": {
                "X4": "median",  # Imputation simple pour X4
                "others": ["knn", "mice"]  # Méthodes multivariées
            },
            "KNN_NEIGHBORS": 5,
            "MICE_MAX_ITER": 10,
            
            # Paramètres de modélisation
            "CV_FOLDS": 5,
            "SCORING": "f1",  # Métrique principale pour le challenge
            "PRIMARY_METRIC": "f1",
            "SCORING_METRICS": ["f1", "roc_auc", "precision", "recall", "accuracy"],
            "N_JOBS": -1,
            
            # Paramètres de sauvegarde
            "SAVE_INTERMEDIATE": True,
            "FIGURE_FORMAT": "png",
            "FIGURE_DPI": 300,
            
            # Seed pour la reproductibilité
            "RANDOM_STATE": self.random_state
        }
    
    def _init_column_config(self):
        """Initialise la configuration des colonnes."""
        self.COLUMN_CONFIG = {
            # Colonnes attendues dans le dataset
            "FEATURE_COLS": [f"X{i}" for i in range(1, 1559)],  # X1 à X1558
            "TARGET_COL": "y",
            
            # Colonnes spéciales
            "CONTINUOUS_COLS": ["X1", "X2", "X3"],  # Pour transformation Yeo-Johnson
            "BINARY_COLS": [],  # À identifier lors de l'EDA
            
            # Types de données attendus
            "DTYPES": {
                "X1": "float64",
                "X2": "float64", 
                "X3": "float64",
                "y": "int64"
            }
        }
    
    def _init_pipeline_config(self):
        """Initialise la configuration des pipelines."""
        try:
            self.PIPELINE_CONFIG = {
                # Pipeline KNN
                "knn_pipeline": {
                    "imputer": KNNImputer(n_neighbors=self.PROJECT_CONFIG["KNN_NEIGHBORS"]),
                    "scaler": StandardScaler(),
                    "name": "KNN_Imputation"
                },
                
                # Pipeline MICE
                "mice_pipeline": {
                    "imputer": IterativeImputer(
                        max_iter=self.PROJECT_CONFIG["MICE_MAX_ITER"], 
                        random_state=self.PROJECT_CONFIG["RANDOM_STATE"]
                    ),
                    "scaler": StandardScaler(),
                    "name": "MICE_Imputation"
                }
            }
        except:
            # Configuration simple si sklearn n'est pas disponible
            self.PIPELINE_CONFIG = {
                "knn_pipeline": {
                    "name": "KNN_Imputation",
                    "n_neighbors": self.PROJECT_CONFIG["KNN_NEIGHBORS"]
                },
                "mice_pipeline": {
                    "name": "MICE_Imputation",
                    "max_iter": self.PROJECT_CONFIG["MICE_MAX_ITER"]
                }
            }
    
    def _init_viz_config(self):
        """Initialise la configuration des visualisations."""
        self.VIZ_CONFIG = {
            "color_palette": "husl",
            "figure_size": (12, 8),
            "subplot_size": (15, 10),
            "heatmap_size": (20, 16),
            "distribution_bins": 50,
            "correlation_method": "pearson",
            "missing_values_color": "yellow",
            "outlier_color": "red",
            "class_colors": {
                0: "#3498db",  # Bleu pour classe 0
                1: "#e74c3c"   # Rouge pour classe 1
            }
        }
    
    def _init_model_config(self):
        """Initialise la configuration des modèles."""
        self.MODEL_CONFIG = {
            # Hyperparamètres par défaut optimisés pour F1-score
            "logistic_regression": {
                "C": [0.001, 0.01, 0.1, 1, 10, 100],
                "penalty": ["l1", "l2"],
                "solver": ["liblinear", "saga"],
                "max_iter": 1000,
                "class_weight": ["balanced", None]  # Important pour F1
            },
            "random_forest": {
                "n_estimators": [100, 200, 300],
                "max_depth": [10, 20, 30, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "class_weight": ["balanced", "balanced_subsample", None]  # Important pour F1
            },
            "xgboost": {
                "n_estimators": [100, 200, 300],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.1, 0.3],
                "subsample": [0.7, 0.8, 0.9],
                "scale_pos_weight": [1, 2, 3]  # Important pour déséquilibre des classes
            },
            "svm": {
                "C": [0.1, 1, 10],
                "kernel": ["rbf", "linear"],
                "gamma": ["scale", "auto"],
                "class_weight": ["balanced", None]  # Important pour F1
            }
        }
        
        # Configuration spécifique pour l'optimisation F1
        self.F1_OPTIMIZATION = {
            "threshold_search": True,  # Recherche du seuil optimal
            "threshold_range": (0.1, 0.9),  # Plage de recherche
            "threshold_step": 0.05,  # Pas de recherche
            "use_class_weight": True,  # Utiliser class_weight='balanced'
            "smote_ratio": "auto",  # Ratio pour SMOTE
            "ensemble_voting": "soft"  # Voting pour les ensembles
        }
    
    def init_save_paths(self, paths: Dict[str, Path]):
        """
        Initialise les chemins de sauvegarde avec les paths du projet.
        
        Args:
            paths: Dictionnaire des chemins du projet (depuis project_setup)
        """
        ROOT_DIR = Path(paths["ROOT_DIR"])
        DATA_PROCESSED = Path(paths["DATA_PROCESSED"])
        MODELS_DIR = Path(paths["MODELS_DIR"])
        FIGURES_DIR = Path(paths["FIGURES_DIR"])
        
        self.SAVE_PATHS = {
            # Données
            "data_knn": DATA_PROCESSED / f"{self.project_name}_knn_imputed_v{self.version}.csv",
            "data_mice": DATA_PROCESSED / f"{self.project_name}_mice_imputed_v{self.version}.csv",
            "data_train_knn": DATA_PROCESSED / f"{self.project_name}_train_knn_v{self.version}.csv",
            "data_test_knn": DATA_PROCESSED / f"{self.project_name}_test_knn_v{self.version}.csv",
            "data_train_mice": DATA_PROCESSED / f"{self.project_name}_train_mice_v{self.version}.csv",
            "data_test_mice": DATA_PROCESSED / f"{self.project_name}_test_mice_v{self.version}.csv",
            
            # Rapports
            "eda_report": ROOT_DIR / "reports" / f"eda_report_v{self.version}.html",
            "preprocessing_report": ROOT_DIR / "reports" / f"preprocessing_report_v{self.version}.txt",
            
            # Modèles
            "model_baseline": MODELS_DIR / f"baseline_model_v{self.version}.pkl",
            "model_best": MODELS_DIR / f"best_model_v{self.version}.pkl",
            
            # Figures
            "fig_missing_values": FIGURES_DIR / "eda" / "missing_values_heatmap.png",
            "fig_distributions": FIGURES_DIR / "eda" / "feature_distributions.png",
            "fig_correlations": FIGURES_DIR / "eda" / "correlation_matrix.png",
            "fig_outliers": FIGURES_DIR / "preprocessing" / "outliers_detection.png",
            "fig_model_comparison": FIGURES_DIR / "modeling" / "model_comparison.png"
        }
        
        # Créer les sous-dossiers pour les figures
        for path in ["eda", "preprocessing", "modeling"]:
            (FIGURES_DIR / path).mkdir(parents=True, exist_ok=True)
    
    def get(self, key: str, default=None) -> Any:
        """
        Récupère une valeur de configuration.
        
        Args:
            key: Clé de configuration (format: "SECTION.KEY")
            default: Valeur par défaut si la clé n'existe pas
            
        Returns:
            La valeur de configuration
        """
        sections = key.split(".")
        config = self
        
        try:
            for section in sections:
                if hasattr(config, section):
                    config = getattr(config, section)
                else:
                    config = config[section]
            return config
        except (KeyError, AttributeError):
            return default
    
    def update(self, key: str, value: Any) -> None:
        """
        Met à jour une valeur de configuration.
        
        Args:
            key: Clé de configuration (format: "SECTION.KEY")
            value: Nouvelle valeur
        """
        sections = key.split(".")
        config_dict = getattr(self, sections[0])
        
        if len(sections) == 2:
            config_dict[sections[1]] = value
        elif len(sections) == 3:
            config_dict[sections[1]][sections[2]] = value
        
        print(f"✅ Configuration mise à jour : {key} = {value}")
    
    def save_config(self, filepath: Path) -> None:
        """
        Sauvegarde la configuration dans un fichier JSON.
        
        Args:
            filepath: Chemin du fichier de sauvegarde
        """
        config_to_save = {
            "PROJECT_CONFIG": self.PROJECT_CONFIG,
            "COLUMN_CONFIG": self.COLUMN_CONFIG,
            "VIZ_CONFIG": self.VIZ_CONFIG,
            "MODEL_CONFIG": self.MODEL_CONFIG,
            "PIPELINE_CONFIG": {
                k: {"name": v["name"]} for k, v in self.PIPELINE_CONFIG.items()
            },
            "SAVE_PATHS": {k: str(v) for k, v in self.SAVE_PATHS.items()} if hasattr(self, 'SAVE_PATHS') else {}
        }
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(config_to_save, f, indent=4)
        
        print(f"✅ Configuration sauvegardée dans : {filepath}")
    
    def load_config(self, filepath: Path) -> None:
        """
        Charge la configuration depuis un fichier JSON.
        
        Args:
            filepath: Chemin du fichier à charger
        """
        with open(filepath, 'r') as f:
            config = json.load(f)
        
        self.PROJECT_CONFIG.update(config.get("PROJECT_CONFIG", {}))
        self.COLUMN_CONFIG.update(config.get("COLUMN_CONFIG", {}))
        self.VIZ_CONFIG.update(config.get("VIZ_CONFIG", {}))
        self.MODEL_CONFIG.update(config.get("MODEL_CONFIG", {}))
        
        print(f"✅ Configuration chargée depuis : {filepath}")
    
    def display_config(self) -> None:
        """Affiche un résumé de la configuration."""
        print("📋 Configuration du projet STA211")
        print("="*50)
        print(f"📊 Projet : {self.PROJECT_CONFIG['PROJECT_NAME']} v{self.PROJECT_CONFIG['VERSION']}")
        print(f"👤 Auteur : {self.PROJECT_CONFIG['AUTHOR']}")
        print(f"🎲 Random State : {self.PROJECT_CONFIG['RANDOM_STATE']}")
        print("\n🔧 Paramètres clés :")
        print(f"  - Test size : {self.PROJECT_CONFIG['TEST_SIZE']*100}%")
        print(f"  - CV folds : {self.PROJECT_CONFIG['CV_FOLDS']}")
        print(f"  - Métrique principale : {self.PROJECT_CONFIG['PRIMARY_METRIC']} ⭐")
        print(f"  - Scoring CV : {self.PROJECT_CONFIG['SCORING']}")
        print(f"  - Métriques suivies : {', '.join(self.PROJECT_CONFIG['SCORING_METRICS'])}")
        print(f"  - Seuil valeurs manquantes : {self.PROJECT_CONFIG['MISSING_THRESHOLD']*100}%")
        print(f"  - Seuil corrélation : {self.PROJECT_CONFIG['CORRELATION_THRESHOLD']}")
        print("\n💾 Méthodes d'imputation :")
        print(f"  - X4 : {self.PROJECT_CONFIG['IMPUTATION_METHODS']['X4']}")
        print(f"  - Autres : {', '.join(self.PROJECT_CONFIG['IMPUTATION_METHODS']['others'])}")
        print("="*50)


# Fonction de création rapide pour compatibilité
def create_config(project_name: str, version: str, author: str, paths: Dict[str, Path]) -> ProjectConfig:
    """
    Crée et initialise une configuration complète.
    
    Args:
        project_name: Nom du projet
        version: Version du projet
        author: Auteur du projet
        paths: Dictionnaire des chemins (depuis project_setup)
        
    Returns:
        Instance de ProjectConfig configurée
    """
    config = ProjectConfig(project_name, version, author)
    config.init_save_paths(paths)
    return config