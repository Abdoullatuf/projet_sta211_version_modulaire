# modules/config/setup.py

"""
Module de configuration et d'initialisation pour le projet STA211.
Gère les imports, la configuration globale, et la validation de l'environnement.
"""

import sys
import os
import warnings
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

# ============================================================================
# CONFIGURATION DES IMPORTS
# ============================================================================

class ProjectSetup:
    """Classe pour configurer l'environnement du projet STA211."""
    
    def __init__(self, random_state=42, verbose=True):
        self.random_state = random_state
        self.verbose = verbose
        self.optional_packages = {}
        self.validation_results = {}
        self.imports_ok = False
        
    def setup_basic_imports(self):
        """Configure les imports de base."""
        global np, pd, plt, sns, stats
        
        try:
            # Manipulation des données
            import numpy as np
            import pandas as pd
            
            # Visualisation
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Statistiques
            from scipy import stats
            
            if self.verbose:
                print("📦 Imports de base chargés")
            
            return True
            
        except ImportError as e:
            if self.verbose:
                print(f"❌ Erreur imports de base : {e}")
            return False
    
    def setup_sklearn_imports(self):
        """Configure les imports scikit-learn."""
        try:
            # Import critique pour IterativeImputer
            from sklearn.experimental import enable_iterative_imputer
            
            # Prétraitement
            from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
            from sklearn.preprocessing import StandardScaler, PowerTransformer
            from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
            
            # Modèles essentiels
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.linear_model import LogisticRegression
            
            # Métriques
            from sklearn.metrics import f1_score, classification_report, roc_auc_score
            
            if self.verbose:
                print("⚙️ Imports scikit-learn chargés")
            
            return True
            
        except ImportError as e:
            if self.verbose:
                print(f"❌ Erreur imports scikit-learn : {e}")
            return False
    
    def check_optional_packages(self):
        """Vérifie les packages optionnels."""
        
        optional_checks = {
            'umap': lambda: __import__('umap.umap_', fromlist=['UMAP']),
            'imbalanced_learn': lambda: __import__('imblearn.over_sampling', fromlist=['SMOTE']),
            'xgboost': lambda: __import__('xgboost')
        }
        
        for name, import_func in optional_checks.items():
            try:
                module = import_func()
                self.optional_packages[name] = {'available': True, 'module': module}
                if self.verbose:
                    print(f"  ✅ {name} disponible")
            except ImportError:
                self.optional_packages[name] = {'available': False, 'module': None}
                if self.verbose:
                    print(f"  ⚠️ {name} non disponible")
        
        return self.optional_packages
    
    def configure_globals(self):
        """Configure les paramètres globaux."""
        
        # Warnings
        warnings.filterwarnings("ignore")
        
        # Pandas
        pd.set_option("display.max_columns", None)
        pd.set_option("display.max_rows", 100)
        pd.set_option("display.float_format", '{:.4f}'.format)
        
        # Matplotlib/Seaborn
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['font.size'] = 12
        sns.set_style("whitegrid")
        sns.set_palette("husl")
        
        # Reproductibilité
        import random
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        
        if self.verbose:
            print(f"🎲 Configuration globale appliquée (seed={self.random_state})")
    
    def validate_setup(self):
        """Valide que la configuration est correcte."""
        
        critical_modules = ['numpy', 'pandas', 'matplotlib', 'seaborn', 'sklearn']
        
        validation_ok = True
        
        for module_name in critical_modules:
            try:
                if module_name == 'sklearn':
                    import sklearn
                    version = sklearn.__version__
                elif module_name == 'numpy':
                    version = np.__version__
                elif module_name == 'pandas':
                    version = pd.__version__
                elif module_name == 'matplotlib':
                    version = plt.matplotlib.__version__
                elif module_name == 'seaborn':
                    version = sns.__version__
                
                self.validation_results[module_name] = {"status": "✅", "version": version}
                
                if self.verbose:
                    print(f"  ✅ {module_name} (v{version})")
                    
            except Exception as e:
                self.validation_results[module_name] = {"status": "❌", "error": str(e)}
                validation_ok = False
                
                if self.verbose:
                    print(f"  ❌ {module_name} - Erreur : {e}")
        
        self.imports_ok = validation_ok
        return validation_ok
    
    def setup_all(self):
        """Lance la configuration complète."""
        
        if self.verbose:
            print("🚀 Initialisation de l'environnement STA211...")
        
        # Étapes de configuration
        steps = [
            ("Imports de base", self.setup_basic_imports),
            ("Imports scikit-learn", self.setup_sklearn_imports),
            ("Configuration globale", lambda: (self.configure_globals(), True)[1]),
            ("Packages optionnels", lambda: (self.check_optional_packages(), True)[1]),
            ("Validation", self.validate_setup)
        ]
        
        for step_name, step_func in steps:
            if self.verbose:
                print(f"\n📋 {step_name}...")
                
            success = step_func()
            
            if not success and step_name in ["Imports de base", "Imports scikit-learn"]:
                if self.verbose:
                    print(f"❌ Échec critique à l'étape : {step_name}")
                return False
        
        if self.verbose:
            self.print_summary()
        
        return True
    
    def print_summary(self):
        """Affiche un résumé de la configuration."""
        
        print(f"\n📊 Résumé de la configuration :")
        print(f"  🐍 Python : {sys.version.split()[0]}")
        print(f"  🎲 Random State : {self.random_state}")
        
        # Packages optionnels
        available = [name for name, info in self.optional_packages.items() if info['available']]
        if available:
            print(f"  ✅ Packages optionnels : {', '.join(available)}")
        
        if self.imports_ok:
            print(f"\n✅ Configuration terminée avec succès")
        else:
            print(f"\n⚠️ Configuration terminée avec des avertissements")
    
    def get_exports(self):
        """Retourne les variables à exporter."""
        
        return {
            'RANDOM_STATE': self.random_state,
            'OPTIONAL_PACKAGES': self.optional_packages,
            'VALIDATION_RESULTS': self.validation_results,
            'IMPORTS_OK': self.imports_ok,
            # Modules principaux
            'np': np,
            'pd': pd, 
            'plt': plt,
            'sns': sns,
            'stats': stats
        }


# ============================================================================
# FONCTION RAPIDE POUR LE NOTEBOOK
# ============================================================================

def quick_setup(random_state=42, verbose=True):
    """
    Configuration rapide pour le notebook.
    
    Args:
        random_state: Graine aléatoire
        verbose: Affichage des détails
        
    Returns:
        dict: Variables exportées
    """
    
    setup = ProjectSetup(random_state=random_state, verbose=verbose)
    success = setup.setup_all()
    
    if success:
        return setup.get_exports()
    else:
        raise RuntimeError("Échec de la configuration de l'environnement")

def silent_setup(random_state=42):
    """Configuration silencieuse (pour tests)."""
    return quick_setup(random_state=random_state, verbose=False)


# ============================================================================
# UTILITAIRES SUPPLÉMENTAIRES
# ============================================================================

def check_environment_health():
    """Vérifie la santé de l'environnement."""
    
    health_checks = {
        'python_version': sys.version_info >= (3, 8),
        'memory_available': True,  # Placeholder
        'disk_space': True,  # Placeholder
    }
    
    return health_checks

def get_package_versions():
    """Retourne les versions des packages installés."""
    
    try:
        setup = ProjectSetup(verbose=False)
        setup.setup_all()
        return setup.validation_results
    except:
        return {}