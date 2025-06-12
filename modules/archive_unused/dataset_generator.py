# modules/preprocessing/dataset_generator.py

"""
Module pour la génération automatisée des datasets finaux.
Gère la création de multiples versions de datasets avec différentes 
configurations de preprocessing.
"""

import pandas as pd
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Union, Any
import logging

# Configuration du logging
logger = logging.getLogger(__name__)

class DatasetGenerator:
    """
    Générateur de datasets avec différentes configurations de preprocessing.
    """
    
    def __init__(self, data_processed_dir: Path, models_dir: Path, random_state: int = 42):
        """
        Initialise le générateur de datasets.
        
        Args:
            data_processed_dir: Répertoire de sauvegarde des données traitées
            models_dir: Répertoire de sauvegarde des modèles/transformateurs
            random_state: Graine aléatoire pour la reproductibilité
        """
        self.data_processed_dir = Path(data_processed_dir)
        self.models_dir = Path(models_dir)
        self.random_state = random_state
        
        # Création des répertoires si nécessaire
        self.data_processed_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration par défaut
        self.default_common_params = {
            "strategy": "mixed_mar_mcar",
            "mar_cols": ["X1_trans", "X2_trans", "X3_trans"],
            "mcar_cols": ["X4"],
            "correlation_threshold": 0.95,
            "processed_data_dir": self.data_processed_dir,
            "models_dir": self.models_dir,
            "display_info": True
        }
    
    def create_dataset_configs(self, file_path: Path, custom_params: Dict = None) -> Dict:
        """
        Crée les configurations pour tous les datasets à générer.
        
        Args:
            file_path: Chemin vers le fichier de données brutes
            custom_params: Paramètres personnalisés à fusionner
            
        Returns:
            Dict: Configurations pour chaque dataset
        """
        
        # Fusion avec les paramètres personnalisés
        common_params = self.default_common_params.copy()
        if custom_params:
            common_params.update(custom_params)
        
        common_params["file_path"] = file_path
        
        # Configurations spécifiques pour chaque combinaison
        configs = {
            "mice_with_outliers": {
                **common_params,
                "mar_method": "mice",
                "drop_outliers": False,
                "save_transformer": True,  # Premier appel, on sauvegarde
                "output_name": "final_dataset_mice_with_outliers"
            },
            "mice_no_outliers": {
                **common_params,
                "mar_method": "mice", 
                "drop_outliers": True,
                "save_transformer": False,  # Transformateur déjà sauvegardé
                "output_name": "final_dataset_mice_no_outliers"
            },
            "knn_with_outliers": {
                **common_params,
                "mar_method": "knn",
                "knn_k": 5,  # k optimal par défaut
                "drop_outliers": False,
                "save_transformer": False,
                "output_name": "final_dataset_knn_with_outliers"
            },
            "knn_no_outliers": {
                **common_params,
                "mar_method": "knn",
                "knn_k": 5,
                "drop_outliers": True, 
                "save_transformer": False,
                "output_name": "final_dataset_knn_no_outliers"
            }
        }
        
        return configs
    
    def generate_single_dataset(self, config_name: str, config: Dict, 
                               save_format: str = 'parquet') -> Dict:
        """
        Génère un seul dataset selon la configuration.
        
        Args:
            config_name: Nom du dataset
            config: Configuration du dataset
            save_format: Format de sauvegarde ('parquet', 'csv', 'both')
            
        Returns:
            Dict: Informations sur le dataset généré
        """
        
        # Import dynamique pour éviter les dépendances circulaires
        from preprocessing.final_preprocessing import prepare_final_dataset
        
        logger.info(f"Génération du dataset : {config_name}")
        
        try:
            # Extraction du nom de sortie
            output_name = config.pop('output_name')
            
            # Génération du dataset
            df = prepare_final_dataset(**config)
            
            # Sauvegarde selon le format choisi
            saved_paths = []
            file_sizes = {}
            
            if save_format in ['parquet', 'both']:
                parquet_path = self.data_processed_dir / f"{output_name}.parquet"
                df.to_parquet(parquet_path, index=False)
                saved_paths.append(parquet_path)
                
                size_mb = parquet_path.stat().st_size / (1024 * 1024)
                file_sizes['.parquet'] = f"{size_mb:.2f} MB"
                logger.info(f"Parquet sauvegardé : {parquet_path} ({size_mb:.2f} MB)")
                
            if save_format in ['csv', 'both']:
                csv_path = self.data_processed_dir / f"{output_name}.csv"
                df.to_csv(csv_path, index=False)
                saved_paths.append(csv_path)
                
                size_mb = csv_path.stat().st_size / (1024 * 1024)
                file_sizes['.csv'] = f"{size_mb:.2f} MB"
                logger.info(f"CSV sauvegardé : {csv_path} ({size_mb:.2f} MB)")
            
            # Retour des informations
            return {
                'dataframe': df,
                'paths': saved_paths,
                'primary_path': saved_paths[0],
                'shape': df.shape,
                'missing_values': df.isnull().sum().sum(),
                'file_sizes': file_sizes,
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération de {config_name} : {e}")
            return {
                'dataframe': None,
                'paths': [],
                'primary_path': None,
                'shape': (0, 0),
                'missing_values': 0,
                'file_sizes': {},
                'status': 'error',
                'error_message': str(e)
            }
    
    def generate_all_datasets(self, file_path: Path, save_format: str = 'parquet',
                             custom_configs: Dict = None) -> Dict:
        """
        Génère tous les datasets selon les configurations.
        
        Args:
            file_path: Chemin vers le fichier de données brutes
            save_format: Format de sauvegarde ('parquet', 'csv', 'both')
            custom_configs: Configurations personnalisées (optionnel)
            
        Returns:
            Dict: Dictionnaire contenant tous les datasets générés
        """
        
        # Validation du format
        valid_formats = ['parquet', 'csv', 'both']
        if save_format not in valid_formats:
            raise ValueError(f"Format non supporté : {save_format}. Utilisez {valid_formats}")
        
        # Création des configurations
        if custom_configs is None:
            configs = self.create_dataset_configs(file_path)
        else:
            configs = custom_configs
        
        logger.info(f"Génération de {len(configs)} datasets en format {save_format}")
        
        datasets = {}
        
        for config_name, config in configs.items():
            print(f"\n📊 Génération du dataset : {config_name}")
            print("-" * 50)
            
            result = self.generate_single_dataset(config_name, config, save_format)
            
            if result['status'] == 'success':
                print(f"✅ Dataset généré avec succès")
                print(f"   📏 Dimensions : {result['shape']}")
                print(f"   🔍 Variables manquantes : {result['missing_values']}")
                
                for path in result['paths']:
                    ext = path.suffix
                    size = result['file_sizes'].get(ext, 'N/A')
                    print(f"   💾 {path.name} : {size}")
            else:
                print(f"❌ Erreur : {result['error_message']}")
            
            datasets[config_name] = result
        
        return datasets
    
    def create_summary_table(self, datasets: Dict) -> pd.DataFrame:
        """
        Crée un tableau récapitulatif des datasets générés.
        
        Args:
            datasets: Dictionnaire des datasets générés
            
        Returns:
            pd.DataFrame: Tableau récapitulatif
        """
        
        summary_data = []
        
        for name, info in datasets.items():
            if info['status'] == 'success':
                # Gestion des chemins multiples
                if len(info['paths']) > 1:
                    files_info = ' + '.join([p.name for p in info['paths']])
                    sizes_info = ' + '.join([f"{ext}: {size}" for ext, size in info['file_sizes'].items()])
                else:
                    files_info = info['primary_path'].name if info['primary_path'] else 'N/A'
                    sizes_info = list(info['file_sizes'].values())[0] if info['file_sizes'] else "N/A"
                
                summary_data.append({
                    'Dataset': name,
                    'Statut': '✅ Succès',
                    'Lignes': info['shape'][0],
                    'Colonnes': info['shape'][1], 
                    'Valeurs manquantes': info['missing_values'],
                    'Fichier(s)': files_info,
                    'Taille': sizes_info
                })
            else:
                summary_data.append({
                    'Dataset': name,
                    'Statut': '❌ Erreur',
                    'Lignes': 0,
                    'Colonnes': 0,
                    'Valeurs manquantes': 0,
                    'Fichier(s)': 'N/A',
                    'Taille': 'N/A'
                })
        
        return pd.DataFrame(summary_data)
    
    def validate_datasets_quality(self, datasets: Dict) -> Dict:
        """
        Valide la qualité des datasets générés.
        
        Args:
            datasets: Dictionnaire des datasets générés
            
        Returns:
            Dict: Rapport de validation
        """
        
        validation_report = {}
        
        for name, info in datasets.items():
            if info['status'] != 'success':
                validation_report[name] = {'status': 'error', 'checks': {}}
                continue
                
            df = info['dataframe']
            checks = {}
            
            # Vérifications de base
            checks['has_target'] = 'y' in df.columns
            checks['no_missing_values'] = info['missing_values'] == 0
            checks['has_transformed_vars'] = any('_trans' in col for col in df.columns)
            checks['target_distribution'] = dict(df['y'].value_counts()) if 'y' in df.columns else None
            checks['data_types'] = dict(df.dtypes.value_counts())
            
            # Score de qualité global
            quality_score = sum([
                checks['has_target'],
                checks['no_missing_values'],
                checks['has_transformed_vars']
            ]) / 3
            
            validation_report[name] = {
                'status': 'success',
                'quality_score': quality_score,
                'checks': checks
            }
        
        return validation_report
    
    def save_generation_config(self, datasets: Dict, save_format: str, 
                              output_path: Path = None) -> Path:
        """
        Sauvegarde la configuration de génération.
        
        Args:
            datasets: Datasets générés
            save_format: Format utilisé
            output_path: Chemin de sauvegarde (optionnel)
            
        Returns:
            Path: Chemin du fichier de configuration
        """
        
        if output_path is None:
            output_path = self.data_processed_dir / "dataset_generation_config.json"
        
        config_summary = {
            "generation_date": datetime.now().isoformat(),
            "save_format": save_format,
            "random_state": self.random_state,
            "datasets_generated": list(datasets.keys()),
            "successful_datasets": [
                name for name, info in datasets.items() 
                if info['status'] == 'success'
            ],
            "common_parameters": {
                "strategy": "mixed_mar_mcar",
                "correlation_threshold": 0.95,
                "random_state": self.random_state
            },
            "specific_configs": {
                name: {
                    "method": "mice" if "mice" in name else "knn",
                    "outliers": "with" if "with" in name else "without",
                    "k_value": 5 if "knn" in name else None,
                    "status": info['status']
                }
                for name, info in datasets.items()
            },
            "file_info": {
                name: {
                    "paths": [str(p) for p in info['paths']],
                    "file_sizes": info['file_sizes'],
                    "shape": info['shape']
                }
                for name, info in datasets.items()
                if info['status'] == 'success'
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config_summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Configuration sauvegardée : {output_path}")
        return output_path


# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

def quick_generate_datasets(file_path: Path, data_processed_dir: Path, 
                           models_dir: Path, save_format: str = 'parquet',
                           random_state: int = 42) -> Dict:
    """
    Fonction rapide pour générer tous les datasets avec les paramètres par défaut.
    
    Args:
        file_path: Chemin vers le fichier de données brutes
        data_processed_dir: Répertoire de sauvegarde des données
        models_dir: Répertoire des modèles
        save_format: Format de sauvegarde
        random_state: Graine aléatoire
        
    Returns:
        Dict: Datasets générés
    """
    
    generator = DatasetGenerator(data_processed_dir, models_dir, random_state)
    return generator.generate_all_datasets(file_path, save_format)

def print_generation_summary(datasets: Dict):
    """
    Affiche un résumé de la génération des datasets.
    
    Args:
        datasets: Dictionnaire des datasets générés
    """
    
    print("\n" + "=" * 70)
    print("📋 RÉSUMÉ DES DATASETS FINAUX GÉNÉRÉS")
    print("=" * 70)
    
    # Statistiques globales
    successful = sum(1 for info in datasets.values() if info['status'] == 'success')
    failed = len(datasets) - successful
    
    print(f"✅ Datasets générés avec succès : {successful}/{len(datasets)}")
    if failed > 0:
        print(f"❌ Échecs : {failed}")
    
    # Tableau détaillé
    generator = DatasetGenerator(Path('.'), Path('.'))  # Dummy pour utiliser la méthode
    summary_df = generator.create_summary_table(datasets)
    print(f"\n{summary_df.to_string(index=False)}")
    
    return summary_df