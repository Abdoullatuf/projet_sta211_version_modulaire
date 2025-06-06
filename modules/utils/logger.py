# utils/logger.py
"""
Configuration du système de logging pour le projet STA211.
"""
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_dir: Path = Path("logs")
) -> logging.Logger:
    """
    Configure le système de logging.
    
    Parameters:
    -----------
    log_level : str
        Niveau de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    log_file : str, optional
        Nom du fichier de log. Si None, utilise la date/heure
    log_dir : Path
        Répertoire pour les logs
        
    Returns:
    --------
    logging.Logger : Logger configuré
    """
    # Créer le répertoire de logs
    log_dir.mkdir(exist_ok=True)
    
    # Nom du fichier de log
    if log_file is None:
        log_file = f"sta211_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    log_path = log_dir / log_file
    
    # Configuration du logger
    logger = logging.getLogger("STA211")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Handler pour fichier
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Handler pour console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging configuré - Niveau: {log_level}, Fichier: {log_path}")
    
    return logger

def get_logger(name: str = "STA211") -> logging.Logger:
    """
    Récupère un logger existant ou en crée un nouveau.
    
    Parameters:
    -----------
    name : str
        Nom du logger
        
    Returns:
    --------
    logging.Logger : Instance du logger
    """
    return logging.getLogger(name)

# Logger par défaut
logger = get_logger()