# modules/config/display_config.py
"""
Configuration de l'affichage et des visualisations pour le projet STA211

Ce module centralise tous les paramètres d'affichage :
- Configuration matplotlib et seaborn
- Paramètres pandas
- Thèmes et palettes de couleurs
- Configuration des notebooks
"""

import warnings
import logging
from typing import Dict, Optional, Tuple, Union

logger = logging.getLogger(__name__)

def setup_display_config(
    figure_size: Tuple[int, int] = (12, 8),
    dpi: int = 100,
    style: str = "whitegrid",
    palette: str = "husl",
    font_scale: float = 1.1,
    context: str = "notebook"
) -> Dict:
    """
    Configuration complète de l'affichage pour le projet
    
    Parameters:
    -----------
    figure_size : Tuple[int, int]
        Taille par défaut des figures (largeur, hauteur)
    dpi : int
        Résolution des figures
    style : str
        Style seaborn ('whitegrid', 'darkgrid', 'white', 'dark', 'ticks')
    palette : str
        Palette de couleurs seaborn
    font_scale : float
        Facteur d'échelle des polices
    context : str
        Contexte seaborn ('paper', 'notebook', 'talk', 'poster')
        
    Returns:
    --------
    Dict
        Configuration appliquée
    """
    
    logger.info("Configuration de l'affichage...")
    
    # Configuration des warnings
    setup_warnings()
    
    # Configuration pandas
    pandas_config = setup_pandas_display()
    
    # Configuration matplotlib
    matplotlib_config = setup_matplotlib(figure_size, dpi)
    
    # Configuration seaborn
    seaborn_config = setup_seaborn(style, palette, font_scale, context)
    
    # Configuration IPython/Jupyter
    jupyter_config = setup_jupyter_display()
    
    # Configuration consolidée
    display_config = {
        'pandas': pandas_config,
        'matplotlib': matplotlib_config,
        'seaborn': seaborn_config,
        'jupyter': jupyter_config,
        'warnings_configured': True
    }
    
    logger.info("Configuration de l'affichage terminée")
    return display_config

def setup_warnings():
    """Configure la gestion des warnings"""
    # Suppression des warnings non critiques
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    warnings.filterwarnings('ignore', message='.*IPython.*')
    warnings.filterwarnings('ignore', message='.*matplotlib.*')
    
    # Garder les warnings critiques
    warnings.filterwarnings('default', category=DeprecationWarning)
    warnings.filterwarnings('default', category=ImportWarning)

def setup_pandas_display() -> Dict:
    """
    Configure les options d'affichage de pandas
    
    Returns:
    --------
    Dict
        Configuration pandas appliquée
    """
    try:
        import pandas as pd
        
        # Options d'affichage générales
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', 100)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', 100)
        
        # Format des nombres
        pd.set_option('display.float_format', '{:.4f}'.format)
        pd.set_option('display.precision', 4)
        
        # Affichage des DataFrames
        pd.set_option('display.expand_frame_repr', False)
        pd.set_option('display.show_dimensions', True)
        
        # Options HTML pour Jupyter
        pd.set_option('display.html.border', 0)
        pd.set_option('display.html.table_schema', False)
        
        config = {
            'max_columns': None,
            'max_rows': 100,
            'float_format': '{:.4f}',
            'precision': 4,
            'expand_frame_repr': False
        }
        
        logger.debug("Configuration pandas appliquée")
        return config
        
    except ImportError:
        logger.warning("Pandas non disponible")
        return {}

def setup_matplotlib(
    figure_size: Tuple[int, int] = (12, 8),
    dpi: int = 100
) -> Dict:
    """
    Configure matplotlib avec les paramètres optimaux
    
    Parameters:
    -----------
    figure_size : Tuple[int, int]
        Taille par défaut des figures
    dpi : int
        Résolution des figures
        
    Returns:
    --------
    Dict
        Configuration matplotlib appliquée
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        
        # Taille et résolution des figures
        plt.rcParams['figure.figsize'] = figure_size
        plt.rcParams['figure.dpi'] = dpi
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['savefig.bbox'] = 'tight'
        plt.rcParams['savefig.facecolor'] = 'white'
        plt.rcParams['savefig.edgecolor'] = 'none'
        
        # Polices et texte
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10
        plt.rcParams['figure.titlesize'] = 16
        
        # Style des axes - Configuration compatible
        plt.rcParams['axes.spines.top'] = False
        plt.rcParams['axes.spines.right'] = False
        plt.rcParams['axes.grid'] = True
        
        # Grille - utiliser grid.alpha au lieu de axes.grid.alpha
        if 'grid.alpha' in plt.rcParams:
            plt.rcParams['grid.alpha'] = 0.3
        
        # Couleurs et style
        plt.rcParams['axes.prop_cycle'] = mpl.cycler(
            'color', ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
                     '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        )
        
        # Backend pour Jupyter
        try:
            # Vérification si on est dans Jupyter
            if 'ipykernel' in str(type(get_ipython())):
                try:
                    plt.rcParams['backend'] = 'module://matplotlib_inline.backend_inline'
                except:
                    pass  # Backend par défaut si inline non disponible
        except NameError:
            pass  # Pas dans Jupyter
        
        config = {
            'figure_size': figure_size,
            'dpi': dpi,
            'save_dpi': 300,
            'font_size': 12,
            'backend': plt.get_backend()
        }
        
        logger.debug("Configuration matplotlib appliquée")
        return config
        
    except ImportError:
        logger.warning("Matplotlib non disponible")
        return {}

def setup_seaborn(
    style: str = "whitegrid",
    palette: str = "husl", 
    font_scale: float = 1.1,
    context: str = "notebook"
) -> Dict:
    """
    Configure seaborn avec les paramètres optimaux
    
    Parameters:
    -----------
    style : str
        Style des graphiques
    palette : str
        Palette de couleurs
    font_scale : float
        Facteur d'échelle des polices
    context : str
        Contexte d'affichage
        
    Returns:
    --------
    Dict
        Configuration seaborn appliquée
    """
    try:
        import seaborn as sns
        
        # Configuration de base
        sns.set_style(style)
        sns.set_palette(palette)
        sns.set_context(context, font_scale=font_scale)
        
        # Paramètres spécifiques
        sns.set_theme(
            style=style,
            palette=palette,
            context=context,
            font_scale=font_scale,
            rc={
                'figure.facecolor': 'white',
                'axes.facecolor': 'white',
                'savefig.facecolor': 'white'
            }
        )
        
        config = {
            'style': style,
            'palette': palette,
            'context': context,
            'font_scale': font_scale,
            'version': sns.__version__
        }
        
        logger.debug("Configuration seaborn appliquée")
        return config
        
    except ImportError:
        logger.warning("Seaborn non disponible")
        return {}

def setup_jupyter_display() -> Dict:
    """
    Configure l'affichage pour les notebooks Jupyter
    
    Returns:
    --------
    Dict
        Configuration Jupyter appliquée
    """
    config = {}
    
    try:
        # Configuration IPython
        from IPython.display import set_matplotlib_formats
        from IPython.core.display import HTML
        
        # Format des images matplotlib
        set_matplotlib_formats('retina', 'png')
        
        # CSS personnalisé pour les notebooks
        custom_css = """
        <style>
        .output_png {
            display: table-cell;
            text-align: center;
            vertical-align: middle;
        }
        div.cell {
            width: 100%;
            margin-left: auto;
            margin-right: auto;
        }
        .rendered_html table {
            margin: auto;
            border-collapse: collapse;
            border: 1px solid #ddd;
        }
        .rendered_html th, .rendered_html td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        .rendered_html th {
            background-color: #f2f2f2;
            font-weight: bold;
        }
        </style>
        """
        
        # Application du CSS (silencieuse)
        try:
            HTML(custom_css)
        except:
            pass
        
        config = {
            'matplotlib_formats': ['retina', 'png'],
            'custom_css_applied': True,
            'ipython_available': True
        }
        
        logger.debug("Configuration Jupyter appliquée")
        
    except ImportError:
        config = {'ipython_available': False}
        logger.debug("Configuration Jupyter non disponible (environnement non-Jupyter)")
    
    return config

def get_color_palette(palette_name: str = "project", n_colors: int = 10) -> list:
    """
    Retourne une palette de couleurs personnalisée
    
    Parameters:
    -----------
    palette_name : str
        Nom de la palette ('project', 'categorical', 'sequential', 'diverging')
    n_colors : int
        Nombre de couleurs à retourner
        
    Returns:
    --------
    list
        Liste des couleurs hex
    """
    palettes = {
        'project': [
            '#1f77b4',  # Bleu
            '#ff7f0e',  # Orange
            '#2ca02c',  # Vert
            '#d62728',  # Rouge
            '#9467bd',  # Violet
            '#8c564b',  # Marron
            '#e377c2',  # Rose
            '#7f7f7f',  # Gris
            '#bcbd22',  # Olive
            '#17becf'   # Cyan
        ],
        'categorical': [
            '#E31A1C', '#1F78B4', '#33A02C', '#FF7F00',
            '#6A3D9A', '#B15928', '#A6CEE3', '#B2DF8A',
            '#FB9A99', '#FDBF6F', '#CAB2D6', '#FFFF99'
        ],
        'sequential': [
            '#440154', '#482777', '#3f4a8a', '#31678e',
            '#26838f', '#1f9d8a', '#6cce5a', '#b6de2b',
            '#fee825', '#fff200'
        ],
        'diverging': [
            '#67001f', '#b2182b', '#d6604d', '#f4a582',
            '#fddbc7', '#d1e5f0', '#92c5de', '#4393c3',
            '#2166ac', '#053061'
        ]
    }
    
    if palette_name in palettes:
        colors = palettes[palette_name]
        # Répéter la palette si nécessaire
        while len(colors) < n_colors:
            colors.extend(palettes[palette_name])
        return colors[:n_colors]
    else:
        # Utiliser seaborn par défaut
        try:
            import seaborn as sns
            return sns.color_palette(palette_name, n_colors).as_hex()
        except:
            return palettes['project'][:n_colors]

def create_custom_style(name: str = "sta211_style") -> Dict:
    """
    Crée un style personnalisé pour le projet
    
    Parameters:
    -----------
    name : str
        Nom du style personnalisé
        
    Returns:
    --------
    Dict
        Dictionnaire de style personnalisé
    """
    style = {
        # Figures
        'figure.figsize': (12, 8),
        'figure.dpi': 100,
        'figure.facecolor': 'white',
        'figure.edgecolor': 'none',
        
        # Axes
        'axes.facecolor': 'white',
        'axes.edgecolor': '#cccccc',
        'axes.linewidth': 1,
        'axes.grid': True,
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'axes.spines.top': False,
        'axes.spines.right': False,
        
        # Grille
        'grid.color': '#cccccc',
        'grid.linestyle': '-',
        'grid.linewidth': 0.5,
        'grid.alpha': 0.3,
        
        # Texte
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        
        # Couleurs
        'axes.prop_cycle': "cycler('color', ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])",
        
        # Sauvegarde
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.facecolor': 'white',
        'savefig.edgecolor': 'none'
    }
    
    return style

def reset_display_config():
    """Remet la configuration d'affichage aux valeurs par défaut"""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Reset matplotlib
        plt.rcdefaults()
        
        # Reset seaborn
        sns.reset_defaults()
        
        logger.info("Configuration d'affichage remise à zéro")
        
    except ImportError:
        logger.warning("Impossible de remettre à zéro (modules non disponibles)")

def display_color_palette(palette_name: str = "project", save_path: Optional[str] = None):
    """
    Affiche une palette de couleurs
    
    Parameters:
    -----------
    palette_name : str
        Nom de la palette à afficher
    save_path : Optional[str]
        Chemin de sauvegarde de la figure
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        colors = get_color_palette(palette_name, 10)
        
        fig, ax = plt.subplots(figsize=(12, 2))
        
        # Affichage des couleurs
        for i, color in enumerate(colors):
            ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=color))
            ax.text(i + 0.5, 0.5, f'{i+1}\n{color}', 
                   ha='center', va='center', fontsize=8)
        
        ax.set_xlim(0, len(colors))
        ax.set_ylim(0, 1)
        ax.set_title(f'Palette de couleurs: {palette_name}', fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
        
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Palette sauvegardée: {save_path}")
        
        plt.show()
        
    except ImportError:
        logger.error("Impossible d'afficher la palette (matplotlib non disponible)")

# Test de la configuration si exécuté directement
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test de la configuration
    config = setup_display_config()
    print("Configuration d'affichage:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Test de la palette
    try:
        display_color_palette("project")
    except:
        print("Affichage de la palette non disponible")