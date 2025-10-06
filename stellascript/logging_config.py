"""
Configuration centralisée du logging pour StellaScript.
Logging professionnel, équilibré et non-verbeux.
"""
import logging
import sys
from pathlib import Path


class CustomFormatter(logging.Formatter):
    """
    Formatter personnalisé pour retirer le préfixe 'stellascript.' du nom du logger.
    """
    def format(self, record):
        if record.name.startswith('stellascript.'):
            record.name = record.name[len('stellascript.'):]
        return super().format(record)


def setup_logging(level=logging.INFO, log_file=None):
    """
    Configure le logging pour l'application.

    Args:
        level: Niveau de logging (DEBUG, INFO, WARNING, ERROR)
        log_file: Chemin du fichier de log optionnel
    """
    # Format personnalisé pour un affichage plus propre
    formatter = CustomFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

    # Configuration du logger racine
    root_logger = logging.getLogger('stellascript')
    root_logger.setLevel(level)

    # Éviter les doublons
    if root_logger.handlers:
        return root_logger

    # Handler console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    root_logger.addHandler(console_handler)

    # Handler fichier optionnel
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.StreamHandler(open(log_path, 'a', encoding='utf-8'))
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        root_logger.addHandler(file_handler)

    return root_logger


def get_logger(name):
    """
    Retourne un logger configuré pour le module spécifié.
    Le nom du module est utilisé directement pour créer la hiérarchie.
    """
    # Le nom du module (ex: 'stellascript.audio.enhancement') est suffisant.
    # Le logger racine 'stellascript' est déjà configuré.
    return logging.getLogger(name)


# Configuration par défaut lors de l'import
setup_logging()
