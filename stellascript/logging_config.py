"""
Configuration centralisée du logging pour StellaScript.
Logging professionnel, équilibré et non-verbeux.
"""
import logging
import sys
from pathlib import Path


class CustomFormatter(logging.Formatter):
    """
    Custom log formatter to remove the 'stellascript.' prefix from logger names.

    This formatter simplifies the log output by stripping the base package name,
    making the logs cleaner and easier to read.
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        Formats the log record.

        Args:
            record (logging.LogRecord): The original log record.

        Returns:
            str: The formatted log message.
        """
        if record.name.startswith('stellascript.'):
            record.name = record.name[len('stellascript.'):]
        return super().format(record)


def setup_logging(level: int = logging.INFO, log_file: str | None = None) -> logging.Logger:
    """
    Configures the logging for the entire application.

    This function sets up a root logger for the 'stellascript' package with
    a custom formatter. It supports logging to both the console and an optional
    log file.

    Args:
        level (int): The logging level (e.g., logging.INFO, logging.DEBUG).
        log_file (str | None): Optional path to a file for logging.

    Returns:
        logging.Logger: The configured root logger for the application.
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


def get_logger(name: str) -> logging.Logger:
    """
    Retrieves a logger instance for a specific module.

    This is a convenience function to get a logger that is part of the
    'stellascript' hierarchy. The logger inherits its configuration from the
    root logger set up by `setup_logging`.

    Args:
        name (str): The name of the logger, typically `__name__` of the module.

    Returns:
        logging.Logger: A configured logger instance.
    """
    return logging.getLogger(name)


# Configuration par défaut lors de l'import
setup_logging()
