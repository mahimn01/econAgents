"""
Logging setup for RRCE Framework.
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Optional

def setup_logger(logging_config: Optional[dict] = None) -> logging.Logger:
    """
    Setup logging for RRCE Framework.
    
    Args:
        logging_config: Logging configuration dictionary
        
    Returns:
        Configured logger
    """
    if logging_config is None:
        logging_config = {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'file': 'logs/rrce_framework.log',
            'max_file_size_mb': 10,
            'backup_count': 5,
        }
    
    # Create logs directory
    log_file = Path(logging_config.get('file', 'logs/rrce_framework.log'))
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = logging.getLogger('rrce_framework')
    logger.setLevel(getattr(logging, logging_config.get('level', 'INFO')))
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(logging_config.get('format'))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=logging_config.get('max_file_size_mb', 10) * 1024 * 1024,
        backupCount=logging_config.get('backup_count', 5)
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger