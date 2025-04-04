import logging

def configure_module_logging(module_configs):
    """Configure logging for multiple modules at once.
    
    Args:
        module_configs: Dict mapping module names to their config dicts
            e.g., {'read_ct': {'file': 'read_ct.log', 'level': logging.INFO, 'console': True}}
            
    Each module config can contain:
        file: Path to log file (optional)
        level: Log level for this module (optional, defaults to INFO)
        console: Boolean to enable/disable console logging (optional, defaults to True)
        propagate: Whether to propagate to parent loggers (optional, defaults to True)
    """
   # Configure root logger first
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    
    # Common formatter for all handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

    # Configure each module logger
    for module_name, config in module_configs.items():
        logger = logging.getLogger(module_name)
        
        # Clear existing handlers to avoid duplicates
        if logger.hasHandlers():
            logger.handlers.clear()
        
        # Set log level (default to INFO if not specified)
        logger.setLevel(config.get('level', logging.INFO))
            
        # Add file handler if specified
        if 'file' in config:
            file_handler = logging.FileHandler(config['file'])
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        # Add console handler if enabled (default is True)
        console_enabled = config.get('console', True)
        if console_enabled:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            
        # Control propagation
        logger.propagate = config.get('propagate', True)

