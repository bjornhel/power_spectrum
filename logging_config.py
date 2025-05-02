"""
Logging configuration utilities for the power_spectrum package.

This module provides functions to configure logging for multiple Python modules
simultaneously, allowing for customized log levels, file outputs, and console
outputs for each module. It uses Python's built-in logging module to create
a flexible, consistent logging setup across the application.

The primary function, configure_module_logging(), accepts a dictionary mapping
module names to their configuration settings, making it easy to set up different
logging behaviors for different parts of the application.

Example usage:
    import logging
    from logging_config import configure_module_logging
    
    # Configure multiple modules at once
    configure_module_logging({
        'import_ct': {
            'file': 'import_ct.log',
            'level': logging.INFO,
            'console': True,
            'overwrite': False
        },
        'project_data': {
            'file': 'project_data.log',
            'level': logging.DEBUG,
            'console': False
        }
    })
"""

import logging

def configure_module_logging(module_configs):
    """Configure logging for multiple modules at once.
    
    Creates and configures loggers for each module specified in the module_configs
    dictionary. Each logger can have customized settings for log level, file output,
    console output, and other behaviors.
    
    Args:
        module_configs (dict): Dictionary mapping module names to their configuration.
            Each module's configuration is a dictionary that may contain:
                - file (str, optional): Path to log file
                - level (int, optional): Log level (defaults to INFO)
                - console (bool, optional): Enable console logging (defaults to True)
                - overwrite (bool, optional): Overwrite existing log file (defaults to True)
    
    Example:
        >>> configure_module_logging({
        ...     'import_ct': {
        ...         'file': 'import_ct.log',
        ...         'level': logging.INFO,
        ...         'console': True
        ...     }
        ... })
    
    Notes:
        - Each logger's handlers are cleared before configuration to avoid duplicates
        - Logger propagation is disabled by default
        - The default log format includes timestamp, logger name, level, and message
    """
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
            overwrite = config.get('overwrite', True)   # Default to overwrite
            file_mode = 'w' if overwrite else 'a'       
            file_handler = logging.FileHandler(config['file'], mode=file_mode)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        # Add console handler if enabled (default is True)
        console_enabled = config.get('console', True)
        if console_enabled:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # Disable propagation since we don't use any parent loggers
        logger.propagate = True
