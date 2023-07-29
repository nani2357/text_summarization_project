# Import necessary modules
import os
from box.exceptions import BoxValueError
import yaml
from textSummarization.logging import logger
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any

# Define a function to read a YAML file and return its contents as a ConfigBox object
@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """Reads a YAML file and returns its contents.

    Args:
        path_to_yaml (Path): The path to the YAML file.

    Raises:
        ValueError: If the YAML file is empty.
        e: Any other exceptions that occur.

    Returns:
        ConfigBox: The contents of the YAML file as a ConfigBox object.
    """
    try:
        # Open the YAML file and load its contents
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            # Log a message indicating the file was loaded successfully
            logger.info(f"YAML file: {path_to_yaml} loaded successfully")
            # Return the contents as a ConfigBox object
            return ConfigBox(content)
    except BoxValueError:
        # If a BoxValueError is raised (which would happen if the YAML file is empty), raise a ValueError
        raise ValueError("YAML file is empty")
    except Exception as e:
        # If any other exception is raised, re-raise it
        raise e






# Define a function to create a list of directories
@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """Creates a list of directories.

    Args:
        path_to_directories (list): A list of paths to the directories to be created.
        verbose (bool, optional): If True, log a message for each directory created. Defaults to True.
    """
    for path in path_to_directories:
        # Create the directory
        os.makedirs(path, exist_ok=True)
        # If verbose is True, log a message indicating the directory was created
        if verbose:
            logger.info(f"Created directory at: {path}")
            
            
            

# Define a function to get the size of a file
@ensure_annotations
def get_size(path: Path) -> str:
    """Gets the size of a file in kilobytes.

    Args:
        path (Path): The path to the file.

    Returns:
        str: The size of the file in kilobytes.
    """
    # Get the size of the file in bytes, convert it to kilobytes, and round it to the nearest whole number
    size_in_kb = round(os.path.getsize(path)/1024)
    # Return the size as a string
    return f"~ {size_in_kb} KB"
