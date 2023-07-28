# Import necessary modules
import os
from pathlib import Path
import logging

# Configure the logging module to display messages with a certain format and severity level
logging.basicConfig(level=logging.INFO, format= '[%(asctime)s]:%(message)s:')

# Define the project name
project_name = "text_summarization"

# Define a list of file paths
list_of_files = [".github/workflows/.gitkeep",
                 f"src/{project_name}/__init__.py",
                 f"src/{project_name}/components/__init__.py",
                 f"src/{project_name}/utils/__init__.py",
                 f"src/{project_name}/utils/common.py",
                 f"src/{project_name}/logging/__init__.py",
                 f"src/{project_name}/config/__init__.py",
                 f"src/{project_name}/config/config.py",
                 f"src/{project_name}/pipeline/__init__.py",
                 f"src/{project_name}/entity/__init__.py",
                 f"src/{project_name}/constants/__init__.py",
                 "config/config.yaml",
                 "params.yaml",
                 "app.py",
                 "main.py",
                 "Dockerfile",
                 "setup.py",
                 "research/trials.ipynb"
                 ]

# Loop over each file path in the list
for filepath in list_of_files:
    # Create a Path object from the file path
    filepath = Path(filepath)
    # Split the file path into the directory and the file name
    filedir, filename = os.path.split(filepath)
    
    # If the directory part of the file path is not empty
    if filedir != "":
        # Create the directory
        os.makedirs(filedir, exist_ok=True)
        # Log a message saying the directory was created
        logging.info(f"Creating directory: {filedir} for the file: {filename}")
        
    # If the file doesn't exist or is empty
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        # Create the file
        with open(filepath,'w') as f:
            pass
        # Log a message saying the file was created
        logging.info(f"Creating empty file: {filepath}")
            
    else:
        # If the file already exists and is not empty, log a message saying so
        logging.info(f"{filename} is already exists")
