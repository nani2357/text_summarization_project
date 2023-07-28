# Import necessary modules
import os
import sys
import logging

# Define a string for the logging format. This includes the time, severity level, module name, and message.
logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

# Define the directory where the log file will be stored
log_dir = "logs"

# Define the path to the log file
log_filepath = os.path.join(log_dir,"running_logs.log")

# Create the log directory if it doesn't already exist
os.makedirs(log_dir, exist_ok=True)

# Configure the logging module
logging.basicConfig(
    # Set the severity level to INFO. This means it will handle all messages of level INFO and above.
    level=logging.INFO,
    # Set the logging format to the string defined earlier
    format=logging_str,
    # Define the handlers for the logger. 
    # FileHandler will write the logs to a file, and StreamHandler will write the logs to the console (stdout).
    handlers=[
        logging.FileHandler(log_filepath),
        logging.StreamHandler(sys.stdout)
    ]
)

# Get a logger with the name "textsummerizerLogger". 
# This logger will inherit the settings from the root logger configured above.
# This logger can be used to log messages in the application.
logger = logging.getLogger("textsummerizerLogger")
