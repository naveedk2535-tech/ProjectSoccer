"""
PythonAnywhere WSGI configuration.
This file is referenced by PythonAnywhere's web app config.
"""
import sys
import os

# Add project directory to path
project_home = '/home/zziai38/ProjectSoccer'
if project_home not in sys.path:
    sys.path.insert(0, project_home)

# Load environment variables from .env
from dotenv import load_dotenv
load_dotenv(os.path.join(project_home, '.env'))

# Import Flask app
from app import app as application
