"""
Main entry point for the Query Cache Service.
"""

import sys
import os
from pathlib import Path

# Add the parent directory to Python path
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

from src.api.server import main

if __name__ == "__main__":
    main()