#!/bin/bash

# Create a dedicated virtual environment for query cache service
python -m venv query_cache_env

# Activate the environment
source query_cache_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Deactivate when done
deactivate

echo "Query cache environment has been set up."
echo "To use it, run: source query_cache_env/bin/activate"