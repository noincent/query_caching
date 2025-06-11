#!/bin/bash

# Activate the dedicated environment
if [ -d "query_cache_env" ]; then
    source query_cache_env/bin/activate
    echo "Activated query cache environment"
else
    echo "Warning: Dedicated environment not found. Consider running setup_env.sh first."
fi

# Run the database initialization script
echo "Initializing database with sample data..."
python initialize_db.py

# Check the result
if [ $? -eq 0 ]; then
    echo "Database initialization completed successfully"
else
    echo "Database initialization failed"
fi

# Deactivate environment when done
if [ -d "query_cache_env" ]; then
    deactivate
fi