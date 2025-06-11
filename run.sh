#!/bin/bash
# Unified Run Script for WTL Query Cache Service
# This script handles starting the main service with various configuration options

# Default settings
PORT=6000
HOST="0.0.0.0"
DEBUG=false
CONFIG="config.json"
SERVICE_TYPE="api"  # api or demo

# Parse command line options
while [[ $# -gt 0 ]]; do
  case $1 in
    --port)
      PORT="$2"
      shift 2
      ;;
    --host)
      HOST="$2"
      shift 2
      ;;
    --debug)
      DEBUG=true
      shift
      ;;
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --demo)
      SERVICE_TYPE="demo"
      shift
      ;;
    --api)
      SERVICE_TYPE="api"
      shift
      ;;
    --help)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --port PORT     Port to run the server on (default: 6000 for API, 7000 for demo)"
      echo "  --host HOST     Host to bind to (default: 0.0.0.0)"
      echo "  --debug         Run in debug mode"
      echo "  --config FILE   Path to config file (default: config.json)"
      echo "  --demo          Run the demo web server instead of the API"
      echo "  --api           Run the API server (default)"
      echo "  --help          Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Set default port for demo mode if not specified
if [ "$SERVICE_TYPE" = "demo" ] && [ "$PORT" = "6000" ]; then
  PORT=7000
fi

# Activate the virtual environment if it exists
if [ -d "query_cache_env" ]; then
  echo "Activating virtual environment..."
  source query_cache_env/bin/activate
else
  echo "Warning: Virtual environment not found. Consider running setup_env.sh first."
fi

# Make sure the config file exists
if [ ! -f "$CONFIG" ]; then
  echo "Config file '$CONFIG' not found!"
  echo "Creating a default config file..."
  
  # Create a default config if it doesn't exist
  cat > "$CONFIG" << EOF
{
  "templates_path": "data/templates.pkl",
  "entity_dictionary_path": "data/entity_dictionary.json",
  "similarity_threshold": 0.75,
  "max_templates": 1000,
  "model_name": "all-MiniLM-L6-v2",
  "use_predefined_templates": true,
  "use_wtl_templates": true
}
EOF
fi

# Make sure necessary directories exist
mkdir -p data
mkdir -p logs

# Export environment variables
export QUERY_CACHE_PORT=$PORT
export QUERY_CACHE_DEBUG=$DEBUG
export QUERY_CACHE_CONFIG=$(pwd)/$CONFIG
export QUERY_CACHE_LOGS_DIR=$(pwd)/logs

# Create a banner for the server
echo "========================================================"
if [ "$SERVICE_TYPE" = "api" ]; then
  echo "           WTL Query Cache API Server                  "
else
  echo "           WTL Query Cache Demo Server                 "
fi
echo "========================================================"
echo "Host: $HOST"
echo "Port: $PORT"
echo "Debug mode: $DEBUG"
echo "Config file: $CONFIG"
echo "Logs directory: logs/"
echo "========================================================"
echo "Starting server..."
echo ""

# Run the appropriate server
if [ "$SERVICE_TYPE" = "api" ]; then
  # API server
  python -m src.api.server
else
  # Demo server
  # Create necessary directories for demo
  mkdir -p visualizations
  mkdir -p templates
  
  # Set additional environment variables for demo
  export WEB_DEMO_PORT=$PORT
  export WEB_DEMO_DEBUG=$DEBUG
  
  # Run the demo server
  python demos/update_demo_server.py --host "$HOST" --port "$PORT" --config "$CONFIG" $([[ "$DEBUG" == "true" ]] && echo "--debug")
fi

# Deactivate the virtual environment when done (if it was activated)
if [ -d "query_cache_env" ]; then
  deactivate
fi