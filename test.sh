#!/bin/bash
# Unified Test Script for WTL Query Cache Service

# Parse command line options
TEST_TYPE="all"  # all, simple, entity, template, cache, integration, api
VERBOSE=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --simple)
      TEST_TYPE="simple"
      shift
      ;;
    --entity)
      TEST_TYPE="entity"
      shift
      ;;
    --template)
      TEST_TYPE="template"
      shift
      ;;
    --cache)
      TEST_TYPE="cache"
      shift
      ;;
    --integration)
      TEST_TYPE="integration"
      shift
      ;;
    --api)
      TEST_TYPE="api"
      shift
      ;;
    --verbose)
      VERBOSE=true
      shift
      ;;
    --help)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --simple       Run basic tests only"
      echo "  --entity       Run entity extractor tests"
      echo "  --template     Run template matcher tests"
      echo "  --cache        Run query cache tests"
      echo "  --integration  Run integration tests"
      echo "  --api          Run API tests"
      echo "  --verbose      Show detailed test output"
      echo "  --help         Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Activate the virtual environment if it exists
if [ -d "query_cache_env" ]; then
  echo "Activating virtual environment..."
  source query_cache_env/bin/activate
else
  echo "Warning: Virtual environment not found. Consider running setup_env.sh first."
fi

# Run the tests
PYTHON_CMD="python"
VERBOSE_FLAG=""

if [ "$VERBOSE" = true ]; then
  VERBOSE_FLAG="--verbose"
fi

echo "Running $TEST_TYPE tests..."

# Run tests using the unified test runner
$PYTHON_CMD run_tests.py $TEST_TYPE $VERBOSE_FLAG

# Get the exit code
EXIT_CODE=$?

# Deactivate the virtual environment when done (if it was activated)
if [ -d "query_cache_env" ]; then
  deactivate
fi

# Exit with the test exit code
exit $EXIT_CODE