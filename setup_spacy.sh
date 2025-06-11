#!/bin/bash
# Improved setup script for installing spaCy and downloading the English model

set -e  # Exit on any error

echo "Installing spaCy and downloading the English model..."

# Activate the virtual environment if it exists
if [ -d "query_cache_env" ]; then
    source query_cache_env/bin/activate
    echo "Activated virtual environment: query_cache_env"
fi

# Install or upgrade spaCy
echo "Installing spaCy..."
pip install -U spacy
echo "✓ Installed spaCy"

# Try multiple methods to install the English model (large version for better performance)
echo "Downloading English model: en_core_web_lg..."

# Method 1: Try the standard spacy download command for large model
if python -m spacy download en_core_web_lg; then
    echo "✓ Downloaded large English model via spacy download"
    MODEL_INSTALLED=true
else
    echo "⚠ Large model download failed, trying small model as fallback..."
    MODEL_INSTALLED=false
fi

# Method 2: Try installing small model as fallback
if [ "$MODEL_INSTALLED" = false ]; then
    echo "Trying small model as fallback..."
    if python -m spacy download en_core_web_sm; then
        echo "✓ Downloaded small English model via spacy download"
        MODEL_INSTALLED=true
    else
        echo "⚠ Standard download failed, trying alternative methods..."
        MODEL_INSTALLED=false
    fi
fi

# Method 3: Try installing via pip if the first methods failed
if [ "$MODEL_INSTALLED" = false ]; then
    echo "Trying pip install method..."
    if pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl; then
        echo "✓ Downloaded English model via pip"
        MODEL_INSTALLED=true
    else
        echo "⚠ Pip install from GitHub also failed"
    fi
fi

# Method 4: Try installing from PyPI (if available)
if [ "$MODEL_INSTALLED" = false ]; then
    echo "Trying PyPI installation..."
    if pip install en-core-web-sm; then
        echo "✓ Downloaded English model from PyPI"
        MODEL_INSTALLED=true
    else
        echo "⚠ PyPI installation also failed"
    fi
fi

# Method 5: Install a basic model as fallback
if [ "$MODEL_INSTALLED" = false ]; then
    echo "All methods failed. Installing basic English model as fallback..."
    if pip install spacy[lookups]; then
        python -c "
import spacy
from spacy.lang.en import English
nlp = English()
# Save a basic model for testing
nlp.to_disk('en_core_web_sm_basic')
print('✓ Created basic English model')
"
        MODEL_INSTALLED=true
    fi
fi

# Test the installation
echo "Testing spaCy installation..."

if [ "$MODEL_INSTALLED" = true ]; then
    # Test with the actual model (try large first, then fall back to small)
    if python -c "
import spacy
try:
    # Try to load large model first
    try:
        nlp = spacy.load('en_core_web_lg')
        model_name = 'en_core_web_lg (large)'
    except OSError:
        # Fall back to small model
        nlp = spacy.load('en_core_web_sm')
        model_name = 'en_core_web_sm (small)'
    
    doc = nlp('This is a test sentence with entities like Apple and Google.')
    print(f'✓ SpaCy test successful with {model_name}!')
    print(f'  Sample: \"{doc[0].text}\" is tagged as {doc[0].pos_}')
    
    # Test entity recognition
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    if entities:
        print(f'  Entities found: {entities}')
    else:
        print('  No entities found (model may be basic)')
except Exception as e:
    print(f'✗ SpaCy test failed: {e}')
    exit(1)
"; then
        echo "✓ Setup complete! SpaCy is ready for use."
    else
        echo "✗ Installation test failed"
        exit 1
    fi
else
    echo "✗ Failed to install any English model"
    exit 1
fi

echo ""
echo "You can now use spaCy for improved entity extraction in the Query Cache Service."
echo "If you encounter issues, try running: pip install -U spacy[lookups]"