#!/bin/bash
# Install CPU-only requirements for Query Cache Service
# This script avoids CUDA dependencies and installs PyTorch CPU-only version

set -e  # Exit on any error

echo "Installing CPU-only requirements for Query Cache Service..."

# Force CPU-only environment variables
export CUDA_VISIBLE_DEVICES=""
export TORCH_USE_CUDA_DSA=1
export TOKENIZERS_PARALLELISM=false

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "Using virtual environment: $VIRTUAL_ENV"
else
    echo "Warning: Not in a virtual environment. Consider using one for isolation."
fi

# Update pip to latest version
echo "Updating pip..."
pip install --upgrade pip

# Install CPU-only PyTorch first (this is crucial to avoid CUDA issues)
echo "Installing CPU-only PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other basic requirements
echo "Installing basic requirements..."
pip install flask>=2.0.1
pip install spacy>=3.0.0
pip install numpy>=1.20.0
pip install python-dotenv>=0.19.0
pip install pymysql>=1.0.2
pip install pandas>=1.3.0
pip install matplotlib>=3.4.0
pip install seaborn>=0.11.0

# Install fallback similarity libraries
echo "Installing fallback similarity libraries..."
pip install scikit-learn>=1.0.0
pip install jellyfish>=0.9.0

# Install sentence transformers (should use CPU-only PyTorch now)
echo "Installing sentence-transformers with CPU-only PyTorch..."
pip install sentence-transformers>=2.2.0

# Test the installation
echo "Testing installations..."

# Test PyTorch CPU-only
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print('✓ PyTorch installed successfully (CPU-only)')
"

# Test sentence transformers
python -c "
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
try:
    from sentence_transformers import SentenceTransformer
    print('✓ Sentence transformers installed successfully')
except Exception as e:
    print(f'⚠ Sentence transformers issue: {e}')
"

# Test scikit-learn
python -c "
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    print('✓ Scikit-learn installed successfully')
except Exception as e:
    print(f'⚠ Scikit-learn issue: {e}')
"

# Test other dependencies
python -c "
import flask
import spacy
import numpy
import pandas
import matplotlib
print('✓ All basic dependencies installed successfully')
"

echo ""
echo "✓ CPU-only requirements installed successfully!"
echo ""
echo "Next steps:"
echo "1. Install spaCy models: ./install_spacy_lg.sh"
echo "2. Test the service: python -m src.api.server"
echo ""
echo "The service will now use CPU-only dependencies and should avoid CUDA issues."