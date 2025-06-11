#!/bin/bash
# Install spaCy large English model for better entity recognition

set -e  # Exit on any error

echo "Installing spaCy large English model (en_core_web_lg)..."
echo "This model provides better entity recognition but is larger (~750MB)"

# Check if spacy is installed
if ! python -c "import spacy" 2>/dev/null; then
    echo "SpaCy not found. Installing spaCy first..."
    pip install -U spacy
fi

# Download the large English model
echo "Downloading en_core_web_lg..."
if python -m spacy download en_core_web_lg; then
    echo "✓ Successfully downloaded en_core_web_lg"
else
    echo "Failed to download en_core_web_lg. Trying alternative methods..."
    
    # Try direct pip install
    if pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.8.0/en_core_web_lg-3.8.0-py3-none-any.whl; then
        echo "✓ Successfully installed en_core_web_lg via direct download"
    else
        echo "Failed to install via direct download. You may need to:"
        echo "1. Check your internet connection"
        echo "2. Ensure you have enough disk space (~750MB)"
        echo "3. Try manually: pip install en-core-web-lg"
        exit 1
    fi
fi

# Test the installation
echo "Testing the large model installation..."
python -c "
import spacy
try:
    nlp = spacy.load('en_core_web_lg')
    doc = nlp('Apple Inc. is looking for a Senior Software Engineer in San Francisco.')
    print('✓ en_core_web_lg loaded successfully!')
    
    # Test entity recognition
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    print(f'✓ Entity recognition test: {entities}')
    
    # Test word vectors (available in large model)
    if nlp.vocab.vectors.size > 0:
        print(f'✓ Word vectors available: {nlp.vocab.vectors.size} vectors')
    else:
        print('⚠ Word vectors not available (this is unusual for the large model)')
        
except Exception as e:
    print(f'✗ Test failed: {e}')
    exit 1
"

echo ""
echo "✓ SpaCy large model setup complete!"
echo "The query cache service will now use en_core_web_lg for better entity recognition."
echo ""
echo "Model features:"
echo "- Better named entity recognition"
echo "- Word vectors for semantic similarity"
echo "- Improved accuracy for complex queries"
echo "- Better handling of domain-specific entities"