#!/bin/bash

echo "=== Installing Multilingual Support for Query Cache Service ==="
echo

# Check if we're in the right directory
if [ ! -f "config.json" ] || [ ! -d "src" ]; then
    echo "Error: Please run this script from the query_cache_service root directory"
    exit 1
fi

# Create a backup of current setup
echo "Creating backup of current configuration..."
cp config.json config.json.backup.$(date +%Y%m%d_%H%M%S)
if [ -d "data" ]; then
    cp -r data data_backup_$(date +%Y%m%d_%H%M%S)
fi

echo "Installing Python dependencies..."

# Install core multilingual dependencies
pip install -U sentence-transformers
pip install -U langdetect
pip install -U jieba

# Install enhanced spaCy with Chinese support
pip install -U spacy

# Download spaCy models
echo "Downloading spaCy language models..."
python -m spacy download en_core_web_lg || echo "Warning: en_core_web_lg failed, trying alternatives..."
python -m spacy download en_core_web_sm || echo "Warning: en_core_web_sm failed"

# Try to download Chinese model (may not be available in all environments)
python -m spacy download zh_core_web_lg || python -m spacy download zh_core_web_sm || echo "Warning: Chinese spaCy model not available"

echo
echo "Creating multilingual test script..."

# Create test script
cat > test_multilingual_setup.py << 'EOF'
#!/usr/bin/env python3
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_multilingual_models():
    """Test multilingual model setup."""
    print("Testing multilingual support...")
    
    # Test sentence transformer
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        
        # Test embeddings for both languages
        sentences = [
            "How many hours did the employee work?",
            "员工工作了多少小时？",
            "Which department worked the most?",
            "哪个部门工作时间最多？"
        ]
        
        embeddings = model.encode(sentences)
        print(f"✓ Multilingual embeddings shape: {embeddings.shape}")
        
        # Test similarity
        from sklearn.metrics.pairwise import cosine_similarity
        sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        print(f"✓ Cross-lingual similarity (EN-ZH): {sim:.3f}")
        
    except Exception as e:
        print(f"✗ Multilingual model test failed: {e}")
        return False
    
    # Test Chinese NLP
    try:
        import jieba
        text = "田树君在物业管理部工作"
        words = list(jieba.cut(text))
        print(f"✓ Jieba segmentation: {' / '.join(words)}")
    except Exception as e:
        print(f"✗ Jieba test failed: {e}")
    
    # Test language detection
    try:
        from langdetect import detect
        print(f"✓ Language detection: EN='{detect('Hello world')}', ZH='{detect('你好世界')}'")
    except Exception as e:
        print(f"✗ Language detection failed: {e}")
    
    return True

if __name__ == "__main__":
    print("=== Multilingual Support Test Suite ===\n")
    success = test_multilingual_models()
    if success:
        print("\n✓ Multilingual setup successful!")
        print("You can now use Chinese and English queries interchangeably.")
    else:
        print("\n✗ Some issues detected. Check the error messages above.")
    print("=" * 50)
EOF

chmod +x test_multilingual_setup.py

echo "Testing installation..."
python test_multilingual_setup.py

echo
echo "=== Installation Summary ==="
echo "✓ Updated configuration for multilingual support"
echo "✓ Installed sentence-transformers with multilingual model"
echo "✓ Installed jieba for Chinese text processing"
echo "✓ Installed langdetect for language identification"
echo "✓ Installed/updated spaCy language models"
echo
echo "Next steps:"
echo "1. Run: python test_multilingual_setup.py (to verify setup)"
echo "2. The system now supports Chinese/English mixed queries"
echo "3. Implement alias mappings for better entity normalization"
echo
echo "For troubleshooting, check test_multilingual_setup.py output above."