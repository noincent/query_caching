#!/usr/bin/env python3
"""
Test script to verify CPU-only setup for Query Cache Service.
"""

import os
import sys

def test_environment():
    """Test that environment variables are set correctly."""
    print("=== Environment Test ===")
    
    required_vars = {
        'CUDA_VISIBLE_DEVICES': '',
        'TORCH_USE_CUDA_DSA': '1',
        'TOKENIZERS_PARALLELISM': 'false',
        'SPACY_DISABLE_CUDA': '1'
    }
    
    all_set = True
    for var, expected in required_vars.items():
        actual = os.environ.get(var, '')
        if actual == expected:
            print(f"✓ {var}={actual}")
        else:
            print(f"✗ {var}={actual} (expected: {expected})")
            all_set = False
    
    return all_set

def test_torch():
    """Test PyTorch CPU-only setup."""
    print("\n=== PyTorch Test ===")
    
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        
        cuda_available = torch.cuda.is_available()
        if not cuda_available:
            print("✓ CUDA disabled successfully")
        else:
            print("⚠ CUDA still available (this might cause issues)")
        
        # Test tensor creation
        x = torch.randn(3, 3)
        print(f"✓ Tensor device: {x.device}")
        
        return not cuda_available
        
    except ImportError as e:
        print(f"✗ PyTorch not available: {e}")
        return False
    except Exception as e:
        print(f"✗ PyTorch error: {e}")
        return False

def test_sentence_transformers():
    """Test sentence transformers with CPU-only setup."""
    print("\n=== Sentence Transformers Test ===")
    
    try:
        from sentence_transformers import SentenceTransformer
        print("✓ Sentence transformers imported successfully")
        
        # Try to load a small model
        try:
            model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✓ Model loaded successfully")
            
            # Test encoding
            sentences = ["This is a test sentence."]
            embeddings = model.encode(sentences)
            print(f"✓ Encoding successful, shape: {embeddings.shape}")
            
            return True
            
        except Exception as e:
            print(f"⚠ Model loading failed: {e}")
            print("  This is expected if sentence transformers has CUDA issues")
            return False
            
    except ImportError as e:
        print(f"⚠ Sentence transformers not available: {e}")
        return False

def test_spacy():
    """Test spaCy CPU-only setup."""
    print("\n=== spaCy Test ===")
    
    try:
        import spacy
        print("✓ spaCy imported successfully")
        
        # Try to load models in order of preference
        models_to_try = ['en_core_web_lg', 'en_core_web_sm', 'en_core_web_md']
        
        for model_name in models_to_try:
            try:
                nlp = spacy.load(model_name)
                print(f"✓ Loaded {model_name} successfully")
                
                # Test processing
                doc = nlp("This is a test with Apple Inc. and Google.")
                entities = [(ent.text, ent.label_) for ent in doc.ents]
                print(f"✓ Entity extraction working: {entities}")
                
                return True
                
            except OSError:
                print(f"⚠ {model_name} not available")
                continue
        
        print("✗ No spaCy models available")
        return False
        
    except ImportError as e:
        print(f"✗ spaCy not available: {e}")
        return False

def test_fallback_libraries():
    """Test fallback similarity libraries."""
    print("\n=== Fallback Libraries Test ===")
    
    results = {}
    
    # Test scikit-learn
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Quick test
        vectorizer = TfidfVectorizer()
        texts = ["hello world", "world hello"]
        tfidf_matrix = vectorizer.fit_transform(texts)
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        
        print(f"✓ Scikit-learn TF-IDF working, similarity: {similarity[0][0]:.3f}")
        results['sklearn'] = True
        
    except Exception as e:
        print(f"✗ Scikit-learn error: {e}")
        results['sklearn'] = False
    
    # Test jellyfish
    try:
        import jellyfish
        similarity = jellyfish.jaro_winkler_similarity("hello", "hallo")
        print(f"✓ Jellyfish working, similarity: {similarity:.3f}")
        results['jellyfish'] = True
        
    except Exception as e:
        print(f"⚠ Jellyfish not available: {e}")
        results['jellyfish'] = False
    
    return results

def main():
    """Run all tests."""
    print("Testing CPU-only setup for Query Cache Service")
    print("=" * 50)
    
    # Run tests
    env_ok = test_environment()
    torch_ok = test_torch()
    st_ok = test_sentence_transformers()
    spacy_ok = test_spacy()
    fallback_results = test_fallback_libraries()
    
    # Summary
    print("\n=== Test Summary ===")
    print(f"Environment setup: {'✓' if env_ok else '✗'}")
    print(f"PyTorch CPU-only: {'✓' if torch_ok else '✗'}")
    print(f"Sentence transformers: {'✓' if st_ok else '⚠'}")
    print(f"spaCy: {'✓' if spacy_ok else '⚠'}")
    print(f"Scikit-learn fallback: {'✓' if fallback_results.get('sklearn') else '✗'}")
    print(f"Jellyfish fallback: {'✓' if fallback_results.get('jellyfish') else '⚠'}")
    
    # Determine if setup is usable
    critical_ok = env_ok and torch_ok
    fallback_available = fallback_results.get('sklearn', False)
    
    if critical_ok and (st_ok or fallback_available):
        print("\n✓ Setup is ready for use!")
        print("The Query Cache Service should start without CUDA issues.")
        
        if not st_ok:
            print("Note: Will use TF-IDF fallback instead of sentence transformers.")
            
    elif critical_ok and fallback_available:
        print("\n⚠ Setup has issues but fallbacks are available.")
        print("The service should work with reduced functionality.")
        
    else:
        print("\n✗ Setup has critical issues.")
        print("Please run install_cpu_requirements.sh to fix dependencies.")
        sys.exit(1)

if __name__ == "__main__":
    main()
