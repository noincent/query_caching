"""
Template Matcher Module

This module provides functionality for finding the best matching query template
from a template store based on semantic similarity with CPU-only fallback support.
"""

import os
import json
import pickle
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import sentence transformers with fallback
SENTENCE_TRANSFORMERS_AVAILABLE = False
try:
    # Force CPU usage before any imports
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TORCH_USE_CUDA_DSA"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    import torch
    # Force CPU device
    torch.set_default_tensor_type('torch.FloatTensor')
    
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
    logger.info("Sentence transformers loaded successfully")
except ImportError as e:
    logger.warning(f"Sentence transformers not available: {e}")
    SentenceTransformer = None
except Exception as e:
    logger.warning(f"Error loading sentence transformers: {e}")
    SentenceTransformer = None

# Fallback similarity methods
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
    logger.info("Scikit-learn available for fallback similarity")
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("Scikit-learn not available, using basic keyword matching only")

try:
    import jellyfish
    JELLYFISH_AVAILABLE = True
    logger.info("Jellyfish available for string similarity")
except ImportError:
    JELLYFISH_AVAILABLE = False
    logger.warning("Jellyfish not available, using basic matching only")

class TemplateMatcher:
    """
    Class for matching query templates based on semantic similarity.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", templates_path: Optional[str] = None, force_cpu: bool = True):
        """
        Initialize the template matcher with fallback support for various similarity methods.
        
        Args:
            model_name: Name of the sentence transformer model to use
            templates_path: Path to template store file (JSON or pickle)
            force_cpu: Whether to force CPU-only mode (default True)
        """
        self.model = None
        self.similarity_method = "keyword"  # Default fallback
        
        if force_cpu:
            # Set environment variables to force CPU usage
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            os.environ["TORCH_USE_CUDA_DSA"] = "1"
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        # Try sentence transformers first (best quality)
        if SENTENCE_TRANSFORMERS_AVAILABLE and SentenceTransformer is not None:
            try:
                logger.info(f"Attempting to load sentence transformer model: {model_name}")
                self.model = SentenceTransformer(model_name)
                
                if force_cpu:
                    import torch
                    self.model.to(torch.device('cpu'))
                    logger.info("Forced model to CPU device")
                
                self.similarity_method = "sentence_transformer"
                logger.info(f"Successfully loaded sentence transformer model: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to load sentence transformer model: {e}")
                self.model = None
        
        # Initialize TF-IDF vectorizer as fallback
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        if SKLEARN_AVAILABLE and self.model is None:
            try:
                self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
                self.similarity_method = "tfidf"
                logger.info("Using TF-IDF vectorization for similarity matching")
            except Exception as e:
                logger.warning(f"Failed to initialize TF-IDF vectorizer: {e}")
                self.tfidf_vectorizer = None
        
        # Final fallback: keyword matching
        if self.model is None and self.tfidf_vectorizer is None:
            logger.warning("Using basic keyword matching as final fallback")
            self.similarity_method = "keyword"
        
        self.templates = []
        self.template_embeddings = None
        
        logger.info(f"Template matcher initialized with similarity method: {self.similarity_method}")
        
        if templates_path and os.path.exists(templates_path):
            self.load_templates(templates_path)
    
    def load_templates(self, file_path: str) -> None:
        """
        Load templates from a file (JSON or pickle).
        
        Args:
            file_path: Path to the template file
        """
        try:
            # Try JSON backup file first if pickle file exists
            if file_path.lower().endswith(('.pkl', '.pickle')) and os.path.exists(file_path + '.json'):
                try:
                    with open(file_path + '.json', 'r') as f:
                        data = json.load(f)
                        if 'templates' in data:
                            self.templates = data['templates']
                            logger.info(f"Loaded {len(self.templates)} templates from backup JSON: {file_path}.json")
                            # Generate embeddings
                            self._generate_embeddings()
                            return
                except Exception as json_error:
                    logger.warning(f"Failed to load backup JSON file: {json_error}")
            
            # Proceed with regular loading
            suffix = Path(file_path).suffix.lower()
            if suffix == '.json':
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, dict) and 'templates' in data:
                        self.templates = data['templates']
                    else:
                        self.templates = data
            elif suffix in ['.pkl', '.pickle']:
                try:
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f)
                        if isinstance(data, dict):
                            self.templates = data.get('templates', [])
                            self.template_embeddings = data.get('embeddings')
                        else:
                            self.templates = data if isinstance(data, list) else []
                except Exception as pickle_error:
                    logger.warning(f"Failed to load pickle file, trying as JSON: {pickle_error}")
                    # If pickle fails, try loading as JSON anyway (some systems save .pkl as text)
                    try:
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                            if isinstance(data, dict) and 'templates' in data:
                                self.templates = data['templates']
                            else:
                                self.templates = data
                    except Exception:
                        # If all loading attempts fail, just start with empty templates
                        logger.error(f"Could not load templates from {file_path} in any format")
                        self.templates = []
            else:
                raise ValueError(f"Unsupported file format: {suffix}")
            
            # Generate embeddings if not loaded from file
            if self.template_embeddings is None or len(self.template_embeddings) != len(self.templates):
                self._generate_embeddings()
                
            logger.info(f"Loaded {len(self.templates)} templates from {file_path}")
        except Exception as e:
            logger.error(f"Error loading templates: {e}")
            # Don't re-raise - just start with empty templates
            self.templates = []
            self._generate_embeddings()
    
    def save_templates(self, file_path: str) -> None:
        """
        Save templates and their embeddings to a file.
        
        Args:
            file_path: Path to save the templates to
        """
        try:
            suffix = Path(file_path).suffix.lower()
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Always save a JSON backup for compatibility
            json_path = file_path
            if suffix != '.json':
                json_path = file_path + '.json'
                
            with open(json_path, 'w') as f:
                json.dump({'templates': self.templates}, f, indent=2)
                
            logger.info(f"Saved {len(self.templates)} templates to {json_path} (JSON format)")
            
            # If pickle is requested, still save pickle but as secondary
            if suffix in ['.pkl', '.pickle']:
                try:
                    with open(file_path, 'wb') as f:
                        pickle.dump({
                            'templates': self.templates,
                            'embeddings': self.template_embeddings
                        }, f)
                    logger.info(f"Also saved pickle format to {file_path}")
                except Exception as pickle_error:
                    logger.warning(f"Failed to save pickle format, but JSON backup was successful: {pickle_error}")
            
        except Exception as e:
            logger.error(f"Error saving templates to {file_path}: {e}")
            # Don't re-raise, but log the error
    
    def _generate_embeddings(self) -> None:
        """Generate embeddings for all templates in the store using available methods."""
        if not self.templates:
            self.template_embeddings = np.array([])
            self.tfidf_matrix = None
            return
        
        queries = [template['template_query'] for template in self.templates]
        
        # Try sentence transformer embeddings first
        if self.similarity_method == "sentence_transformer" and self.model is not None:
            try:
                self.template_embeddings = self.model.encode(queries, convert_to_numpy=True)
                logger.info(f"Generated sentence transformer embeddings for {len(queries)} templates")
                return
            except Exception as e:
                logger.error(f"Error generating sentence transformer embeddings: {e}")
                # Fall back to TF-IDF or keywords
                self.similarity_method = "tfidf" if self.tfidf_vectorizer else "keyword"
        
        # Try TF-IDF embeddings
        if self.similarity_method == "tfidf" and self.tfidf_vectorizer is not None:
            try:
                self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(queries)
                logger.info(f"Generated TF-IDF matrix for {len(queries)} templates")
                # Create compatible embeddings array
                self.template_embeddings = self.tfidf_matrix.toarray()
                return
            except Exception as e:
                logger.error(f"Error generating TF-IDF embeddings: {e}")
                self.similarity_method = "keyword"
        
        # Keyword matching doesn't need pre-computed embeddings
        if self.similarity_method == "keyword":
            # Create dummy embeddings for compatibility
            self.template_embeddings = np.zeros((len(self.templates), 10))
            logger.info(f"Using keyword matching for {len(queries)} templates (no pre-computed embeddings needed)")
        else:
            # Fallback to dummy embeddings
            self.template_embeddings = np.zeros((len(self.templates), 384))
            logger.warning(f"Using dummy embeddings for {len(self.templates)} templates")
    
    def add_template(self, template_query: str, sql_template: str, 
                     entity_map: Dict[str, Dict[str, str]],
                     metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a new template to the store.
        
        Args:
            template_query: Template query with entity placeholders
            sql_template: SQL template with entity placeholders
            entity_map: Dictionary mapping placeholders to entity information
            metadata: Optional metadata about the template
        """
        template = {
            'template_query': template_query,
            'sql_template': sql_template,
            'entity_map': entity_map,
            'metadata': metadata or {},
            'usage_count': 0,
            'success_rate': 1.0,
        }
        
        self.templates.append(template)
        
        # Update embeddings
        try:
            if self.model is None:
                # If model is unavailable, use dummy embedding
                if self.template_embeddings is None or len(self.template_embeddings) == 0:
                    self.template_embeddings = np.zeros((1, 384))
                else:
                    new_embedding = np.zeros((1, self.template_embeddings.shape[1]))
                    self.template_embeddings = np.vstack([self.template_embeddings, new_embedding])
            else:
                if self.template_embeddings is None or len(self.template_embeddings) == 0:
                    self.template_embeddings = self.model.encode([template_query], convert_to_numpy=True)
                else:
                    new_embedding = self.model.encode([template_query], convert_to_numpy=True)
                    self.template_embeddings = np.vstack([self.template_embeddings, new_embedding])
        except Exception as e:
            logger.error(f"Error updating embeddings for new template: {e}")
            # Create dummy embedding if needed
            if self.template_embeddings is None:
                self.template_embeddings = np.zeros((len(self.templates), 384))
            elif len(self.template_embeddings) < len(self.templates):
                # Add a zero embedding for the new template
                new_embedding = np.zeros((1, self.template_embeddings.shape[1]))
                self.template_embeddings = np.vstack([self.template_embeddings, new_embedding])
                
        logger.info(f"Added new template: {template_query}")
    
    def calculate_similarity(self, query1: str, query2: str) -> float:
        """
        Calculate similarity score between two queries using available similarity methods.
        
        Args:
            query1: First query string
            query2: Second query string
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        try:
            # Use sentence transformer similarity
            if self.similarity_method == "sentence_transformer" and self.model is not None:
                return self._calculate_sentence_transformer_similarity(query1, query2)
            
            # Use TF-IDF similarity
            elif self.similarity_method == "tfidf" and self.tfidf_vectorizer is not None:
                return self._calculate_tfidf_similarity(query1, query2)
            
            # Use keyword similarity (always available)
            else:
                return self._calculate_keyword_similarity(query1, query2)
                
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            # Final fallback to basic keyword matching
            return self._calculate_keyword_similarity(query1, query2)
    
    def _calculate_sentence_transformer_similarity(self, query1: str, query2: str) -> float:
        """Calculate similarity using sentence transformers."""
        try:
            # Encode both queries
            embedding1 = self.model.encode(query1, convert_to_numpy=True)
            embedding2 = self.model.encode(query2, convert_to_numpy=True)
            
            # Calculate cosine similarity
            similarity = np.dot(embedding1, embedding2) / (
                np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
            )
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error in sentence transformer similarity calculation: {e}")
            return 0.0
    
    def _calculate_tfidf_similarity(self, query1: str, query2: str) -> float:
        """Calculate similarity using TF-IDF."""
        try:
            # Transform both queries
            vectors = self.tfidf_vectorizer.transform([query1, query2])
            
            # Calculate cosine similarity
            similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error in TF-IDF similarity calculation: {e}")
            return 0.0
    
    def _calculate_keyword_similarity(self, query1: str, query2: str) -> float:
        """Calculate similarity using keyword matching."""
        # Convert queries to lowercase and split into words
        query1_words = set(query1.lower().split())
        query2_words = set(query2.lower().split())
        query1_text = query1.lower()
        query2_text = query2.lower()
        
        # Strategy 1: Jaccard similarity (word overlap)
        if query1_words and query2_words:
            intersection = len(query1_words.intersection(query2_words))
            union = len(query1_words.union(query2_words))
            jaccard_score = intersection / union
        else:
            jaccard_score = 0.0
        
        # Strategy 2: String similarity using Jaro-Winkler if available
        jaro_score = 0.0
        if JELLYFISH_AVAILABLE:
            try:
                jaro_score = jellyfish.jaro_winkler_similarity(query1_text, query2_text)
            except:
                jaro_score = 0.0
        
        # Strategy 3: Longest common subsequence ratio
        lcs_score = self._lcs_similarity(query1_text, query2_text)
        
        # Strategy 4: Entity placeholder matching
        placeholder_score = self._placeholder_similarity(query1_text, query2_text)
        
        # Combine scores with weights
        final_score = (
            0.4 * jaccard_score +
            0.2 * jaro_score +
            0.2 * lcs_score +
            0.2 * placeholder_score
        )
        
        return final_score

    def find_matching_template(self, query: str, similarity_threshold: float = 0.6) -> Optional[Dict[str, Any]]:
        """
        Find the best matching template for a query using available similarity methods.
        
        Args:
            query: Natural language query to match
            similarity_threshold: Minimum similarity score to consider a match
            
        Returns:
            Best matching template or None if no match found
        """
        if not self.templates:
            logger.warning("No templates available for matching")
            return None
        
        logger.info(f"Finding template match for query using {self.similarity_method} method")
        
        try:
            # Use sentence transformer similarity
            if self.similarity_method == "sentence_transformer" and self.model is not None:
                return self._sentence_transformer_matching(query, similarity_threshold)
            
            # Use TF-IDF similarity
            elif self.similarity_method == "tfidf" and self.tfidf_vectorizer is not None:
                return self._tfidf_matching(query, similarity_threshold)
            
            # Use keyword similarity (always available)
            else:
                return self._keyword_matching(query, similarity_threshold)
                
        except Exception as e:
            logger.error(f"Error in template matching: {e}")
            # Final fallback to basic keyword matching
            return self._keyword_matching(query, similarity_threshold)
    
    def _sentence_transformer_matching(self, query: str, similarity_threshold: float) -> Optional[Dict[str, Any]]:
        """Semantic similarity using sentence transformers."""
        try:
            # Encode the query
            query_embedding = self.model.encode(query, convert_to_numpy=True)
            
            # Calculate cosine similarity with all templates
            similarities = np.dot(self.template_embeddings, query_embedding) / (
                np.linalg.norm(self.template_embeddings, axis=1) * np.linalg.norm(query_embedding)
            )
            
            return self._process_similarities(similarities, similarity_threshold, "sentence transformer")
            
        except Exception as e:
            logger.error(f"Error in sentence transformer matching: {e}")
            return None
    
    def _tfidf_matching(self, query: str, similarity_threshold: float) -> Optional[Dict[str, Any]]:
        """TF-IDF based similarity matching."""
        try:
            # Transform the query using the fitted vectorizer
            query_vector = self.tfidf_vectorizer.transform([query])
            
            # Calculate cosine similarity
            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            
            return self._process_similarities(similarities, similarity_threshold, "TF-IDF")
            
        except Exception as e:
            logger.error(f"Error in TF-IDF matching: {e}")
            return None
    
    def _process_similarities(self, similarities: np.ndarray, similarity_threshold: float, method_name: str) -> Optional[Dict[str, Any]]:
        """Process similarity scores and return best match."""
        # Sort templates by similarity score (descending)
        sorted_indices = np.argsort(similarities)[::-1]
        
        # Log top matches for debugging
        top_n = min(3, len(sorted_indices))
        top_templates = [(self.templates[idx]['template_query'], float(similarities[idx])) 
                        for idx in sorted_indices[:top_n]]
        logger.debug(f"Top {top_n} {method_name} matches: {top_templates}")
        
        # Find the best match that meets the threshold
        for idx in sorted_indices:
            score = similarities[idx]
            if score >= similarity_threshold:
                template = self.templates[idx].copy()
                template['similarity_score'] = float(score)
                logger.info(f"Found matching template with {method_name} score {score}: {template['template_query']}")
                return template
        
        # If we get here, no template met the threshold
        best_idx = sorted_indices[0]
        best_score = similarities[best_idx]
        logger.info(f"No matching template found above threshold {similarity_threshold} (best {method_name} score: {best_score})")
        return None
    
    def _keyword_matching(self, query: str, similarity_threshold: float) -> Optional[Dict[str, Any]]:
        """
        Keyword-based matching using multiple strategies.
        
        Args:
            query: Query text
            similarity_threshold: Threshold for considering a match
            
        Returns:
            Best matching template or None
        """
        logger.info(f"Using keyword matching for query: {query}")
        
        # Convert query to lowercase and split into words
        query_words = set(query.lower().split())
        query_text = query.lower()
        
        scores = []
        for template in self.templates:
            template_text = template['template_query'].lower()
            template_words = set(template_text.split())
            
            # Strategy 1: Jaccard similarity (word overlap)
            if query_words and template_words:
                intersection = len(query_words.intersection(template_words))
                union = len(query_words.union(template_words))
                jaccard_score = intersection / union
            else:
                jaccard_score = 0.0
            
            # Strategy 2: String similarity using Jaro-Winkler if available
            jaro_score = 0.0
            if JELLYFISH_AVAILABLE:
                try:
                    jaro_score = jellyfish.jaro_winkler_similarity(query_text, template_text)
                except:
                    jaro_score = 0.0
            
            # Strategy 3: Longest common subsequence ratio
            lcs_score = self._lcs_similarity(query_text, template_text)
            
            # Strategy 4: Entity placeholder matching
            placeholder_score = self._placeholder_similarity(query_text, template_text)
            
            # Combine scores with weights
            final_score = (
                0.4 * jaccard_score +
                0.2 * jaro_score +
                0.2 * lcs_score +
                0.2 * placeholder_score
            )
            
            scores.append(final_score)
        
        # Find the best match
        if not scores:
            return None
            
        best_idx = np.argmax(scores)
        best_score = scores[best_idx]
        
        # Use a slightly lower threshold for keyword matching
        adjusted_threshold = similarity_threshold * 0.7
        
        if best_score >= adjusted_threshold:
            template = self.templates[best_idx].copy()
            template['similarity_score'] = best_score
            logger.info(f"Found matching template with keyword score {best_score}: {template['template_query']}")
            return template
            
        logger.info(f"No matching template found with keyword matching (best score: {best_score}, threshold: {adjusted_threshold})")
        return None
    
    def _lcs_similarity(self, s1: str, s2: str) -> float:
        """Calculate similarity based on longest common subsequence."""
        def lcs_length(x, y):
            m, n = len(x), len(y)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if x[i-1] == y[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            
            return dp[m][n]
        
        if not s1 or not s2:
            return 0.0
        
        lcs_len = lcs_length(s1, s2)
        max_len = max(len(s1), len(s2))
        return lcs_len / max_len if max_len > 0 else 0.0
    
    def _placeholder_similarity(self, query: str, template: str) -> float:
        """Calculate similarity based on non-placeholder parts."""
        # Remove common placeholder patterns
        import re
        
        # Remove entity placeholders like {employee_0}, {project_0}, etc.
        template_clean = re.sub(r'\{[^}]+\}', '[ENTITY]', template)
        
        # Split into words and compare
        query_words = set(query.split())
        template_words = set(template_clean.split())
        
        # Remove common stop words and entity markers
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', '[ENTITY]'}
        query_words = query_words - stop_words
        template_words = template_words - stop_words
        
        if not query_words or not template_words:
            return 0.0
        
        intersection = len(query_words.intersection(template_words))
        union = len(query_words.union(template_words))
        
        return intersection / union if union > 0 else 0.0
    
    # Keep the old method name for backward compatibility
    def _fallback_matching(self, query: str, similarity_threshold: float) -> Optional[Dict[str, Any]]:
        """Fallback method - delegates to keyword matching."""
        return self._keyword_matching(query, similarity_threshold)
    
    def update_template_stats(self, template_idx: int, success: bool) -> None:
        """
        Update usage statistics for a template.
        
        Args:
            template_idx: Index of the template to update
            success: Whether the template was successfully used
        """
        if 0 <= template_idx < len(self.templates):
            template = self.templates[template_idx]
            template['usage_count'] += 1
            
            # Update success rate with exponential moving average
            alpha = 0.1  # Weight for new observation
            template['success_rate'] = (1 - alpha) * template['success_rate'] + alpha * (1.0 if success else 0.0)
            
            logger.info(f"Updated template stats: usage={template['usage_count']}, success_rate={template['success_rate']:.2f}")


# Example usage
if __name__ == "__main__":
    # Create a template matcher
    matcher = TemplateMatcher()
    
    # Add some example templates
    matcher.add_template(
        template_query="How many hours did {employee_0} work on the {project_0} project in {time_period_0}?",
        sql_template="SELECT SUM(hours) FROM work_hours WHERE employee = '{employee_0}' AND project = '{project_0}' AND period = '{time_period_0}'",
        entity_map={
            '{employee_0}': {'type': 'employee', 'value': 'John Smith', 'normalized': 'John Smith'},
            '{project_0}': {'type': 'project', 'value': 'Website Redesign', 'normalized': 'Website Redesign'},
            '{time_period_0}': {'type': 'time_period', 'value': 'Q1 2023', 'normalized': 'Q1 2023'}
        }
    )
    
    matcher.add_template(
        template_query="List all projects {employee_0} worked on in {time_period_0}",
        sql_template="SELECT DISTINCT project FROM work_hours WHERE employee = '{employee_0}' AND period = '{time_period_0}'",
        entity_map={
            '{employee_0}': {'type': 'employee', 'value': 'Jane Doe', 'normalized': 'Jane Doe'},
            '{time_period_0}': {'type': 'time_period', 'value': 'Q2 2023', 'normalized': 'Q2 2023'}
        }
    )
    
    # Test finding a matching template
    query = "How many hours did Bob Johnson work on the Mobile App project in Q3 2023?"
    matching_template = matcher.find_matching_template(query)
    
    print(f"Query: {query}")
    if matching_template:
        print(f"Matching template: {matching_template['template_query']}")
        print(f"SQL template: {matching_template['sql_template']}")
        print(f"Similarity score: {matching_template['similarity_score']:.4f}")
    else:
        print("No matching template found")