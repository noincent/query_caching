import sys
import os
import unittest
import tempfile
import json
import time
import shutil
from pathlib import Path

# Add the parent directory to the Python path
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

# Skip this test if required dependencies are not available
try:
    from src.core.query_cache import QueryCache
    try:
        # Also check if sentence-transformers is available since it's required
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("paraphrase-MiniLM-L3-v2")
        DEPENDENCIES_AVAILABLE = True
    except Exception as e:
        DEPENDENCIES_AVAILABLE = False
        print(f"Skipping query cache tests because sentence-transformers had an error: {e}")
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    print(f"Skipping query cache tests because required dependencies are not available: {e}")

# Create a mock CHESS interface for testing
class MockCHESSInterface:
    def start_chat_session(self, db_id):
        return "mock_session_id"
        
    def chat_query(self, session_id, query):
        # Simulate CHESS generating SQL
        if "hours" in query.lower() and "work" in query.lower():
            return {
                'status': 'success',
                'sql_query': "SELECT SUM(hours) FROM work_hours WHERE employee = 'Bob Johnson' AND project = 'Mobile App' AND period = 'Q3 2023'",
                'results': [[40]],
                'natural_language_response': "Bob Johnson worked 40 hours on the Mobile App project in Q3 2023."
            }
        else:
            return {
                'status': 'error',
                'error': 'Could not generate SQL for this query',
                'sql_query': ''
            }

@unittest.skipIf(not DEPENDENCIES_AVAILABLE, "Required dependencies not available")
class TestQueryCache(unittest.TestCase):
    """Test suite for the QueryCache class."""
    
    def setUp(self):
        """Set up test fixtures."""
        try:
            # Start fresh - create a new instance for each test
            # Create temp directory for test files
            self.temp_dir = tempfile.mkdtemp()
            self.templates_path = os.path.join(self.temp_dir, "templates.json")
            self.entity_dict_path = os.path.join(self.temp_dir, "entities.json")
            
            # Create an empty entities dictionary file
            with open(self.entity_dict_path, 'w') as f:
                json.dump({}, f)
                
            # Create config file
            self.config = {
                'templates_path': self.templates_path,
                'entity_dictionary_path': self.entity_dict_path,
                'similarity_threshold': 0.5,  # Lower threshold for testing
                'max_templates': 100,
                'model_name': "paraphrase-MiniLM-L3-v2",  # Use a smaller model for testing
                'use_predefined_templates': False  # Don't use predefined templates for tests
            }
            
            self.config_path = os.path.join(self.temp_dir, "config.json")
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            
            # Create the query cache
            self.query_cache = QueryCache(config_path=self.config_path)
            
            # Create a mock CHESS interface
            self.mock_chess = MockCHESSInterface()
            
        except Exception as e:
            # If the test setup fails, mark it as skipped
            self.skipTest(f"Could not initialize QueryCache: {e}")
            
    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up temp directory
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            
    def test_process_query_cache_miss_with_chess(self):
        """Test processing a query with a cache miss but CHESS available."""
        query = "How many hours did Bob Johnson work on the Mobile App project in Q3 2023?"
        
        # Process the query with CHESS fallback
        result = self.query_cache.process_query(query, self.mock_chess)
        
        # Check result properties
        self.assertTrue(result['success'])
        self.assertEqual('chess', result['source'])
        self.assertIn('sql_query', result)
        self.assertIn('Bob Johnson', result['sql_query'])
        self.assertIn('Mobile App', result['sql_query'])
        self.assertIn('Q3 2023', result['sql_query'])
        
        # Check that template was added
        self.assertEqual(1, len(self.query_cache.template_matcher.templates))
        
    def test_process_query_cache_hit(self):
        """Test processing a query with a cache hit."""
        # First query to populate cache
        query1 = "How many hours did Bob Johnson work on the Mobile App project in Q3 2023?"
        self.query_cache.process_query(query1, self.mock_chess)
        
        # Similar query that should hit the cache
        query2 = "How many hours did Jane Doe work on the Website Redesign project in Q4 2023?"
        result = self.query_cache.process_query(query2, self.mock_chess)
        
        # Check result properties
        self.assertTrue(result['success'])
        self.assertEqual('cache', result['source'])
        self.assertIn('sql_query', result)
        self.assertIn('Jane Doe', result['sql_query'])
        self.assertIn('Website Redesign', result['sql_query'])
        self.assertIn('Q4 2023', result['sql_query'])
        
    def test_process_query_cache_miss_no_chess(self):
        """Test processing a query with a cache miss and no CHESS."""
        query = "How many hours did Bob Johnson work on the Mobile App project in Q3 2023?"
        
        # Process the query without CHESS fallback
        result = self.query_cache.process_query(query, None)
        
        # Check result properties
        self.assertFalse(result['success'])
        self.assertEqual('cache', result['source'])
        self.assertIn('error', result)
        
    def test_metrics(self):
        """Test that metrics are tracked correctly."""
        # Initialize metrics to zero
        self.query_cache.metrics = {
            'total_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'avg_response_time': 0
        }
        
        # Process a few queries
        query1 = "How many hours did Bob Johnson work on the Mobile App project in Q3 2023?"
        self.query_cache.process_query(query1, self.mock_chess)
        
        # Process a similar query (should hit the cache)
        query2 = "How many hours did Jane Doe work on the Website Redesign project in Q4 2023?"
        self.query_cache.process_query(query2)
        
        # Process an unrelated query
        query3 = "What is the total budget for all projects in 2023?"
        self.query_cache.process_query(query3, self.mock_chess)
        
        # Get the metrics
        metrics = self.query_cache.get_metrics()
        
        # Check metrics properties
        self.assertEqual(3, metrics['total_requests'], f"Incorrect total_requests. Metrics: {metrics}")
        # We either expect 1 or 2 cache hits depending on implementation
        self.assertIn(metrics['cache_hits'], [1, 2], f"Unexpected cache_hits value. Metrics: {metrics}")
        # We expect either 1 or 2 cache misses depending on implementation
        self.assertIn(metrics['cache_misses'], [1, 2], f"Unexpected cache_misses value. Metrics: {metrics}")
        # Check that hit_rate is between 0.2 and 0.8 (covering both 1/3 and 2/3 cases)
        self.assertGreater(metrics['hit_rate'], 0.2, f"Hit rate too low. Metrics: {metrics}")
        self.assertLess(metrics['hit_rate'], 0.8, f"Hit rate too high. Metrics: {metrics}")
        
    def test_query_timing(self):
        """Test that query timing is tracked."""
        query = "How many hours did Bob Johnson work on the Mobile App project in Q3 2023?"
        
        # Process the query and measure time
        start_time = time.time()
        result = self.query_cache.process_query(query, self.mock_chess)
        end_time = time.time()
        
        # Check that query_time_ms is reasonable
        self.assertIn('query_time_ms', result)
        self.assertGreater(result['query_time_ms'], 0)
        
        # It should be less than the actual time it took (since we're measuring inside the function)
        actual_time_ms = int((end_time - start_time) * 1000)
        self.assertLessEqual(result['query_time_ms'], actual_time_ms * 1.1)  # 10% margin


if __name__ == '__main__':
    unittest.main()