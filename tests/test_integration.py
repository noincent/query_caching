"""
Test script for the Query Cache integration with CHESS.

This script tests the integration by running a series of queries and verifying
that the cache and CHESS fallback work as expected.
"""

import json
import time
import logging
import unittest
import sys
from pathlib import Path

# Add the parent directory to the Python path
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Try to import the integration
try:
    from demo_integration import DemoIntegration
    INTEGRATION_AVAILABLE = True
except ImportError:
    INTEGRATION_AVAILABLE = False
    logger.warning("DemoIntegration not available, skipping integration tests")

def run_test_query(integration, query, user_id=None, skip_cache=False):
    """Run a test query and log the results."""
    logger.info(f"\nRunning query: {query}")
    
    start_time = time.time()
    result = integration.process_query(
        query=query,
        user_id=user_id,
        skip_cache=skip_cache
    )
    total_time = time.time() - start_time
    
    # Log the results
    logger.info(f"Source: {result.get('source', 'unknown')}")
    logger.info(f"Success: {result.get('success', False)}")
    logger.info(f"Processing time: {result.get('processing_time_ms', 0)}ms")
    logger.info(f"Total time: {total_time*1000:.2f}ms")
    
    if result.get('error'):
        logger.error(f"Error: {result['error']}")
    else:
        logger.info(f"SQL query: {result.get('sql_query', 'No SQL generated')}")
        
        if result.get('template_query'):
            logger.info(f"Template query: {result.get('template_query', '')}")
            logger.info(f"Similarity score: {result.get('similarity_score', 0.0):.2f}")
    
    return result

@unittest.skipIf(not INTEGRATION_AVAILABLE, "Integration module not available")
class TestIntegration(unittest.TestCase):
    """Test suite for CHESS integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        try:
            # Initialize the integration
            self.integration = DemoIntegration(config_path="config.json")
        except Exception as e:
            self.skipTest(f"Could not initialize integration: {e}")
    
    def test_cache_hits_and_misses(self):
        """Test cache hits and misses."""
        # First query to populate cache
        query1 = "How many hours did the 物业管理部 department work in Q1 2025?"
        result1 = run_test_query(self.integration, query1)
        self.assertIn('sql_query', result1)
        
        # Similar query to test cache hit
        query2 = "How many hours did the 财务管理中心 department work in Q2 2025?"
        result2 = run_test_query(self.integration, query2)
        self.assertEqual('cache', result2.get('source'), "Second query should hit cache")
        
        # Query with cache bypass
        result3 = run_test_query(self.integration, query1, skip_cache=True)
        self.assertEqual('chess', result3.get('source'), "Query with skip_cache should use CHESS")
    
    def test_metrics(self):
        """Test metrics tracking."""
        # Get initial metrics
        initial_metrics = self.integration.get_metrics()
        initial_requests = initial_metrics.get('metrics', {}).get('total_requests', 0)
        
        # Run a query
        query = "How many hours did 田树君 spend on 材料下单 tasks in Q2 2025?"
        run_test_query(self.integration, query)
        
        # Check that metrics were updated
        final_metrics = self.integration.get_metrics()
        final_requests = final_metrics.get('metrics', {}).get('total_requests', 0)
        self.assertEqual(initial_requests + 1, final_requests)
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Clear the cache
        if hasattr(self, 'integration'):
            self.integration.clear_cache()
        
# For manual testing
def run_manual_test():
    """Run a manual test with detailed logging."""
    logger.info("Testing Query Cache integration with CHESS")
    
    # Initialize the integration
    integration = DemoIntegration(config_path="config.json")
    
    # Get initial metrics
    initial_metrics = integration.get_metrics()
    logger.info(f"Initial metrics: {json.dumps(initial_metrics, indent=2)}")
    
    # Test queries
    test_queries = [
        "How many hours did the 物业管理部 department work in Q1 2025?",
        "How many hours were spent on 施工现场配合 tasks in 2025?",
        "Compare hours between 财务管理中心 and 营销策略中心 departments in Q3 2025",
        "How many hours did 田树君 spend on 材料下单 tasks in Q2 2025?",
        "Which employee worked the most hours in March 2025?",
        "Show the total hours by department for Q2 2025",
        "What percentage of total hours were spent on the 行政中心 project in 2025?",
        "List all projects for 物业管理部 department"
    ]
    
    # Run each query once to populate cache
    logger.info("\n=== Pass 1: Populating Cache ===")
    for query in test_queries:
        run_test_query(integration, query)
    
    # Get metrics after first pass
    pass1_metrics = integration.get_metrics()
    logger.info(f"\nMetrics after Pass 1: {json.dumps(pass1_metrics, indent=2)}")
    
    # Run the same queries again to test cache hits
    logger.info("\n=== Pass 2: Testing Cache Hits ===")
    for query in test_queries:
        run_test_query(integration, query)
    
    # Get metrics after second pass
    pass2_metrics = integration.get_metrics()
    logger.info(f"\nMetrics after Pass 2: {json.dumps(pass2_metrics, indent=2)}")
    
    # Test bypassing the cache
    logger.info("\n=== Pass 3: Testing Cache Bypass ===")
    for query in test_queries[:2]:  # Just test a couple of queries
        run_test_query(integration, query, skip_cache=True)
    
    # Get final metrics
    final_metrics = integration.get_metrics()
    logger.info(f"\nFinal metrics: {json.dumps(final_metrics, indent=2)}")
    
    # Analyze hit rate
    hit_count = pass2_metrics.get('metrics', {}).get('cache_hits', 0) - pass1_metrics.get('metrics', {}).get('cache_hits', 0)
    total_count = len(test_queries)
    hit_rate = (hit_count / total_count) * 100 if total_count > 0 else 0
    
    logger.info(f"\n=== Test Results ===")
    logger.info(f"Total queries: {total_count}")
    logger.info(f"Cache hits on pass 2: {hit_count}")
    logger.info(f"Pass 2 hit rate: {hit_rate:.1f}%")
    
    template_count = final_metrics.get('metrics', {}).get('template_count', 0)
    logger.info(f"Total templates: {template_count}")
    
    avg_time = final_metrics.get('metrics', {}).get('avg_response_time_ms', 0)
    logger.info(f"Average response time: {avg_time:.2f}ms")
    
    # Clear the cache for cleanup
    clear_result = integration.clear_cache()
    logger.info(f"\nCache cleared: {clear_result.get('success', False)}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--manual":
        run_manual_test()
    else:
        unittest.main()