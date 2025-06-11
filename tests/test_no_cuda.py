#!/usr/bin/env python3
"""
Test script to verify template matching works without CUDA
"""

import os
import sys
import unittest
from pathlib import Path

# Add src directory to path
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

# Try to import template matcher
try:
    from src.utils.template_matcher import TemplateMatcher
    MATCHER_AVAILABLE = True
except ImportError:
    MATCHER_AVAILABLE = False
    print("TemplateMatcher not available, skipping no-CUDA tests")

@unittest.skipIf(not MATCHER_AVAILABLE, "TemplateMatcher not available")
class TestNoCuda(unittest.TestCase):
    """Test suite for template matching without CUDA."""
    
    def setUp(self):
        """Set up test fixtures."""
        try:
            # Force CPU environment variables
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            
            # Initialize matcher
            self.matcher = TemplateMatcher()
            
            # Add test templates
            self.matcher.add_template(
                template_query="How many hours did {employee_0} work in {time_period_0}?",
                sql_template="SELECT SUM(hours) FROM work_hours WHERE employee = '{employee_0}' AND period = '{time_period_0}'",
                entity_map={
                    '{employee_0}': {'type': 'employee', 'value': 'PLACEHOLDER', 'normalized': 'PLACEHOLDER'},
                    '{time_period_0}': {'type': 'time_period', 'value': 'PLACEHOLDER', 'normalized': 'PLACEHOLDER'}
                }
            )
            
            self.matcher.add_template(
                template_query="How many hours did the {department_0} department work in {time_period_0}?",
                sql_template="SELECT SUM(hours) FROM work_hours WHERE department = '{department_0}' AND period = '{time_period_0}'",
                entity_map={
                    '{department_0}': {'type': 'department', 'value': 'PLACEHOLDER', 'normalized': 'PLACEHOLDER'},
                    '{time_period_0}': {'type': 'time_period', 'value': 'PLACEHOLDER', 'normalized': 'PLACEHOLDER'}
                }
            )
        except Exception as e:
            self.skipTest(f"Could not initialize TemplateMatcher: {e}")
    
    def test_matching_employee_query(self):
        """Test matching an employee query."""
        query = "How many hours did John Smith work in Q1 2025?"
        result = self.matcher.find_matching_template(query)
        
        self.assertIsNotNone(result)
        self.assertEqual("How many hours did {employee_0} work in {time_period_0}?", result['template_query'])
    
    def test_matching_department_query(self):
        """Test matching a department query."""
        query = "How many hours did the Engineering department work in Q2 2025?"
        result = self.matcher.find_matching_template(query)
        
        self.assertIsNotNone(result)
        self.assertEqual("How many hours did the {department_0} department work in {time_period_0}?", result['template_query'])
    
    def test_non_matching_query(self):
        """Test a query that doesn't match any template."""
        query = "What is the weather like today?"
        result = self.matcher.find_matching_template(query)
        
        self.assertIsNone(result)

# For manual testing
def run_manual_test():
    """Manual test to verify template matching without CUDA."""
    print("Testing template matching without CUDA...")
    
    # Force CPU environment variables
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Initialize matcher
    print("Initializing template matcher...")
    matcher = TemplateMatcher()
    
    # Add test templates
    print("Adding test templates...")
    matcher.add_template(
        template_query="How many hours did {employee_0} work in {time_period_0}?",
        sql_template="SELECT SUM(hours) FROM work_hours WHERE employee = '{employee_0}' AND period = '{time_period_0}'",
        entity_map={
            '{employee_0}': {'type': 'employee', 'value': 'PLACEHOLDER', 'normalized': 'PLACEHOLDER'},
            '{time_period_0}': {'type': 'time_period', 'value': 'PLACEHOLDER', 'normalized': 'PLACEHOLDER'}
        }
    )
    
    matcher.add_template(
        template_query="How many hours did the {department_0} department work in {time_period_0}?",
        sql_template="SELECT SUM(hours) FROM work_hours WHERE department = '{department_0}' AND period = '{time_period_0}'",
        entity_map={
            '{department_0}': {'type': 'department', 'value': 'PLACEHOLDER', 'normalized': 'PLACEHOLDER'},
            '{time_period_0}': {'type': 'time_period', 'value': 'PLACEHOLDER', 'normalized': 'PLACEHOLDER'}
        }
    )
    
    # Test query matching
    test_queries = [
        "How many hours did John Smith work in Q1 2025?",
        "How many hours did the Engineering department work in Q2 2025?",
        "What is the weather like today?"
    ]
    
    print("\nTesting query matching:")
    for query in test_queries:
        print(f"\nQuery: {query}")
        result = matcher.find_matching_template(query)
        if result:
            print(f"✓ Matched: {result['template_query']}")
            print(f"  Score: {result['similarity_score']:.4f}")
        else:
            print("✗ No match found")
    
    print("\nTemplate matching test completed")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--manual":
        run_manual_test()
    else:
        unittest.main()