"""
Test script for the Query Cache Service API.
"""

import requests
import json
import time
import unittest
import sys
from pathlib import Path

# Add the parent directory to the Python path
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

# Set base URL (can be overridden in tests)
BASE_URL = "http://localhost:6000"

class TestQueryCacheAPI(unittest.TestCase):
    """Test suite for the Query Cache Service API."""
    
    @classmethod
    def setUpClass(cls):
        """Set up class fixtures - check if the API is running."""
        try:
            # Try to connect to the API
            response = requests.get(f"{BASE_URL}/health", timeout=1)
            if response.status_code != 200:
                raise unittest.SkipTest("API not available, skipping API tests")
        except Exception as e:
            raise unittest.SkipTest(f"Could not connect to API: {e}")
    
    def test_add_and_query_template(self):
        """Test adding a template and then querying it."""
        # Add a template
        hours_template = "How many hours did {employee_0} work on the {project_0} project in {time_period_0}?"
        hours_sql = "SELECT SUM(hours) FROM work_hours WHERE employee = '{employee_0}' AND project = '{project_0}' AND period = '{time_period_0}'"
        hours_entity_map = {
            "{employee_0}": {"type": "employee", "value": "Bob Johnson", "normalized": "Bob Johnson"},
            "{project_0}": {"type": "project", "value": "Mobile App", "normalized": "Mobile App"},
            "{time_period_0}": {"type": "time_period", "value": "Q3 2023", "normalized": "Q3 2023"}
        }
        
        url = f"{BASE_URL}/add"
        data = {
            "template_query": hours_template,
            "sql_query": hours_sql,
            "entity_map": hours_entity_map
        }
        
        try:
            response = requests.post(url, json=data)
            response.raise_for_status()
            result = response.json()
            self.assertTrue(result["success"], "Failed to add template")
            
            # Query the template
            url = f"{BASE_URL}/query"
            data = {"query": "How many hours did John Smith work on the Website Redesign project in Q4 2023?"}
            
            response = requests.post(url, json=data)
            result = response.json()
            
            self.assertIn('source', result)
            self.assertIn('sql_query', result)
            self.assertIn('John Smith', result['sql_query'])
            self.assertIn('Website Redesign', result['sql_query'])
            self.assertIn('Q4 2023', result['sql_query'])
            
        except Exception as e:
            self.fail(f"API request failed: {e}")
    
    def test_get_templates(self):
        """Test getting all templates."""
        url = f"{BASE_URL}/templates"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            result = response.json()
            
            self.assertIn('templates', result)
            self.assertIsInstance(result['templates'], list)
            
        except Exception as e:
            self.fail(f"API request failed: {e}")
    
    def test_get_metrics(self):
        """Test getting metrics."""
        url = f"{BASE_URL}/metrics"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            result = response.json()
            
            self.assertIn('metrics', result)
            self.assertIsInstance(result['metrics'], dict)
            self.assertIn('total_requests', result['metrics'])
            self.assertIn('cache_hits', result['metrics'])
            self.assertIn('cache_misses', result['metrics'])
            
        except Exception as e:
            self.fail(f"API request failed: {e}")

# For manual testing
def run_manual_test():
    """Run a manual test of the API."""
    def clear_cache():
        """Try to clear the cache (mock implementation)."""
        print("Simulating cache clear...")
        
    def add_template(template_query, sql_template, entity_map):
        """Add a template to the cache."""
        url = f"{BASE_URL}/add"
        data = {
            "template_query": template_query,
            "sql_query": sql_template,
            "entity_map": entity_map
        }
        
        print(f"Adding template: {template_query}")
        print(f"SQL template: {sql_template}")
        
        try:
            response = requests.post(url, json=data)
            response.raise_for_status()
            result = response.json()
            print(f"Result: {result}")
            return result
        except Exception as e:
            print(f"Error adding template: {e}")
            return {"success": False, "error": str(e)}

    def query_cache(query):
        """Query the cache."""
        url = f"{BASE_URL}/query"
        data = {"query": query}
        
        print(f"Querying cache: {query}")
        
        try:
            response = requests.post(url, json=data)
            result = response.json()
            print(f"Result: {json.dumps(result, indent=2)}")
            return result
        except Exception as e:
            print(f"Error querying cache: {e}")
            return {"success": False, "error": str(e)}

    def get_templates():
        """Get all templates in the cache."""
        url = f"{BASE_URL}/templates"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            result = response.json()
            print(f"Templates: {json.dumps(result, indent=2)}")
            return result
        except Exception as e:
            print(f"Error getting templates: {e}")
            return {"success": False, "error": str(e)}

    # Clear the cache (mock)
    clear_cache()
    
    # Add some test templates
    hours_template = "How many hours did {employee_0} work on the {project_0} project in {time_period_0}?"
    hours_sql = "SELECT SUM(hours) FROM work_hours WHERE employee = '{employee_0}' AND project = '{project_0}' AND period = '{time_period_0}'"
    hours_entity_map = {
        "{employee_0}": {"type": "employee", "value": "Bob Johnson", "normalized": "Bob Johnson"},
        "{project_0}": {"type": "project", "value": "Mobile App", "normalized": "Mobile App"},
        "{time_period_0}": {"type": "time_period", "value": "Q3 2023", "normalized": "Q3 2023"}
    }
    
    projects_template = "What projects did {employee_0} work on in {time_period_0}?"
    projects_sql = "SELECT DISTINCT project FROM work_hours WHERE employee = '{employee_0}' AND period = '{time_period_0}'"
    projects_entity_map = {
        "{employee_0}": {"type": "employee", "value": "Jane Doe", "normalized": "Jane Doe"},
        "{time_period_0}": {"type": "time_period", "value": "Q2 2023", "normalized": "Q2 2023"}
    }
    
    # Add templates
    add_template(hours_template, hours_sql, hours_entity_map)
    add_template(projects_template, projects_sql, projects_entity_map)
    
    # Check templates
    get_templates()
    
    # Test queries
    print("\nTesting cache hits:")
    query_cache("How many hours did John Smith work on the Website Redesign project in Q4 2023?")
    query_cache("What projects did Bob Johnson work on in Q1 2023?")
    
    # Test cache misses
    print("\nTesting cache misses:")
    query_cache("What was the total cost of the Mobile App project?")
    query_cache("When will the Database Migration project be completed?")
    
    # Get final cache state
    print("\nFinal cache state:")
    get_templates()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--manual":
        run_manual_test()
    else:
        unittest.main()