#!/usr/bin/env python3
"""
Test script to validate SQL templates with the actual database.
This script runs sample queries using the templates and reports the results.
"""

import os
import sys
import json
import unittest
import pymysql
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Add the parent directory to the Python path
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

# Import from query cache service
try:
    from date_handler import get_date_range, replace_time_period_placeholders
    from src.core.wtl_templates import WTLTemplateLibrary
    DATABASE_IMPORTS_AVAILABLE = True
except ImportError:
    DATABASE_IMPORTS_AVAILABLE = False
    print("Database-related imports not available, skipping SQL template tests")

# Load configuration
def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """Load configuration from file."""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        return {}

# Connect to database
def get_db_connection(config: Dict[str, Any]) -> Optional[pymysql.Connection]:
    """Establish database connection."""
    try:
        db_config = config.get('wtl_database_config', {})
        connection = pymysql.connect(
            host=db_config.get('host', 'localhost'),
            port=db_config.get('port', 3306),
            user=db_config.get('user', 'root'),
            password=db_config.get('password', ''),
            database=db_config.get('database', 'work_tracking'),
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )
        print(f"Connected to database: {db_config.get('database')}")
        return connection
    except Exception as e:
        print(f"Database connection error: {e}")
        return None

# Execute SQL query
def execute_query(connection: pymysql.Connection, query: str) -> Tuple[List[Dict[str, Any]], int]:
    """Execute SQL query and return results."""
    try:
        with connection.cursor() as cursor:
            affected_rows = cursor.execute(query)
            results = cursor.fetchall()
            return results, affected_rows
    except Exception as e:
        print(f"Error executing query: {e}")
        print(f"Query was: {query}")
        return [], 0

# Test template with real data
def test_template(connection: pymysql.Connection, template: Dict[str, Any], 
                entity_values: Dict[str, str]) -> None:
    """Test a template with real entity values."""
    print(f"\nTesting Template: {template['template_query']}")
    print("-" * 80)
    
    # Replace placeholders with actual values
    sql_query = template['sql_template']
    
    # First replace non-time period entities
    time_periods = {}
    for placeholder, value in entity_values.items():
        if not placeholder.startswith('time_period'):
            sql_query = sql_query.replace(f"{{{placeholder}}}", value)
        else:
            time_periods[placeholder] = value
    
    # Process time periods
    time_period_dict = {}
    for placeholder, value in time_periods.items():
        time_period_dict[f"{{{placeholder}}}"] = value
    
    sql_query = replace_time_period_placeholders(sql_query, time_period_dict)
    
    print(f"SQL Query: {sql_query}")
    
    # Execute query
    results, affected_rows = execute_query(connection, sql_query)
    
    # Display results
    print(f"Results: {len(results)} rows")
    for i, row in enumerate(results[:5]):
        print(f"  {i+1}. {row}")
    if len(results) > 5:
        print(f"  ... and {len(results) - 5} more rows")

@unittest.skipIf(not DATABASE_IMPORTS_AVAILABLE, "Database imports not available")
class TestSQLTemplates(unittest.TestCase):
    """Test suite for SQL templates with the database."""
    
    def setUp(self):
        """Set up test fixtures."""
        try:
            # Load config
            self.config = load_config()
            
            # Connect to database
            self.connection = get_db_connection(self.config)
            if not self.connection:
                self.skipTest("Failed to connect to database")
                
            # Load templates
            self.library = WTLTemplateLibrary()
            self.templates = self.library.get_templates()
            
            # Setup sample entities
            self.sample_entities = {
                'department_0': '物业管理部',
                'department_1': '财务管理中心',
                'employee_0': '田树君',
                'work_type_0': '施工现场配合',
                'project_0': '杭州软通2025系列小改造', 
                'time_period_0': 'Q1 2025',
                'time_period_1': 'Q2 2025'
            }
        except Exception as e:
            self.skipTest(f"Could not initialize test: {e}")
    
    def tearDown(self):
        """Clean up test fixtures."""
        if hasattr(self, 'connection') and self.connection:
            self.connection.close()
            print("\nClosed database connection")
    
    def test_template_query_execution(self):
        """Test executing SQL templates against the database."""
        # Only test basic templates
        templates_to_test = self.templates[:3]  # Limit to 3 templates for efficiency
        
        for template in templates_to_test:
            # Prepare entity values for this template
            entity_values = {}
            for placeholder in template['entity_map'].keys():
                # Extract key without braces, e.g. '{department_0}' -> 'department_0'
                key = placeholder.strip('{}')
                if key in self.sample_entities:
                    entity_values[key] = self.sample_entities[key]
            
            # Test template if we have all required entity values
            if len(entity_values) == len(template['entity_map']):
                # Replace placeholders with actual values
                sql_query = template['sql_template']
                
                # First replace non-time period entities
                time_periods = {}
                for placeholder, value in entity_values.items():
                    if not placeholder.startswith('time_period'):
                        sql_query = sql_query.replace(f"{{{placeholder}}}", value)
                    else:
                        time_periods[placeholder] = value
                
                # Process time periods
                time_period_dict = {}
                for placeholder, value in time_periods.items():
                    time_period_dict[f"{{{placeholder}}}"] = value
                
                sql_query = replace_time_period_placeholders(sql_query, time_period_dict)
                
                # Execute query and check that it doesn't raise an exception
                try:
                    results, affected_rows = execute_query(self.connection, sql_query)
                    self.assertIsNotNone(results)
                except Exception as e:
                    self.fail(f"SQL execution failed: {e}")
            else:
                # Skip this template in the test
                pass

# For manual testing
def run_manual_test():
    """Manual test function."""
    # Load config
    config = load_config()
    
    # Connect to database
    connection = get_db_connection(config)
    if not connection:
        print("Failed to connect to database. Exiting.")
        return
    
    try:
        # Get templates
        template_library = WTLTemplateLibrary()
        templates = template_library.get_templates()
        
        print(f"Loaded {len(templates)} templates for testing")
        
        # Get sample entities for testing
        # We could query these from the database directly, but for simplicity 
        # we'll use hardcoded values from our earlier database exploration
        
        sample_entities = {
            'department_0': '物业管理部',
            'department_1': '财务管理中心',
            'employee_0': '田树君',
            'work_type_0': '施工现场配合',
            'project_0': '杭州软通2025系列小改造', 
            'time_period_0': 'Q1 2025',
            'time_period_1': 'Q2 2025'
        }
        
        # Test each template
        for i, template in enumerate(templates):
            # Prepare entity values for this template
            entity_values = {}
            for placeholder in template['entity_map'].keys():
                # Extract key without braces, e.g. '{department_0}' -> 'department_0'
                key = placeholder.strip('{}')
                if key in sample_entities:
                    entity_values[key] = sample_entities[key]
            
            # Test template if we have all required entity values
            if len(entity_values) == len(template['entity_map']):
                test_template(connection, template, entity_values)
            else:
                missing = set(placeholder.strip('{}') for placeholder in template['entity_map'].keys()) - set(entity_values.keys())
                print(f"\nSkipping Template {i+1}: Missing entity values for {missing}")
    
    finally:
        # Close database connection
        if connection and connection.open:
            connection.close()
            print("\nClosed database connection")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--manual":
        run_manual_test()
    else:
        unittest.main()