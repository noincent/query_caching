"""
Test script for the template library and improved entity extraction.
"""

import logging
import unittest
import sys
from typing import Dict, List, Any

from src.core.template_library import TemplateLibrary
from src.utils.entity_extractor import EntityExtractor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestTemplateLibrary(unittest.TestCase):
    """Tests for the template library functionality."""
    
    def test_template_library_init(self):
        """Test if template library initializes correctly."""
        library = TemplateLibrary()
        templates = library.get_templates()
        self.assertGreater(len(templates), 0, "Template library should contain templates")
        
    def test_template_structure(self):
        """Test if templates have required fields."""
        library = TemplateLibrary()
        templates = library.get_templates()
        
        for template in templates:
            self.assertIn('template_query', template, "Template should have 'template_query' field")
            self.assertIn('sql_template', template, "Template should have 'sql_template' field")
            self.assertIn('entity_map', template, "Template should have 'entity_map' field")

class TestEntityExtraction(unittest.TestCase):
    """Tests for the entity extraction functionality."""
    
    def setUp(self):
        """Set up the entity extractor."""
        self.extractor = EntityExtractor({
            'employee': ['John Smith', 'Jane Doe', 'Bob Johnson'],
            'department': ['Engineering', 'Marketing', 'Sales', 'HR'],
            'project': ['Website Redesign', 'Mobile App', 'Database Migration']
        })
    
    def test_time_period_extraction(self):
        """Test time period extraction."""
        query = "How many hours did Bob Johnson work on the Mobile App project in Q3 2023?"
        template, entity_map = self.extractor.extract_and_normalize(query)
        
        # Check if at least one time period entity is extracted
        time_period_entities = [k for k in entity_map.keys() if 'time_period' in k]
        self.assertGreater(len(time_period_entities), 0, "Should extract time period entity")
        
        # Check if the template is normalized
        self.assertIn("{time_period", template, "Template should contain time period placeholder")
    
    def test_employee_extraction(self):
        """Test employee extraction."""
        query = "How many hours did Bob Johnson work on the Mobile App project in Q3 2023?"
        template, entity_map = self.extractor.extract_and_normalize(query)
        
        # Check if at least one employee entity is extracted
        employee_entities = [k for k in entity_map.keys() if 'employee' in k]
        self.assertGreater(len(employee_entities), 0, "Should extract employee entity")
        
        # Check if the template is normalized
        self.assertIn("{employee", template, "Template should contain employee placeholder")
    
    def test_project_extraction(self):
        """Test project extraction."""
        query = "How many hours did Bob Johnson work on the Mobile App project in Q3 2023?"
        template, entity_map = self.extractor.extract_and_normalize(query)
        
        # Check if at least one project entity is extracted
        project_entities = [k for k in entity_map.keys() if 'project' in k]
        self.assertGreater(len(project_entities), 0, "Should extract project entity")
        
        # Check if the template is normalized
        self.assertIn("{project", template, "Template should contain project placeholder")

# Only run the actual test when we're executing this file directly
def print_template_info():
    """Print information about templates - for manual testing."""
    library = TemplateLibrary()
    templates = library.get_templates()
    
    print(f"Template library initialized with {len(templates)} templates")
    
    # Print the first few templates
    for i, template in enumerate(templates[:3]):
        print(f"\nTemplate {i+1}:")
        print(f"Query: {template['template_query']}")
        print(f"SQL: {template['sql_template']}")
        print(f"Entity map: {template['entity_map']}")
        print(f"Metadata: {template.get('metadata', {})}")

def print_entity_extraction():
    """Print entity extraction results - for manual testing."""
    extractor = EntityExtractor({
        'employee': ['John Smith', 'Jane Doe', 'Bob Johnson'],
        'department': ['Engineering', 'Marketing', 'Sales', 'HR'],
        'project': ['Website Redesign', 'Mobile App', 'Database Migration']
    })
    
    # Test with various time periods
    test_queries = [
        "How many hours did Bob Johnson work on the Mobile App project in Q3 2023?",
        "What projects did Jane Doe work on in Q2 2023?",
        "How many hours did John Smith work on the Website Redesign project in 2023?",
        "Who worked on the Database Migration project in March 2023?",
        "What projects did HR work on in 2022?",
        "How many hours did Engineering spend on the Mobile App in last quarter?",
        "Compare the hours worked by Jane Doe and John Smith in this month",
        "Which projects had the most hours in Q1 2023?"
    ]
    
    for query in test_queries:
        print(f"\nOriginal query: {query}")
        template, entity_map = extractor.extract_and_normalize(query)
        print(f"Template: {template}")
        print(f"Entity map: {entity_map}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--print":
        print("\n==== Testing Template Library ====")
        print_template_info()
        
        print("\n==== Testing Improved Entity Extraction ====")
        print_entity_extraction()
    else:
        unittest.main()