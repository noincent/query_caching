#!/usr/bin/env python3
"""
Test script for quarter handling in the QueryCache.
This script tests the quarter handling in the QueryCache with weekly date ranges.
"""

import os
import sys
import json
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import from the local directory
from src.core.query_cache import QueryCache
from src.utils.entity_extractor import EntityExtractor

def test_quarter_entity_extraction():
    """Test the quarter entity extraction in QueryCache."""
    extractor = EntityExtractor()
    
    # Test various quarter formats
    test_queries = [
        "How many hours did Bob work in Q1 2025?",
        "Show me the project status for Q2 2025",
        "What's the progress in Q3 2025?",
        "Generate report for Q4 2025",
        "first quarter of 2025 statistics",
        "second quarter 2025 financial report",
        "third quarter revenue for department X",
        "fourth quarter expenses"
    ]
    
    print("\nQuarter Entity Extraction Tests:")
    for query in test_queries:
        template_query, entity_map = extractor.extract_and_normalize(query)
        
        # Find the time_period entity if it exists
        time_period = None
        for placeholder, info in entity_map.items():
            if info['type'] == 'time_period':
                time_period = info['normalized']
                break
                
        print(f"\nQuery: {query}")
        print(f"Extracted time period: {time_period}")
        print(f"Template query: {template_query}")

def test_query_cache_quarters():
    """Test the quarter handling in QueryCache."""
    # Initialize QueryCache
    cache = QueryCache(use_predefined_templates=False)
    
    # Add a test template
    template_query = "How many hours did {employee_0} work on {project_0} in {time_period_0}?"
    sql_query = """
    SELECT 
        e.name AS employee, 
        p.name AS project, 
        SUM(wh.hours) AS total_hours
    FROM 
        work_hours wh
    JOIN 
        employees e ON wh.employee_id = e.id
    JOIN 
        projects p ON wh.project_id = p.id
    WHERE 
        e.name = 'Bob' 
        AND p.name = 'Project X'
        AND wh.start_date >= '2025-01-01'
        AND wh.end_date <= '2025-03-31'
    GROUP BY 
        e.name, p.name
    """
    
    entity_map = {
        '{employee_0}': {'type': 'employee', 'value': 'Bob', 'normalized': 'Bob'},
        '{project_0}': {'type': 'project', 'value': 'Project X', 'normalized': 'Project X'},
        '{time_period_0}': {'type': 'time_period', 'value': 'Q1 2025', 'normalized': 'Q1 2025'}
    }
    
    cache.add_template(template_query, sql_query, entity_map)
    
    # Test with Q1 2025
    test_query = "How many hours did Alice work on Project Y in Q1 2025?"
    result = cache.process_query(test_query)
    
    print("\nQueryCache Test with Q1 2025:")
    print(f"Original query: {test_query}")
    print(f"Processed template query: {result['template_query']}")
    print(f"Matching template: {result['matching_template']}")
    print(f"Generated SQL: {result['sql_query']}")
    
    # Test with another quarter
    test_query = "How many hours did Alice work on Project Y in Q2 2025?"
    result = cache.process_query(test_query)
    
    print("\nQueryCache Test with Q2 2025:")
    print(f"Original query: {test_query}")
    print(f"Processed template query: {result['template_query']}")
    print(f"Matching template: {result['matching_template']}")
    print(f"Generated SQL: {result['sql_query']}")
    
    # Test with 'first quarter'
    test_query = "How many hours did Alice work on Project Y in first quarter of 2025?"
    result = cache.process_query(test_query)
    
    print("\nQueryCache Test with 'first quarter of 2025':")
    print(f"Original query: {test_query}")
    print(f"Processed template query: {result['template_query']}")
    print(f"Matching template: {result['matching_template']}")
    print(f"Generated SQL: {result['sql_query']}")

if __name__ == "__main__":
    test_quarter_entity_extraction()
    test_query_cache_quarters()