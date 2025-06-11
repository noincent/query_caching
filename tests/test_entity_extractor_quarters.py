#!/usr/bin/env python3
"""
Test script for quarter entity extraction.
This script tests the quarter entity extraction in EntityExtractor.
"""

import sys
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the parent directory to the Python path
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))

# Import the EntityExtractor
from src.utils.entity_extractor import EntityExtractor

def test_quarter_entity_extraction():
    """Test the quarter entity extraction."""
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
        print(f"Entity map: {entity_map}")

if __name__ == "__main__":
    test_quarter_entity_extraction()