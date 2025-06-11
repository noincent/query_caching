#!/usr/bin/env python3
"""
Test script for date replacement.
Tests the date handler's replacement functionality.
"""

import sys
import re
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import from the local directory
from date_handler import get_date_range, replace_time_period_placeholders

def test_sql_with_quarters():
    """Test SQL query generation with quarter placeholders."""
    
    # Demo SQL queries with different placeholder styles
    sql_queries = [
        # Direct period reference
        "SELECT * FROM work_hour WHERE period = '{time_period_0}'",
        
        # Start/end date format
        "SELECT * FROM work_hour WHERE start_date >= '{time_period_0}-start' AND end_date <= '{time_period_0}-end'",
        
        # NOT format 
        "SELECT * FROM work_hour WHERE NOT (start_date > '{time_period_0}-end' OR end_date < '{time_period_0}-start')",
        
        # Complex query
        """
        SELECT e.department, COALESCE(SUM(w.hour), 0) as total_hours 
        FROM work_hour w 
        JOIN employee e ON w.employee_id = e.uuid 
        WHERE e.department IN ('物业管理部', '财务管理中心') 
        AND NOT (w.start_date > '{time_period_0}-end' OR w.end_date < '{time_period_0}-start') 
        GROUP BY e.department
        """,
        
        # Complex query with another time period
        """
        SELECT '{time_period_0}' as period, COALESCE(SUM(w.hour), 0) as total_hours 
        FROM work_hour w 
        JOIN employee e ON w.employee_id = e.uuid 
        WHERE e.department = '物业管理部' 
        AND NOT (w.start_date > '{time_period_0}-end' OR w.end_date < '{time_period_0}-start') 
        UNION 
        SELECT '{time_period_1}' as period, COALESCE(SUM(w.hour), 0) as total_hours 
        FROM work_hour w 
        JOIN employee e ON w.employee_id = e.uuid 
        WHERE e.department = '物业管理部' 
        AND NOT (w.start_date > '{time_period_1}-end' OR w.end_date < '{time_period_1}-start')
        """
    ]
    
    # Test with different quarters
    quarters = {
        "{time_period_0}": "Q3 2023",
        "{time_period_1}": "Q4 2023"
    }
    
    print("\nSQL Generation Tests for Quarters:")
    for i, sql in enumerate(sql_queries):
        print(f"\nOriginal SQL {i+1}:")
        print(sql)
        
        # Replace placeholders
        processed_sql = replace_time_period_placeholders(sql, quarters)
        
        print(f"\nProcessed SQL {i+1}:")
        print(processed_sql)
        
        # Verify no raw placeholders remain
        if '-start' in processed_sql or '-end' in processed_sql:
            has_placeholder = re.search(r"'[^']*time_period[^']*-(?:start|end)'", processed_sql)
            if has_placeholder:
                print(f"ERROR: Unreplaced placeholder found: {has_placeholder.group(0)}")
            else:
                print("OK: All placeholders properly replaced")
        else:
            print("OK: No placeholders present in the result")

if __name__ == "__main__":
    test_sql_with_quarters()