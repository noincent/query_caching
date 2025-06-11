#!/usr/bin/env python3
"""
Test script for quarter date handling.
This script tests the quarter date handling in date_handler.py.
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import from the local directory
from date_handler import get_date_range, replace_time_period_placeholders

def test_quarter_dates():
    """Test the quarter date ranges."""
    
    # Test get_date_range with quarters
    q1_2025 = get_date_range("Q1 2025")
    q2_2025 = get_date_range("Q2 2025")
    q3_2025 = get_date_range("Q3 2025")
    q4_2025 = get_date_range("Q4 2025")
    
    print("\nQuarter Date Ranges (First Monday - Last Sunday):")
    print(f"Q1 2025: {q1_2025[0]} to {q1_2025[1]}")
    print(f"Q2 2025: {q2_2025[0]} to {q2_2025[1]}")
    print(f"Q3 2025: {q3_2025[0]} to {q3_2025[1]}")
    print(f"Q4 2025: {q4_2025[0]} to {q4_2025[1]}")
    
    # Test replace_time_period_placeholders with quarters
    sql_template = """
    SELECT 
        e.name, 
        p.name, 
        SUM(wh.hours) 
    FROM 
        work_hours wh
    JOIN 
        employees e ON wh.employee_id = e.id
    JOIN 
        projects p ON wh.project_id = p.id
    WHERE 
        e.name = '{employee_0}' 
        AND p.name = '{project_0}'
        AND {time_period_0}
    GROUP BY 
        e.name, p.name
    """
    
    time_periods = {
        "{time_period_0}": "Q1 2025"
    }
    
    sql_query = replace_time_period_placeholders(sql_template, time_periods)
    
    print("\nSQL Query with Q1 2025:")
    print(sql_query)
    
    # Test with explicit start/end placeholders
    sql_template_explicit = """
    SELECT 
        e.name, 
        p.name, 
        SUM(wh.hours) 
    FROM 
        work_hours wh
    JOIN 
        employees e ON wh.employee_id = e.id
    JOIN 
        projects p ON wh.project_id = p.id
    WHERE 
        e.name = '{employee_0}' 
        AND p.name = '{project_0}'
        AND wh.start_date >= '{time_period_0}-start'
        AND wh.end_date <= '{time_period_0}-end'
    GROUP BY 
        e.name, p.name
    """
    
    sql_query_explicit = replace_time_period_placeholders(sql_template_explicit, time_periods)
    
    print("\nSQL Query with Explicit Q1 2025 Start/End:")
    print(sql_query_explicit)
    
    # Test this quarter and last quarter
    this_quarter = get_date_range("this quarter")
    last_quarter = get_date_range("last quarter")
    
    current_quarter = (datetime.now().month - 1) // 3 + 1
    
    print("\nRelative Quarter Dates:")
    print(f"This Quarter (Q{current_quarter} {datetime.now().year}): {this_quarter[0]} to {this_quarter[1]}")
    
    last_q = current_quarter - 1
    last_q_year = datetime.now().year
    if last_q == 0:
        last_q = 4
        last_q_year -= 1
        
    print(f"Last Quarter (Q{last_q} {last_q_year}): {last_quarter[0]} to {last_quarter[1]}")

if __name__ == "__main__":
    test_quarter_dates()