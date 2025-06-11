#!/usr/bin/env python3
"""
Test script for date_handler.py SQL replacement.
This tests SQL generation with quarter placeholders using date_handler.py.
"""

import sys
import logging
from datetime import datetime
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the parent directory to the Python path
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))

# Import date_handler.py
from date_handler import replace_time_period_placeholders, get_date_range, get_quarter_week_range

def test_sql_generation():
    """Test SQL generation with quarter placeholders."""
    
    # SQL template with direct time_period placeholder
    sql_template_direct = """
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
        e.name = 'Bob' 
        AND p.name = 'Project X'
        AND {time_period_0}
    GROUP BY 
        e.name, p.name
    """
    
    # SQL template with start/end placeholders
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
        e.name = 'Bob' 
        AND p.name = 'Project X'
        AND wh.start_date >= '{time_period_0}-start'
        AND wh.end_date <= '{time_period_0}-end'
    GROUP BY 
        e.name, p.name
    """
    
    # Test with different quarters
    quarters = ["Q1 2025", "Q2 2025", "Q3 2025", "Q4 2025"]
    
    print("\nSQL Generation Tests for Quarters:")
    for quarter in quarters:
        time_periods = {"{time_period_0}": quarter}
        
        # Test with direct placeholder
        sql_direct = replace_time_period_placeholders(sql_template_direct, time_periods)
        print(f"\nSQL with direct placeholder for {quarter}:")
        print(sql_direct)
        
        # Test with explicit placeholders
        sql_explicit = replace_time_period_placeholders(sql_template_explicit, time_periods)
        print(f"\nSQL with explicit placeholders for {quarter}:")
        print(sql_explicit)
        
    # Test with other time periods to make sure they still work
    other_periods = ["January 2025", "2025", "this month", "last month"]
    
    print("\nSQL Generation Tests for Other Periods:")
    for period in other_periods:
        time_periods = {"{time_period_0}": period}
        
        # Test with direct placeholder
        sql_direct = replace_time_period_placeholders(sql_template_direct, time_periods)
        print(f"\nSQL with direct placeholder for {period}:")
        print(sql_direct)
        
        # Test with explicit placeholders
        sql_explicit = replace_time_period_placeholders(sql_template_explicit, time_periods)
        print(f"\nSQL with explicit placeholders for {period}:")
        print(sql_explicit)

if __name__ == "__main__":
    test_sql_generation()