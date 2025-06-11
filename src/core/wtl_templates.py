"""
WTL-Specific Templates Module

This module provides specialized templates for the WTL Employee Tracker database
that are optimized for common query patterns in the employee work hour tracking system.
"""

import logging
from typing import Dict, List, Any

# Set up logging
logger = logging.getLogger(__name__)

def load_wtl_templates() -> List[Dict[str, Any]]:
    """
    Load WTL-specific templates for employee work hour tracking queries.
    
    Returns:
        List of WTL-specific template dictionaries
    """
    templates = []
    
    # Template 1: Employee hours by project and time period
    templates.append({
        'template_query': "How many hours did {employee_0} work on {project_0} in {time_period_0}?",
        'sql_template': """
            SELECT SUM(wh.hour) as total_hours 
            FROM work_hour wh 
            JOIN employee e ON e.uuid = wh.employee_id 
            JOIN project p ON p.uuid = wh.project_id 
            WHERE e.name = '{employee_0}' 
            AND p.name LIKE '%{project_0}%' 
            AND wh.start_date >= '{time_period_0}_start' 
            AND wh.end_date <= '{time_period_0}_end'
        """,
        'entity_map': {
            '{employee_0}': {'type': 'employee', 'value': 'PLACEHOLDER', 'normalized': 'PLACEHOLDER'},
            '{project_0}': {'type': 'project', 'value': 'PLACEHOLDER', 'normalized': 'PLACEHOLDER'},
            '{time_period_0}': {'type': 'time_period', 'value': 'PLACEHOLDER', 'normalized': 'PLACEHOLDER'}
        },
        'metadata': {
            'source': 'wtl_predefined',
            'description': 'Get total hours worked by an employee on a specific project in a time period',
            'database_schema': 'wtl_employee_tracker'
        }
    })
    
    # Template 2: Employee productivity by department
    templates.append({
        'template_query': "How many hours did employees in {department_0} work in {time_period_0}?",
        'sql_template': """
            SELECT e.name, SUM(wh.hour) as total_hours
            FROM work_hour wh 
            JOIN employee e ON e.uuid = wh.employee_id 
            WHERE e.department = '{department_0}' 
            AND wh.start_date >= '{time_period_0}_start' 
            AND wh.end_date <= '{time_period_0}_end'
            GROUP BY e.name 
            ORDER BY total_hours DESC
        """,
        'entity_map': {
            '{department_0}': {'type': 'department', 'value': 'PLACEHOLDER', 'normalized': 'PLACEHOLDER'},
            '{time_period_0}': {'type': 'time_period', 'value': 'PLACEHOLDER', 'normalized': 'PLACEHOLDER'}
        },
        'metadata': {
            'source': 'wtl_predefined',
            'description': 'Get total hours worked by all employees in a department during a time period',
            'database_schema': 'wtl_employee_tracker'
        }
    })
    
    logger.info(f"Loaded {len(templates)} WTL-specific templates")
    return templates

def get_wtl_entity_dictionary() -> Dict[str, List[str]]:
    """
    Get WTL-specific entity dictionary for better entity recognition.
    
    Returns:
        Dictionary mapping entity types to known values
    """
    return {
        'department': [
            'Engineering', 'Marketing', 'Sales', 'HR', 'Finance', 'Operations'
        ],
        'time_period': [
            'this week', 'last week', 'this month', 'last month', 'this quarter',
            'last quarter', 'this year', 'last year', 'Q1', 'Q2', 'Q3', 'Q4'
        ]
    }