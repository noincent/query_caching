"""
Template Library Module

This module provides predefined templates for common query patterns
to improve the reliability of the query caching system.
"""

import logging
import os
from typing import Dict, List, Any
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TemplateLibrary:
    """
    Class for managing predefined templates for common query patterns.
    """
    
    def __init__(self, include_wtl_templates=False):
        """
        Initialize the template library with predefined templates.
        
        Args:
            include_wtl_templates: Whether to include WTL-specific templates
        """
        self.templates = self._initialize_templates()
        
        # Add WTL templates if specified
        if include_wtl_templates:
            try:
                from src.core.wtl_templates import load_wtl_templates
                wtl_templates = load_wtl_templates()
                self.templates.extend(wtl_templates)
                logger.info(f"Added {len(wtl_templates)} WTL-specific templates")
            except ImportError as e:
                logger.warning(f"Could not load WTL templates: {e}")
        
        logger.info(f"Template library initialized with {len(self.templates)} templates")
    
    def _initialize_templates(self) -> List[Dict[str, Any]]:
        """
        Initialize the library with predefined templates.
        
        Returns:
            List of template dictionaries
        """
        templates = []
        
        # Template 1: Hours worked by employee on project in time period
        templates.append({
            'template_query': "How many hours did {employee_0} work on the {project_0} project in {time_period_0}?",
            'sql_template': "SELECT SUM(hour) FROM work_hour WHERE employee = '{employee_0}' AND project = '{project_0}' AND NOT (start_date > '{time_period_0}-end' OR end_date < '{time_period_0}-start')",
            'entity_map': {
                '{employee_0}': {'type': 'employee', 'value': 'PLACEHOLDER', 'normalized': 'PLACEHOLDER'},
                '{project_0}': {'type': 'project', 'value': 'PLACEHOLDER', 'normalized': 'PLACEHOLDER'},
                '{time_period_0}': {'type': 'time_period', 'value': 'PLACEHOLDER', 'normalized': 'PLACEHOLDER'}
            },
            'metadata': {
                'source': 'predefined',
                'description': 'Query for total hours worked by an employee on a specific project in a time period'
            }
        })
        
        # Template 2: Projects an employee worked on in time period
        templates.append({
            'template_query': "What projects did {employee_0} work on in {time_period_0}?",
            'sql_template': "SELECT DISTINCT project FROM work_hour WHERE employee = '{employee_0}' AND NOT (start_date > '{time_period_0}-end' OR end_date < '{time_period_0}-start')",
            'entity_map': {
                '{employee_0}': {'type': 'employee', 'value': 'PLACEHOLDER', 'normalized': 'PLACEHOLDER'},
                '{time_period_0}': {'type': 'time_period', 'value': 'PLACEHOLDER', 'normalized': 'PLACEHOLDER'}
            },
            'metadata': {
                'source': 'predefined',
                'description': 'Query for projects an employee worked on in a time period'
            }
        })
        
        # Template 3: Employees who worked on a project in time period
        templates.append({
            'template_query': "Who worked on the {project_0} project in {time_period_0}?",
            'sql_template': "SELECT DISTINCT employee FROM work_hour WHERE project = '{project_0}' AND NOT (start_date > '{time_period_0}-end' OR end_date < '{time_period_0}-start')",
            'entity_map': {
                '{project_0}': {'type': 'project', 'value': 'PLACEHOLDER', 'normalized': 'PLACEHOLDER'},
                '{time_period_0}': {'type': 'time_period', 'value': 'PLACEHOLDER', 'normalized': 'PLACEHOLDER'}
            },
            'metadata': {
                'source': 'predefined',
                'description': 'Query for employees who worked on a specific project in a time period'
            }
        })
        
        # Template 4: Total hours worked by employee in time period
        templates.append({
            'template_query': "How many hours did {employee_0} work in {time_period_0}?",
            'sql_template': "SELECT SUM(hour) FROM work_hour WHERE employee = '{employee_0}' AND NOT (start_date > '{time_period_0}-end' OR end_date < '{time_period_0}-start')",
            'entity_map': {
                '{employee_0}': {'type': 'employee', 'value': 'PLACEHOLDER', 'normalized': 'PLACEHOLDER'},
                '{time_period_0}': {'type': 'time_period', 'value': 'PLACEHOLDER', 'normalized': 'PLACEHOLDER'}
            },
            'metadata': {
                'source': 'predefined',
                'description': 'Query for total hours worked by an employee in a time period'
            }
        })
        
        # Template 5: Total hours worked on project in time period
        templates.append({
            'template_query': "How many hours were spent on the {project_0} project in {time_period_0}?",
            'sql_template': "SELECT SUM(hour) FROM work_hour WHERE project = '{project_0}' AND NOT (start_date > '{time_period_0}-end' OR end_date < '{time_period_0}-start')",
            'entity_map': {
                '{project_0}': {'type': 'project', 'value': 'PLACEHOLDER', 'normalized': 'PLACEHOLDER'},
                '{time_period_0}': {'type': 'time_period', 'value': 'PLACEHOLDER', 'normalized': 'PLACEHOLDER'}
            },
            'metadata': {
                'source': 'predefined',
                'description': 'Query for total hours spent on a project in a time period'
            }
        })
        
        # Template 6: Compare hours between two employees
        templates.append({
            'template_query': "Compare the hours worked by {employee_0} and {employee_1} in {time_period_0}",
            'sql_template': "SELECT employee, SUM(hour) as total_hours FROM work_hour WHERE employee IN ('{employee_0}', '{employee_1}') AND NOT (start_date > '{time_period_0}-end' OR end_date < '{time_period_0}-start') GROUP BY employee",
            'entity_map': {
                '{employee_0}': {'type': 'employee', 'value': 'PLACEHOLDER', 'normalized': 'PLACEHOLDER'},
                '{employee_1}': {'type': 'employee', 'value': 'PLACEHOLDER', 'normalized': 'PLACEHOLDER'},
                '{time_period_0}': {'type': 'time_period', 'value': 'PLACEHOLDER', 'normalized': 'PLACEHOLDER'}
            },
            'metadata': {
                'source': 'predefined',
                'description': 'Query to compare hours worked by two employees in a time period'
            }
        })
        
        # Template 7: Compare hours between two projects
        templates.append({
            'template_query': "Compare the hours spent on {project_0} and {project_1} in {time_period_0}",
            'sql_template': "SELECT project, SUM(hour) as total_hours FROM work_hour WHERE project IN ('{project_0}', '{project_1}') AND NOT (start_date > '{time_period_0}-end' OR end_date < '{time_period_0}-start') GROUP BY project",
            'entity_map': {
                '{project_0}': {'type': 'project', 'value': 'PLACEHOLDER', 'normalized': 'PLACEHOLDER'},
                '{project_1}': {'type': 'project', 'value': 'PLACEHOLDER', 'normalized': 'PLACEHOLDER'},
                '{time_period_0}': {'type': 'time_period', 'value': 'PLACEHOLDER', 'normalized': 'PLACEHOLDER'}
            },
            'metadata': {
                'source': 'predefined',
                'description': 'Query to compare hours spent on two projects in a time period'
            }
        })
        
        # Template 8: Projects with most hours in time period
        templates.append({
            'template_query': "Which projects had the most hours in {time_period_0}?",
            'sql_template': "SELECT project, SUM(hour) as total_hours FROM work_hour WHERE NOT (start_date > '{time_period_0}-end' OR end_date < '{time_period_0}-start') GROUP BY project ORDER BY total_hours DESC LIMIT 5",
            'entity_map': {
                '{time_period_0}': {'type': 'time_period', 'value': 'PLACEHOLDER', 'normalized': 'PLACEHOLDER'}
            },
            'metadata': {
                'source': 'predefined',
                'description': 'Query for projects with the most hours in a time period'
            }
        })
        
        # Template 9: Employees with most hours in time period
        templates.append({
            'template_query': "Which employees worked the most hours in {time_period_0}?",
            'sql_template': "SELECT employee, SUM(hour) as total_hours FROM work_hour WHERE NOT (start_date > '{time_period_0}-end' OR end_date < '{time_period_0}-start') GROUP BY employee ORDER BY total_hours DESC LIMIT 5",
            'entity_map': {
                '{time_period_0}': {'type': 'time_period', 'value': 'PLACEHOLDER', 'normalized': 'PLACEHOLDER'}
            },
            'metadata': {
                'source': 'predefined',
                'description': 'Query for employees with the most hours in a time period'
            }
        })
        
        # Template 10: List all employee hours across projects for a time period
        templates.append({
            'template_query': "List all hours for {employee_0} by project in {time_period_0}",
            'sql_template': "SELECT project, SUM(hour) as hours FROM work_hour WHERE employee = '{employee_0}' AND NOT (start_date > '{time_period_0}-end' OR end_date < '{time_period_0}-start') GROUP BY project ORDER BY hours DESC",
            'entity_map': {
                '{employee_0}': {'type': 'employee', 'value': 'PLACEHOLDER', 'normalized': 'PLACEHOLDER'},
                '{time_period_0}': {'type': 'time_period', 'value': 'PLACEHOLDER', 'normalized': 'PLACEHOLDER'}
            },
            'metadata': {
                'source': 'predefined',
                'description': 'Query for all hours worked by an employee across different projects in a time period'
            }
        })
        
        return templates
    
    def get_templates(self) -> List[Dict[str, Any]]:
        """
        Get all predefined templates.
        
        Returns:
            List of template dictionaries
        """
        return self.templates
    
    def get_template_by_index(self, index: int) -> Dict[str, Any]:
        """
        Get a template by index.
        
        Args:
            index: Index of the template to retrieve
            
        Returns:
            Template dictionary
        
        Raises:
            IndexError: If index is out of range
        """
        if 0 <= index < len(self.templates):
            return self.templates[index]
        else:
            raise IndexError(f"Template index {index} out of range (0-{len(self.templates)-1})")
    
    def get_templates_by_entity_types(self, entity_types: List[str]) -> List[Dict[str, Any]]:
        """
        Get templates that use all the specified entity types.
        
        Args:
            entity_types: List of entity types to match
            
        Returns:
            List of template dictionaries
        """
        matching_templates = []
        
        for template in self.templates:
            # Get entity types in the template
            template_entity_types = set()
            for entity_info in template['entity_map'].values():
                template_entity_types.add(entity_info['type'])
            
            # Check if all required entity types are present
            if all(entity_type in template_entity_types for entity_type in entity_types):
                matching_templates.append(template)
        
        return matching_templates