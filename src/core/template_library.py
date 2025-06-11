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
        Initialize the library with predefined templates that comply with
        the WTL MySQL 8.0 instruction set (string‑matching rules, joins, etc.).
        """
        emp_like  = "(T1.name LIKE '%{employee}%' OR T1.alias LIKE '%{employee}%' OR T1.name LIKE '%{employee_normalized}%')"
        proj_like = "(P.project_reference_id LIKE '%{project}%' OR P.name LIKE '%{project}%')"

        templates: List[Dict[str, Any]] = [

            # 1 ── Hours an employee spent on a project within a period
            {
                "template_query":
                    "How many hours did {employee_0} work on the {project_0} project in {time_period_0}?",
                "sql_template": f"""
    SELECT
        SUM(T2.hour) AS total_hours
    FROM work_hour            T2
    INNER JOIN employee        T1 ON T1.uuid = T2.employee_id
    INNER JOIN project         P  ON P.uuid  = T2.project_id
    WHERE
        {emp_like.format(employee='{employee_0}', employee_normalized='{employee_0_normalized}')}
        AND {proj_like.format(project='{project_0}')}
        AND NOT (T2.start_date > '{{time_period_0}}-end'
            OR   T2.end_date   < '{{time_period_0}}-start')
    """.strip(),
                "entity_map": {
                    "{employee_0}": {"type": "employee", "value": "PLACEHOLDER", "normalized": "PLACEHOLDER"},
                    "{project_0}":  {"type": "project",  "value": "PLACEHOLDER", "normalized": "PLACEHOLDER"},
                    "{time_period_0}": {"type": "time_period", "value": "PLACEHOLDER", "normalized": "PLACEHOLDER"}
                },
                "metadata": {"source": "predefined",
                            "description": "Total hours an employee logged on a project in a period"}
            },

            # 2 ── Projects an employee worked on in a period
            {
                "template_query":
                    "What projects did {employee_0} work on in {time_period_0}?",
                "sql_template": f"""
    SELECT DISTINCT
        P.project_reference_id,
        P.name
    FROM work_hour            T2
    INNER JOIN employee        T1 ON T1.uuid = T2.employee_id
    LEFT  JOIN project         P  ON P.uuid  = T2.project_id
    WHERE
        {emp_like.format(employee='{employee_0}', employee_normalized='{employee_0_normalized}')}
        AND NOT (T2.start_date > '{{time_period_0}}-end'
            OR   T2.end_date   < '{{time_period_0}}-start')
    """.strip(),
                "entity_map": {
                    "{employee_0}": {"type": "employee", "value": "PLACEHOLDER", "normalized": "PLACEHOLDER"},
                    "{time_period_0}": {"type": "time_period", "value": "PLACEHOLDER", "normalized": "PLACEHOLDER"}
                },
                "metadata": {"source": "predefined",
                            "description": "Projects an employee touched during a period"}
            },

            # 3 ── Employees who worked on a project in a period
            {
                "template_query":
                    "Who worked on the {project_0} project in {time_period_0}?",
                "sql_template": f"""
    SELECT DISTINCT
        T1.name
    FROM work_hour            T2
    INNER JOIN employee        T1 ON T1.uuid = T2.employee_id
    INNER JOIN project         P  ON P.uuid  = T2.project_id
    WHERE
        {proj_like.format(project='{project_0}')}
        AND NOT (T2.start_date > '{{time_period_0}}-end'
            OR   T2.end_date   < '{{time_period_0}}-start')
    """.strip(),
                "entity_map": {
                    "{project_0}": {"type": "project", "value": "PLACEHOLDER", "normalized": "PLACEHOLDER"},
                    "{time_period_0}": {"type": "time_period", "value": "PLACEHOLDER", "normalized": "PLACEHOLDER"}
                },
                "metadata": {"source": "predefined",
                            "description": "Employees who logged hours on a project within a period"}
            },

            # 4 ── Total hours an employee worked in a period
            {
                "template_query":
                    "How many hours did {employee_0} work in {time_period_0}?",
                "sql_template": f"""
    SELECT
        SUM(T2.hour) AS total_hours
    FROM work_hour            T2
    INNER JOIN employee        T1 ON T1.uuid = T2.employee_id
    WHERE
        {emp_like.format(employee='{employee_0}', employee_normalized='{employee_0_normalized}')}
        AND NOT (T2.start_date > '{{time_period_0}}-end'
            OR   T2.end_date   < '{{time_period_0}}-start')
    """.strip(),
                "entity_map": {
                    "{employee_0}": {"type": "employee", "value": "PLACEHOLDER", "normalized": "PLACEHOLDER"},
                    "{time_period_0}": {"type": "time_period", "value": "PLACEHOLDER", "normalized": "PLACEHOLDER"}
                },
                "metadata": {"source": "predefined",
                            "description": "Total hours an employee logged in a period"}
            },

            # 5 ── Total hours spent on a project in a period
            {
                "template_query":
                    "How many hours were spent on the {project_0} project in {time_period_0}?",
                "sql_template": f"""
    SELECT
        SUM(T2.hour) AS total_hours
    FROM work_hour            T2
    INNER JOIN project         P  ON P.uuid = T2.project_id
    WHERE
        {proj_like.format(project='{project_0}')}
        AND NOT (T2.start_date > '{{time_period_0}}-end'
            OR   T2.end_date   < '{{time_period_0}}-start')
    """.strip(),
                "entity_map": {
                    "{project_0}": {"type": "project", "value": "PLACEHOLDER", "normalized": "PLACEHOLDER"},
                    "{time_period_0}": {"type": "time_period", "value": "PLACEHOLDER", "normalized": "PLACEHOLDER"}
                },
                "metadata": {"source": "predefined",
                            "description": "Total hours spent on a project in a period"}
            },

            # 6 ── Compare hours between two employees in a period
            {
                "template_query":
                    "Compare the hours worked by {employee_0} and {employee_1} in {time_period_0}",
                "sql_template": f"""
    SELECT
        T1.name,
        SUM(T2.hour) AS total_hours
    FROM work_hour            T2
    INNER JOIN employee        T1 ON T1.uuid = T2.employee_id
    WHERE
        ({emp_like.format(employee='{employee_0}', employee_normalized='{employee_0_normalized}')}
        OR {emp_like.format(employee='{employee_1}', employee_normalized='{employee_1_normalized}')})
        AND NOT (T2.start_date > '{{time_period_0}}-end'
            OR   T2.end_date   < '{{time_period_0}}-start')
    GROUP BY
        T1.uuid, T1.name
    """.strip(),
                "entity_map": {
                    "{employee_0}": {"type": "employee", "value": "PLACEHOLDER", "normalized": "PLACEHOLDER"},
                    "{employee_1}": {"type": "employee", "value": "PLACEHOLDER", "normalized": "PLACEHOLDER"},
                    "{time_period_0}": {"type": "time_period", "value": "PLACEHOLDER", "normalized": "PLACEHOLDER"}
                },
                "metadata": {"source": "predefined",
                            "description": "Compare two employees’ hours over a period"}
            },

            # 7 ── Compare hours between two projects in a period
            {
                "template_query":
                    "Compare the hours spent on {project_0} and {project_1} in {time_period_0}",
                "sql_template": f"""
    SELECT
        P.project_reference_id,
        P.name,
        SUM(T2.hour) AS total_hours
    FROM work_hour            T2
    INNER JOIN project         P ON P.uuid = T2.project_id
    WHERE
        ({proj_like.format(project='{project_0}')}
        OR {proj_like.format(project='{project_1}')}
        )
        AND NOT (T2.start_date > '{{time_period_0}}-end'
            OR   T2.end_date   < '{{time_period_0}}-start')
    GROUP BY
        P.uuid, P.project_reference_id, P.name
    """.strip(),
                "entity_map": {
                    "{project_0}": {"type": "project", "value": "PLACEHOLDER", "normalized": "PLACEHOLDER"},
                    "{project_1}": {"type": "project", "value": "PLACEHOLDER", "normalized": "PLACEHOLDER"},
                    "{time_period_0}": {"type": "time_period", "value": "PLACEHOLDER", "normalized": "PLACEHOLDER"}
                },
                "metadata": {"source": "predefined",
                            "description": "Compare two projects’ hours over a period"}
            },

            # 8 ── Top‑N projects by hours in a period
            {
                "template_query":
                    "Which projects had the most hours in {time_period_0}?",
                "sql_template": f"""
    SELECT
        P.project_reference_id,
        P.name,
        SUM(T2.hour) AS total_hours
    FROM work_hour            T2
    LEFT JOIN  project         P ON P.uuid = T2.project_id
    WHERE
        NOT (T2.start_date > '{{time_period_0}}-end'
        OR   T2.end_date   < '{{time_period_0}}-start')
    GROUP BY
        P.uuid, P.project_reference_id, P.name
    ORDER BY
        total_hours DESC
    LIMIT 5
    """.strip(),
                "entity_map": {
                    "{time_period_0}": {"type": "time_period", "value": "PLACEHOLDER", "normalized": "PLACEHOLDER"}
                },
                "metadata": {"source": "predefined",
                            "description": "Projects with most hours in a period (top 5)"}
            },

            # 9 ── Top‑N employees by hours in a period
            {
                "template_query":
                    "Which employees worked the most hours in {time_period_0}?",
                "sql_template": f"""
    SELECT
        T1.name,
        SUM(T2.hour) AS total_hours
    FROM work_hour            T2
    INNER JOIN employee        T1 ON T1.uuid = T2.employee_id
    WHERE
        NOT (T2.start_date > '{{time_period_0}}-end'
        OR   T2.end_date   < '{{time_period_0}}-start')
    GROUP BY
        T1.uuid, T1.name
    ORDER BY
        total_hours DESC
    LIMIT 5
    """.strip(),
                "entity_map": {
                    "{time_period_0}": {"type": "time_period", "value": "PLACEHOLDER", "normalized": "PLACEHOLDER"}
                },
                "metadata": {"source": "predefined",
                            "description": "Employees with most hours in a period (top 5)"}
            },

            # 10 ── Employee hours split by project in a period
            {
                "template_query":
                    "List all hours for {employee_0} by project in {time_period_0}",
                "sql_template": f"""
    SELECT
        P.project_reference_id,
        P.name,
        SUM(T2.hour) AS hours
    FROM work_hour            T2
    INNER JOIN employee        T1 ON T1.uuid = T2.employee_id
    LEFT  JOIN project         P  ON P.uuid  = T2.project_id
    WHERE
        {emp_like.format(employee='{employee_0}', employee_normalized='{employee_0_normalized}')}
        AND NOT (T2.start_date > '{{time_period_0}}-end'
            OR   T2.end_date   < '{{time_period_0}}-start')
    GROUP BY
        P.uuid, P.project_reference_id, P.name
    ORDER BY
        hours DESC
    """.strip(),
                "entity_map": {
                    "{employee_0}": {"type": "employee", "value": "PLACEHOLDER", "normalized": "PLACEHOLDER"},
                    "{time_period_0}": {"type": "time_period", "value": "PLACEHOLDER", "normalized": "PLACEHOLDER"}
                },
                "metadata": {"source": "predefined",
                            "description": "Per‑project hour breakdown for an employee in a period"}
            }
        ]

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