"""
SQL Validator Module

This module provides functionality for validating and analyzing SQL templates
to ensure they are safe and syntactically correct.
"""

import re
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

class SQLValidator:
    """Validate and analyze SQL templates."""
    
    def __init__(self):
        self.sql_keywords = {
            'required': ['SELECT', 'FROM'],
            'optional': ['WHERE', 'GROUP BY', 'ORDER BY', 'HAVING', 'LIMIT', 'JOIN', 'LEFT JOIN', 'RIGHT JOIN', 'INNER JOIN'],
            'dangerous': ['DROP', 'DELETE', 'TRUNCATE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE']
        }
    
    def validate_sql_template(self, sql: str, entity_map: Dict[str, Dict] = None) -> Tuple[bool, List[str]]:
        """
        Validate SQL template syntax and safety.
        Returns: (is_valid, list_of_issues)
        """
        issues = []
        sql_upper = sql.upper()
        
        # Check for dangerous operations
        for keyword in self.sql_keywords['dangerous']:
            if keyword in sql_upper:
                issues.append(f"Dangerous operation detected: {keyword}")
        
        # Check for required keywords
        for keyword in self.sql_keywords['required']:
            if keyword not in sql_upper:
                issues.append(f"Missing required keyword: {keyword}")
        
        # Check placeholder format and matching
        if entity_map:
            placeholders_in_sql = re.findall(r'\{[^}]+\}', sql)
            placeholders_in_map = set(entity_map.keys())
            
            # Check for unmatched placeholders
            for placeholder in placeholders_in_sql:
                if placeholder not in placeholders_in_map:
                    issues.append(f"Placeholder {placeholder} in SQL not found in entity map")
            
            # Check for unused placeholders
            for placeholder in placeholders_in_map:
                if placeholder not in sql:
                    issues.append(f"Entity map placeholder {placeholder} not used in SQL")
        
        # Basic syntax checks
        if not self._check_balanced_parentheses(sql):
            issues.append("Unbalanced parentheses")
        
        if not self._check_balanced_quotes(sql):
            issues.append("Unbalanced quotes")
        
        return len(issues) == 0, issues
    
    def _check_balanced_parentheses(self, sql: str) -> bool:
        """Check if parentheses are balanced."""
        count = 0
        for char in sql:
            if char == '(':
                count += 1
            elif char == ')':
                count -= 1
            if count < 0:
                return False
        return count == 0
    
    def _check_balanced_quotes(self, sql: str) -> bool:
        """Check if quotes are balanced."""
        single_quotes = sql.count("'") % 2 == 0
        double_quotes = sql.count('"') % 2 == 0
        return single_quotes and double_quotes
    
    def extract_table_references(self, sql: str) -> List[str]:
        """Extract table names from SQL."""
        tables = []
        
        # Pattern for FROM clause
        from_pattern = r'FROM\s+([^\s,]+)'
        matches = re.finditer(from_pattern, sql, re.IGNORECASE)
        tables.extend([match.group(1) for match in matches])
        
        # Pattern for JOIN clauses
        join_pattern = r'(?:LEFT|RIGHT|INNER|OUTER)?\s*JOIN\s+([^\s]+)'
        matches = re.finditer(join_pattern, sql, re.IGNORECASE)
        tables.extend([match.group(1) for match in matches])
        
        # Clean table names (remove aliases)
        cleaned_tables = []
        for table in tables:
            # Remove alias if present (e.g., "table_name t" -> "table_name")
            table_parts = table.split()
            if table_parts:
                cleaned_tables.append(table_parts[0])
        
        return list(set(cleaned_tables))