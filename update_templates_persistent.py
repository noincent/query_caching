#!/usr/bin/env python3
"""
Update templates script that ensures templates are properly updated and saved.
This script updates existing templates to use the new weekly date approach.
"""

import json
import logging
import os
import pickle
import numpy as np
from pathlib import Path

# Custom JSON encoder to handle numpy arrays and other non-serializable types
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, set):
            return list(obj)
        return super().default(obj)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def update_templates(templates_path):
    """
    Update templates to use the new weekly date approach.
    
    Args:
        templates_path: Path to the templates file
    """
    # Load templates
    template_data = None
    if templates_path.endswith('.json'):
        with open(templates_path, 'r') as f:
            template_data = json.load(f)
    else:
        with open(templates_path, 'rb') as f:
            template_data = pickle.load(f)
    
    if not template_data or 'templates' not in template_data:
        logger.error(f"Invalid template data: {template_data}")
        return
    
    templates = template_data['templates']
    logger.info(f"Loaded {len(templates)} templates")
    
    # Count templates that need updating
    simple_period_templates = []
    start_end_templates = []
    other_templates = []
    
    for i, template in enumerate(templates):
        sql_template = template.get('sql_template', '')
        
        # Check if template uses 'period = ' directly
        if ' period = ' in sql_template or "period='" in sql_template:
            simple_period_templates.append(i)
        # Check if template uses start/end approach
        elif '{time_period_' in sql_template and ('-start' in sql_template or '-end' in sql_template):
            start_end_templates.append(i)
        else:
            other_templates.append(i)
    
    logger.info(f"Found {len(simple_period_templates)} templates using 'period = ' directly")
    logger.info(f"Found {len(start_end_templates)} templates already using start/end approach")
    logger.info(f"Found {len(other_templates)} templates with other patterns")
    
    # Update templates
    updated_templates = 0
    for i in simple_period_templates:
        template = templates[i]
        old_sql = template['sql_template']
        
        # Find all time_period placeholders used in this template
        time_periods = []
        for placeholder, entity_info in template['entity_map'].items():
            if isinstance(entity_info, dict) and entity_info.get('type') == 'time_period':
                time_periods.append(placeholder)
        
        # Replace 'period = {time_period_X}' with the new approach
        updated_sql = old_sql
        for time_period in time_periods:
            period_clause = f"period = '{time_period}'"
            if period_clause in updated_sql:
                new_clause = f"NOT (start_date > '{time_period}-end' OR end_date < '{time_period}-start')"
                updated_sql = updated_sql.replace(period_clause, new_clause)
                
            # Also check for other variants
            period_clause = f"period='{time_period}'"
            if period_clause in updated_sql:
                new_clause = f"NOT (start_date > '{time_period}-end' OR end_date < '{time_period}-start')"
                updated_sql = updated_sql.replace(period_clause, new_clause)
        
        if updated_sql != old_sql:
            templates[i]['sql_template'] = updated_sql
            updated_templates += 1
            logger.info(f"Updated template {i+1}: {template['template_query']}")
    
    logger.info(f"Updated {updated_templates} templates")
    
    # Save updated templates
    if templates_path.endswith('.json'):
        with open(templates_path, 'w') as f:
            json.dump(template_data, f, indent=2, cls=CustomJSONEncoder)
    else:
        with open(templates_path, 'wb') as f:
            pickle.dump(template_data, f)
    
    # If there's a JSON version, save that too
    if not templates_path.endswith('.json'):
        json_path = f"{templates_path}.json"
        with open(json_path, 'w') as f:
            json.dump(template_data, f, indent=2, cls=CustomJSONEncoder)
    
    logger.info(f"Saved updated templates to {templates_path}")

def make_templates_readonly():
    """Make the template files read-only to prevent accidental modifications."""
    templates_path = "data/templates.pkl"
    json_path = f"{templates_path}.json"
    
    try:
        # Make template files read-only
        os.chmod(templates_path, 0o444)  # r--r--r--
        logger.info(f"Made {templates_path} read-only")
        
        if os.path.exists(json_path):
            os.chmod(json_path, 0o444)  # r--r--r--
            logger.info(f"Made {json_path} read-only")
    except Exception as e:
        logger.error(f"Error making files read-only: {e}")
    
def main():
    """Main function."""
    templates_path = "data/templates.pkl"
    
    # Check if templates file exists
    if not os.path.exists(templates_path):
        logger.error(f"Templates file not found: {templates_path}")
        return
    
    # Make sure the templates files are writable
    try:
        if os.path.exists(templates_path):
            os.chmod(templates_path, 0o644)  # rw-r--r--
        
        json_path = f"{templates_path}.json"
        if os.path.exists(json_path):
            os.chmod(json_path, 0o644)  # rw-r--r--
    except Exception as e:
        logger.error(f"Error making files writable: {e}")
    
    # Update templates
    update_templates(templates_path)
    
    # Verify the update
    updated_count = verify_update()
    
    if updated_count > 0:
        logger.info(f"Successfully verified {updated_count} updated templates")
        # Make the templates read-only to prevent accidental modifications
        make_templates_readonly()
    else:
        logger.error("Update verification failed. No templates were updated correctly.")

def verify_update():
    """Verify that templates have been properly updated."""
    templates_path = "data/templates.pkl"
    
    # Load templates
    with open(templates_path, 'rb') as f:
        template_data = pickle.load(f)
    
    if not template_data or 'templates' not in template_data:
        logger.error("Failed to load templates for verification")
        return 0
    
    templates = template_data['templates']
    updated_count = 0
    
    # Check if templates have been updated
    for template in templates:
        sql_template = template.get('sql_template', '')
        
        # Check if template uses the new format
        if 'start_date > ' in sql_template and 'end_date < ' in sql_template:
            updated_count += 1
        elif ' period = ' in sql_template:
            logger.warning(f"Template still uses old format: {template.get('template_query', 'Unknown template')}")
    
    return updated_count

if __name__ == "__main__":
    main()