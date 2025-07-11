"""
Query Cache API Server

This module provides a Flask-based API server for the query caching service.
"""

import os
import sys
import json
import logging
import time
from typing import Dict, Any
from pathlib import Path
from flask import Flask, request, jsonify

# Add the parent directory to Python path
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent.parent
sys.path.append(str(parent_dir))

from src.core.query_cache import QueryCache

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Load configuration
config_path = os.environ.get('QUERY_CACHE_CONFIG', str(parent_dir / 'config.json'))
if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
else:
    config = {}

# Initialize the query cache service
query_cache = QueryCache(config_path=config_path if os.path.exists(config_path) else None)

# Make cache service available to the app
app.cache_service = query_cache
app.config['cache_service'] = query_cache

# Import and register alias management routes
try:
    from .alias_routes import alias_bp
    app.register_blueprint(alias_bp)
    logger.info("Alias management API routes registered")
except ImportError as e:
    logger.warning(f"Could not load alias routes: {e}")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'metrics': query_cache.get_metrics()
    }), 200


@app.route('/query', methods=['POST'])
def process_query():
    """Process a natural language query."""
    start_time = time.time()
    data = request.json
    
    if not data or 'query' not in data:
        return jsonify({
            'success': False,
            'error': 'Missing query parameter'
        }), 400
    
    query = data['query']
    
    try:
        # Extract entities and find matching template
        template_query, entity_map = query_cache.entity_extractor.extract_and_normalize(query)
        matching_template = query_cache.template_matcher.find_matching_template(
            template_query, 
            similarity_threshold=query_cache.config.get('similarity_threshold')
        )
        
        if matching_template:
            # Cache hit - use the template
            sql_template = matching_template['sql_template']
            sql_query = sql_template
            
            # Import date handler for time period resolution
            from ..utils.date_handler import get_date_range
            
            # Replace placeholders in SQL template with entities
            for template_placeholder, template_entity in matching_template['entity_map'].items():
                entity_type = template_entity['type']
                for placeholder, entity_info in entity_map.items():
                    if entity_info['type'] == entity_type:
                        if entity_type == 'time_period':
                            # Special handling for time periods
                            time_period_value = entity_info['normalized']
                            
                            # Check if SQL template uses _start/_end or -start/-end suffixes
                            if (f"{template_placeholder}_start" in sql_query or f"{template_placeholder}_end" in sql_query or
                                f"{template_placeholder}-start" in sql_query or f"{template_placeholder}-end" in sql_query):
                                try:
                                    start_date, end_date = get_date_range(time_period_value)
                                    # Replace all variations of start/end placeholders
                                    sql_query = sql_query.replace(f"'{template_placeholder}_start'", f"'{start_date}'")
                                    sql_query = sql_query.replace(f"'{template_placeholder}_end'", f"'{end_date}'")
                                    sql_query = sql_query.replace(f"'{template_placeholder}-start'", f"'{start_date}'")
                                    sql_query = sql_query.replace(f"'{template_placeholder}-end'", f"'{end_date}'")
                                    # Also handle without quotes
                                    sql_query = sql_query.replace(f"{template_placeholder}_start", f"'{start_date}'")
                                    sql_query = sql_query.replace(f"{template_placeholder}_end", f"'{end_date}'")
                                    sql_query = sql_query.replace(f"{template_placeholder}-start", f"'{start_date}'")
                                    sql_query = sql_query.replace(f"{template_placeholder}-end", f"'{end_date}'")
                                except Exception as e:
                                    logger.warning(f"Failed to resolve time period {time_period_value}: {e}")
                                    sql_query = sql_query.replace(template_placeholder, time_period_value)
                            else:
                                sql_query = sql_query.replace(template_placeholder, time_period_value)
                        else:
                            sql_query = sql_query.replace(template_placeholder, entity_info['normalized'])
            
            # Update metrics and template stats
            query_cache.metrics['total_requests'] += 1
            query_cache.metrics['cache_hits'] += 1
            
            # Find template index by template_query instead of object reference
            template_idx = None
            for idx, template in enumerate(query_cache.template_matcher.templates):
                if template['template_query'] == matching_template['template_query']:
                    template_idx = idx
                    break
            
            if template_idx is not None:
                query_cache.template_matcher.update_template_stats(template_idx, success=True)
            else:
                logger.warning(f"Could not find matching template in templates list for stats update")
            
            return jsonify({
                'success': True,
                'source': 'cache',
                'sql_query': sql_query,
                'template_id': matching_template.get('id', 0),
                'similarity_score': matching_template.get('similarity_score', 0.0),
                'template_query': template_query,
                'entity_map': entity_map,
                'query_time_ms': int((time.time() - start_time) * 1000)
            }), 200
        else:
            # Cache miss - return template query and entity map for later caching
            query_cache.metrics['total_requests'] += 1
            query_cache.metrics['cache_misses'] += 1
            
            return jsonify({
                'success': False,
                'source': 'cache',
                'template_query': template_query,
                'entity_map': entity_map,
                'error': 'No matching template found',
                'query_time_ms': int((time.time() - start_time) * 1000)
            }), 404
    
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/query_with_context', methods=['POST'])
def process_query_with_context():
    """Process a natural language query with session context."""
    start_time = time.time()
    data = request.json
    
    if not data or 'query' not in data:
        return jsonify({
            'success': False,
            'error': 'Missing query parameter'
        }), 400
    
    query = data['query']
    session_context = data.get('session_context', {})
    chat_history = data.get('chat_history', [])
    
    try:
        # Use the enhanced context-aware processing
        result = query_cache.process_query_with_context(
            query=query,
            session_context=session_context,
            chat_history=chat_history
        )
        
        return jsonify(result), 200 if result.get('success') else 404
    
    except Exception as e:
        logger.error(f"Error processing query with context: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/store_with_context', methods=['POST'])
def store_with_context():
    """Store a query result with session context."""
    data = request.json
    
    required_fields = ['query', 'sql_query', 'results', 'session_context']
    if not data or not all(field in data for field in required_fields):
        return jsonify({
            'success': False,
            'error': f'Missing required parameters: {required_fields}'
        }), 400
    
    try:
        # Store the result with context
        success = query_cache.store_with_context(
            query=data['query'],
            sql_query=data['sql_query'],
            results=data['results'],
            natural_language_response=data.get('natural_language_response', ''),
            session_context=data['session_context'],
            context_updates=data.get('context_updates', {}),
            visualization_metadata=data.get('visualization_metadata', {}),
            execution_history=data.get('execution_history', [])
        )
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Result stored successfully with context'
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to store result'
            }), 500
        
    except Exception as e:
        logger.error(f"Error storing result with context: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/add', methods=['POST'])
def add_template():
    """Add a new template to the cache."""
    data = request.json
    
    if not data or 'template_query' not in data or 'sql_query' not in data or 'entity_map' not in data:
        return jsonify({
            'success': False,
            'error': 'Missing required parameters: template_query, sql_query, entity_map'
        }), 400
    
    try:
        # Extract values from request
        template_query = data['template_query']
        sql_query = data['sql_query']
        entity_map = data['entity_map']
        
        # Create SQL template with placeholders
        sql_template = sql_query
        for placeholder, entity_info in entity_map.items():
            normalized_value = entity_info.get('normalized', entity_info.get('value'))
            if normalized_value:
                sql_template = sql_template.replace(normalized_value, placeholder)
        
        # Add the template
        query_cache.template_matcher.add_template(
            template_query=template_query,
            sql_template=sql_template,
            entity_map=entity_map,
            metadata={
                'source': data.get('source', 'external'),
                'timestamp': data.get('timestamp', time.time())
            }
        )
        
        # Save templates if configured
        if query_cache.config.get('templates_path'):
            query_cache.save_state()
        
        return jsonify({
            'success': True,
            'message': 'Template added successfully'
        }), 200
        
    except Exception as e:
        logger.error(f"Error adding template: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/templates', methods=['GET'])
def list_templates():
    """List all templates in the cache."""
    try:
        templates = query_cache.template_matcher.templates
        return jsonify({
            'success': True,
            'count': len(templates),
            'templates': templates
        }), 200
    except Exception as e:
        logger.error(f"Error listing templates: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/metrics', methods=['GET'])
def get_metrics():
    """Get performance metrics for the cache service."""
    try:
        metrics = query_cache.get_metrics()
        return jsonify({
            'success': True,
            'metrics': metrics
        }), 200
    except Exception as e:
        logger.error(f"Error getting metrics: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/save', methods=['POST'])
def save_state():
    """Save the current state of the cache service."""
    try:
        query_cache.save_state()
        return jsonify({
            'success': True,
            'message': 'Cache state saved successfully'
        }), 200
    except Exception as e:
        logger.error(f"Error saving cache state: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/clear_invalid_cache', methods=['POST'])
def clear_invalid_cache():
    """Clear cache entries with unresolved date placeholders."""
    try:
        # Get all templates
        invalid_count = 0
        valid_templates = []
        
        for template in query_cache.template_matcher.templates:
            sql_template = template.get('sql_template', '')
            
            # Check if SQL contains unresolved placeholders
            has_unresolved = False
            unresolved_patterns = [
                r"'\{time_period_\d+\}_start'",
                r"'\{time_period_\d+\}_end'", 
                r"'\{time_period_\d+\}-start'",
                r"'\{time_period_\d+\}-end'",
                r"'Q[1-4]\s+\d{4}_start'",
                r"'Q[1-4]\s+\d{4}_end'",
                r"'Q[1-4]\s+\d{4}-start'",
                r"'Q[1-4]\s+\d{4}-end'",
                r"'\d{4}_start'",
                r"'\d{4}_end'",
                r"'\d{4}-start'",
                r"'\d{4}-end'"
            ]
            
            import re
            for pattern in unresolved_patterns:
                if re.search(pattern, sql_template):
                    has_unresolved = True
                    invalid_count += 1
                    logger.info(f"Found invalid template with unresolved placeholder: {sql_template[:100]}...")
                    break
            
            if not has_unresolved:
                valid_templates.append(template)
        
        # Replace templates list with only valid ones
        query_cache.template_matcher.templates = valid_templates
        
        # Save the cleaned templates if configured
        if hasattr(query_cache.template_matcher, 'templates_path') and query_cache.template_matcher.templates_path:
            query_cache.template_matcher.save_templates()
        
        return jsonify({
            'success': True,
            'message': f'Cleared {invalid_count} invalid cache entries',
            'remaining_templates': len(valid_templates)
        }), 200
        
    except Exception as e:
        logger.error(f"Error clearing invalid cache: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


def main():
    """Start the API server."""
    port = int(os.environ.get('QUERY_CACHE_PORT', 6000))
    debug = os.environ.get('QUERY_CACHE_DEBUG', 'false').lower() == 'true'
    
    logger.info(f"Starting Query Cache API server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)


if __name__ == '__main__':
    main()