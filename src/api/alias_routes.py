"""
Alias Management API Routes

This module provides REST API endpoints for managing multilingual entity aliases
in the query cache service.
"""

import logging
from flask import Blueprint, request, jsonify, current_app
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Create blueprint for alias management routes
alias_bp = Blueprint('alias', __name__, url_prefix='/api/alias')


def get_cache_service():
    """Get the cache service instance from the Flask app."""
    if hasattr(current_app, 'cache_service'):
        return current_app.cache_service
    elif 'cache_service' in current_app.config:
        return current_app.config['cache_service']
    else:
        raise RuntimeError("Cache service not found in application")


@alias_bp.route('/mappings', methods=['GET'])
def get_alias_mappings():
    """
    Get all alias mappings.
    
    Returns:
        JSON response with all alias mappings organized by entity type
    """
    try:
        cache_service = get_cache_service()
        mappings = cache_service.get_alias_mappings()
        
        return jsonify({
            'success': True,
            'data': mappings,
            'total_types': len(mappings),
            'total_entities': sum(len(entities) for entities in mappings.values())
        })
        
    except Exception as e:
        logger.error(f"Error getting alias mappings: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@alias_bp.route('/mappings/<entity_type>', methods=['GET'])
def get_entity_type_mappings(entity_type: str):
    """
    Get alias mappings for a specific entity type.
    
    Args:
        entity_type: Type of entities to retrieve
        
    Returns:
        JSON response with mappings for the specified entity type
    """
    try:
        cache_service = get_cache_service()
        mappings = cache_service.get_alias_mappings()
        
        if entity_type not in mappings:
            return jsonify({
                'success': False,
                'error': f'Entity type "{entity_type}" not found'
            }), 404
        
        return jsonify({
            'success': True,
            'entity_type': entity_type,
            'data': mappings[entity_type],
            'total_entities': len(mappings[entity_type])
        })
        
    except Exception as e:
        logger.error(f"Error getting mappings for entity type {entity_type}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@alias_bp.route('/add', methods=['POST'])
def add_alias():
    """
    Add a new alias mapping.
    
    Request body should contain:
    {
        "canonical": "canonical entity name",
        "alias": "alias to add",
        "entity_type": "type of entity"
    }
    
    Returns:
        JSON response indicating success or failure
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400
        
        canonical = data.get('canonical')
        alias = data.get('alias')
        entity_type = data.get('entity_type')
        
        if not all([canonical, alias, entity_type]):
            return jsonify({
                'success': False,
                'error': 'Missing required fields: canonical, alias, entity_type'
            }), 400
        
        cache_service = get_cache_service()
        success = cache_service.add_alias(canonical, alias, entity_type)
        
        if success:
            # Save the updated mappings
            cache_service.save_state()
            
            return jsonify({
                'success': True,
                'message': f'Added alias "{alias}" for "{canonical}" (type: {entity_type})',
                'canonical': canonical,
                'alias': alias,
                'entity_type': entity_type
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to add alias - multilingual support may not be enabled'
            }), 400
        
    except Exception as e:
        logger.error(f"Error adding alias: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@alias_bp.route('/normalize', methods=['POST'])
def normalize_entity():
    """
    Normalize an entity using alias mappings.
    
    Request body should contain:
    {
        "entity": "entity to normalize",
        "entity_type": "type of entity"
    }
    
    Returns:
        JSON response with normalized entity and all variations
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400
        
        entity = data.get('entity')
        entity_type = data.get('entity_type')
        
        if not all([entity, entity_type]):
            return jsonify({
                'success': False,
                'error': 'Missing required fields: entity, entity_type'
            }), 400
        
        cache_service = get_cache_service()
        normalized = cache_service.normalize_entity(entity, entity_type)
        variations = cache_service.get_entity_variations(normalized, entity_type)
        
        return jsonify({
            'success': True,
            'original': entity,
            'normalized': normalized,
            'entity_type': entity_type,
            'variations': variations,
            'was_normalized': normalized != entity
        })
        
    except Exception as e:
        logger.error(f"Error normalizing entity: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@alias_bp.route('/variations/<entity_type>/<canonical>', methods=['GET'])
def get_entity_variations(entity_type: str, canonical: str):
    """
    Get all variations of a canonical entity.
    
    Args:
        entity_type: Type of the entity
        canonical: Canonical entity name
        
    Returns:
        JSON response with all variations of the entity
    """
    try:
        cache_service = get_cache_service()
        variations = cache_service.get_entity_variations(canonical, entity_type)
        
        return jsonify({
            'success': True,
            'canonical': canonical,
            'entity_type': entity_type,
            'variations': variations,
            'total_variations': len(variations)
        })
        
    except Exception as e:
        logger.error(f"Error getting variations for {canonical}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@alias_bp.route('/search', methods=['POST'])
def search_entities():
    """
    Search for entities across all types or within a specific type.
    
    Request body should contain:
    {
        "query": "search term",
        "entity_type": "specific type (optional)",
        "fuzzy": true/false (optional, default: false)
    }
    
    Returns:
        JSON response with matching entities
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400
        
        query = data.get('query', '').strip()
        entity_type = data.get('entity_type')
        use_fuzzy = data.get('fuzzy', False)
        
        if not query:
            return jsonify({
                'success': False,
                'error': 'Query parameter is required'
            }), 400
        
        cache_service = get_cache_service()
        mappings = cache_service.get_alias_mappings()
        
        results = []
        query_lower = query.lower()
        
        # Search within specific entity type or all types
        search_types = [entity_type] if entity_type else mappings.keys()
        
        for search_type in search_types:
            if search_type not in mappings:
                continue
                
            for canonical, aliases in mappings[search_type].items():
                matches = []
                
                # Check canonical name
                if query_lower in canonical.lower():
                    matches.append({
                        'text': canonical,
                        'type': 'canonical',
                        'exact_match': query_lower == canonical.lower()
                    })
                
                # Check aliases
                for alias in aliases:
                    if query_lower in alias.lower():
                        matches.append({
                            'text': alias,
                            'type': 'alias',
                            'exact_match': query_lower == alias.lower()
                        })
                
                if matches:
                    results.append({
                        'entity_type': search_type,
                        'canonical': canonical,
                        'matches': matches,
                        'all_variations': [canonical] + aliases
                    })
        
        # Sort results by relevance (exact matches first, then by match count)
        results.sort(key=lambda x: (
            -sum(1 for m in x['matches'] if m['exact_match']),  # Exact matches first
            -len(x['matches'])  # Then by number of matches
        ))
        
        return jsonify({
            'success': True,
            'query': query,
            'entity_type': entity_type,
            'results': results,
            'total_matches': len(results)
        })
        
    except Exception as e:
        logger.error(f"Error searching entities: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@alias_bp.route('/stats', methods=['GET'])
def get_alias_stats():
    """
    Get statistics about the alias system.
    
    Returns:
        JSON response with alias system statistics
    """
    try:
        cache_service = get_cache_service()
        stats = cache_service.get_multilingual_stats()
        
        return jsonify({
            'success': True,
            'stats': stats
        })
        
    except Exception as e:
        logger.error(f"Error getting alias stats: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@alias_bp.route('/test', methods=['POST'])
def test_multilingual_extraction():
    """
    Test multilingual entity extraction with a sample query.
    
    Request body should contain:
    {
        "query": "test query in any language"
    }
    
    Returns:
        JSON response with extraction results
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400
        
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({
                'success': False,
                'error': 'Query parameter is required'
            }), 400
        
        cache_service = get_cache_service()
        
        # Extract entities using the multilingual extractor
        template_query, entity_map = cache_service.entity_extractor.extract_and_normalize(query)
        
        # Process entity map for JSON serialization
        processed_entities = {}
        for placeholder, info in entity_map.items():
            processed_entities[placeholder] = {
                'value': info['value'],
                'type': info['type'],
                'normalized': info['normalized'],
                'variations': info.get('variations', [])
            }
        
        return jsonify({
            'success': True,
            'original_query': query,
            'template_query': template_query,
            'entities': processed_entities,
            'entity_count': len(processed_entities),
            'extractor_type': type(cache_service.entity_extractor).__name__
        })
        
    except Exception as e:
        logger.error(f"Error testing multilingual extraction: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# Error handlers
@alias_bp.errorhandler(404)
def alias_not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404


@alias_bp.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        'success': False,
        'error': 'Method not allowed for this endpoint'
    }), 405


# Health check endpoint
@alias_bp.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint for alias management system.
    
    Returns:
        JSON response indicating system health
    """
    try:
        cache_service = get_cache_service()
        stats = cache_service.get_multilingual_stats()
        
        return jsonify({
            'success': True,
            'status': 'healthy',
            'multilingual_enabled': stats.get('multilingual_enabled', False),
            'extractor_type': stats.get('extractor_type', 'unknown')
        })
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'success': False,
            'status': 'unhealthy',
            'error': str(e)
        }), 500