"""
Query Cache Service

This module provides the main functionality for the query caching service,
integrating entity extraction and template matching to provide fast SQL
generation for common query patterns.
"""

import os
import json
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

from ..utils.entity_extractor import EntityExtractor
from ..utils.template_matcher import TemplateMatcher
from ..utils.sql_validator import SQLValidator
from .template_library import TemplateLibrary
from .template_learning import TemplateLearning

# Multilingual support imports
try:
    from ..utils.alias_mapper import AliasMapper
    from ..utils.multilingual_entity_extractor import MultilingualEntityExtractor
    MULTILINGUAL_AVAILABLE = True
except ImportError:
    MULTILINGUAL_AVAILABLE = False
    logger.warning("Multilingual components not available")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QueryCache:
    """
    Main class for the query caching service.
    """
    
    def __init__(self, config_path: Optional[str] = None, use_predefined_templates: bool = True, use_wtl_templates: bool = False):
        """
        Initialize the query cache service.
        
        Args:
            config_path: Path to configuration file
            use_predefined_templates: Whether to use predefined templates
            use_wtl_templates: Whether to use WTL-specific templates
        """
        # Default configuration with method-specific thresholds
        self.config = {
            'templates_path': 'data/templates.pkl',
            'entity_dictionary_path': 'data/entity_dictionary.json',
            'similarity_threshold': 0.65,  # Default fallback
            'similarity_thresholds': {
                'sentence_transformer': 0.75,
                'tfidf': 0.65,
                'keyword': 0.55
            },
            'max_templates': 1000,
            'model_name': 'all-MiniLM-L6-v2',
            'use_predefined_templates': use_predefined_templates,
            'use_wtl_templates': use_wtl_templates
        }
        
        # Load configuration if provided
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
                self.config.update(loaded_config)
        
        # Initialize entity dictionary
        self.entity_dictionary = {}
        if 'entity_dictionary_path' in self.config and os.path.exists(self.config['entity_dictionary_path']):
            with open(self.config['entity_dictionary_path'], 'r', encoding='utf-8') as f:
                self.entity_dictionary = json.load(f)
        
        # Initialize multilingual components if enabled
        self.alias_mapper = None
        if MULTILINGUAL_AVAILABLE and self.config.get('multilingual_enabled', False):
            try:
                # Initialize alias mapper
                alias_path = self.config.get('alias_mapping_path', 'data/alias_mappings.json')
                self.alias_mapper = AliasMapper(alias_path)
                
                # Use multilingual entity extractor
                self.entity_extractor = MultilingualEntityExtractor(
                    self.entity_dictionary, 
                    self.alias_mapper
                )
                logger.info("Multilingual entity extraction enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize multilingual components: {e}")
                self.entity_extractor = EntityExtractor(self.entity_dictionary)
        else:
            # Use standard entity extractor
            self.entity_extractor = EntityExtractor(self.entity_dictionary)
        self.sql_validator = SQLValidator()
        self.template_learning = TemplateLearning(self)
        
        templates_path = self.config.get('templates_path')
        if templates_path and os.path.exists(templates_path):
            self.template_matcher = TemplateMatcher(
                model_name=self.config.get('model_name'),
                templates_path=templates_path,
                similarity_thresholds=self.config.get('similarity_thresholds', {})
            )
        else:
            self.template_matcher = TemplateMatcher(
                model_name=self.config.get('model_name'),
                similarity_thresholds=self.config.get('similarity_thresholds', {})
            )
            
        # Initialize template library with predefined templates if enabled
        if self.config.get('use_predefined_templates', True):
            # Include WTL templates if specified
            self.template_library = TemplateLibrary(include_wtl_templates=self.config.get('use_wtl_templates', False))
            self._load_predefined_templates()
        
        # Performance metrics
        self.metrics = {
            'total_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'avg_response_time': 0
        }
        
        logger.info("Query Cache Service initialized")
        
    def _load_predefined_templates(self):
        """Load predefined templates into the template matcher."""
        predefined_templates = self.template_library.get_templates()
        logger.info(f"Loading {len(predefined_templates)} predefined templates")
        
        for template in predefined_templates:
            self.template_matcher.add_template(
                template_query=template['template_query'],
                sql_template=template['sql_template'],
                entity_map=template['entity_map'],
                metadata=template.get('metadata', {'source': 'predefined'})
            )
            
        logger.info(f"Predefined templates loaded successfully")
    
    def _current_timestamp(self):
        """Get current timestamp."""
        return time.time()
    
    def process_query(self, query: str, chess_interface=None) -> Dict[str, Any]:
        """
        Process a natural language query, attempting to use cached templates.
        
        Args:
            query: Natural language query
            chess_interface: Optional CHESS interface for fallback
            
        Returns:
            Dictionary containing the processing results
        """
        start_time = time.time()
        
        try:
            # Step 1: Extract entities from the query
            template_query, entity_map = self.entity_extractor.extract_and_normalize(query)
            logger.info(f"Extracted template: {template_query}")
            
            # Count number of entities by type
            entity_types = {}
            for placeholder, info in entity_map.items():
                entity_type = info['type']
                if entity_type not in entity_types:
                    entity_types[entity_type] = 0
                entity_types[entity_type] += 1
            
            # Step 2: Find matching template
            matching_template = self.template_matcher.find_matching_template(
                template_query, 
                similarity_threshold=self.config.get('similarity_threshold')
            )
            
            if matching_template:
                # Cache hit - use the template
                logger.info(f"Cache hit: {matching_template['template_query']}")
                
                # Replace placeholders in SQL template with new entities
                sql_template = matching_template['sql_template']
                sql_query = sql_template
                
                # Create a mapping from entity types to entity values
                type_to_entity = {}
                for placeholder, entity_info in entity_map.items():
                    entity_type = entity_info['type']
                    if entity_type not in type_to_entity:
                        type_to_entity[entity_type] = []
                    type_to_entity[entity_type].append(entity_info)
                
                # For each placeholder in the template's entity map, find a suitable entity
                for template_placeholder, template_entity in matching_template['entity_map'].items():
                    entity_type = template_entity['type']
                    
                    if entity_type in type_to_entity and type_to_entity[entity_type]:
                        # Take the first available entity of this type
                        entity_info = type_to_entity[entity_type][0]
                        # Remove it from available entities
                        type_to_entity[entity_type].pop(0)
                        # Replace in the SQL query
                        sql_query = sql_query.replace(template_placeholder, entity_info['normalized'])
                
                result = {
                    'success': True,
                    'source': 'cache',
                    'sql_query': sql_query,
                    'template_id': matching_template.get('id', 0),
                    'similarity_score': matching_template.get('similarity_score', 0.0),
                    'query_time_ms': int((time.time() - start_time) * 1000),
                    'template_query': template_query,
                    'entity_map': entity_map,
                    'matching_template': matching_template['template_query']
                }
                
                # Update template usage statistics
                try:
                    # Since we need to find the exact object in the list (not just equal values),
                    # we need to search by template_query which is the unique identifier
                    template_idx = None
                    for idx, template in enumerate(self.template_matcher.templates):
                        if template['template_query'] == matching_template['template_query']:
                            template_idx = idx
                            break
                            
                    if template_idx is not None:
                        self.template_matcher.update_template_stats(template_idx, success=True)
                    else:
                        logger.warning(f"Could not find matching template in templates list for stats update")
                except Exception as e:
                    logger.error(f"Error updating template stats: {e}")
                
                # Update metrics
                self.metrics['total_requests'] += 1
                self.metrics['cache_hits'] += 1
                
                # Update average response time
                elapsed_time = time.time() - start_time
                self.metrics['avg_response_time'] = (
                    (self.metrics['avg_response_time'] * (self.metrics['total_requests'] - 1) + elapsed_time) / 
                    self.metrics['total_requests']
                )
                
            else:
                # Cache miss
                logger.info(f"Cache miss for query: {query}")
                
                # Try CHESS fallback if available
                if chess_interface:
                    logger.info(f"Trying CHESS fallback")
                    try:
                        session_id = chess_interface.start_chat_session("default")
                        chess_result = chess_interface.chat_query(session_id, query)
                        
                        if chess_result.get('status') == 'success' and chess_result.get('sql_query'):
                            # Create a template from the CHESS result
                            chess_sql = chess_result['sql_query']
                            
                            # Add the template to our cache
                            self.add_template(
                                template_query=template_query,
                                sql_query=chess_sql,
                                entity_map=entity_map
                            )
                            
                            result = {
                                'success': True,
                                'source': 'chess',
                                'sql_query': chess_sql,
                                'query_time_ms': int((time.time() - start_time) * 1000),
                                'template_query': template_query,
                                'entity_map': entity_map,
                                'chess_response': chess_result.get('natural_language_response', '')
                            }
                            
                            # Update metrics
                            self.metrics['total_requests'] += 1
                            self.metrics['cache_misses'] += 1
                            
                            return result
                    except Exception as chess_error:
                        logger.error(f"Error using CHESS fallback: {chess_error}")
                
                # If we get here, either CHESS wasn't available or it failed
                result = {
                    'success': False,
                    'source': 'cache',
                    'error': 'No matching template found',
                    'query_time_ms': int((time.time() - start_time) * 1000),
                    'template_query': template_query,
                    'entity_map': entity_map
                }
                
                # Update metrics
                self.metrics['total_requests'] += 1
                self.metrics['cache_misses'] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            
            # Update metrics for error case
            self.metrics['total_requests'] += 1
            self.metrics['cache_misses'] += 1
            
            return {
                'success': False,
                'source': 'cache',
                'error': str(e),
                'query_time_ms': int((time.time() - start_time) * 1000)
            }
    
    def process_query_with_context(self, 
                                 query: str, 
                                 session_context: Dict[str, Any], 
                                 chat_history: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a query with session context for better matching.
        
        Args:
            query: Natural language query
            session_context: Session context information
            chat_history: Recent chat history
            
        Returns:
            Dictionary containing the processing results
        """
        start_time = time.time()
        
        try:
            # Enhanced entity extraction using session context
            template_query, entity_map = self.entity_extractor.extract_with_context(
                query, session_context, chat_history or []
            )
            logger.info(f"Context-aware extracted template: {template_query}")
            
            # Enhanced template matching considering conversation context
            matching_template = self._find_contextual_match(
                template_query, 
                session_context,
                entity_map
            )
            
            if matching_template:
                # Cache hit with context
                logger.info(f"Contextual cache hit: {matching_template['template_query']}")
                
                # Apply context-aware entity resolution
                sql_query = self._resolve_entities_with_context(
                    matching_template, entity_map, session_context
                )
                
                result = {
                    'success': True,
                    'source': 'cache',
                    'sql_query': sql_query,
                    'results': matching_template.get('sample_results', []),
                    'natural_language_response': matching_template.get('sample_response', 'Retrieved from cache'),
                    'template_id': matching_template.get('id', 0),
                    'similarity_score': matching_template.get('similarity_score', 0.0),
                    'query_time_ms': int((time.time() - start_time) * 1000),
                    'template_query': template_query,
                    'entity_map': entity_map,
                    'matching_template': matching_template['template_query'],
                    'context_preserved': True,
                    'visualization_metadata': matching_template.get('visualization_metadata', {})
                }
                
                # Update metrics
                self.metrics['total_requests'] += 1
                self.metrics['cache_hits'] += 1
                
                return result
            else:
                # Context-aware cache miss
                logger.info(f"Contextual cache miss for query: {query}")
                
                result = {
                    'success': False,
                    'source': 'cache',
                    'error': 'No matching template found with context',
                    'query_time_ms': int((time.time() - start_time) * 1000),
                    'template_query': template_query,
                    'entity_map': entity_map,
                    'session_context': session_context
                }
                
                # Update metrics
                self.metrics['total_requests'] += 1
                self.metrics['cache_misses'] += 1
                
                return result
                
        except Exception as e:
            logger.error(f"Error processing query with context: {e}", exc_info=True)
            
            # Update metrics for error case
            self.metrics['total_requests'] += 1
            self.metrics['cache_misses'] += 1
            
            return {
                'success': False,
                'source': 'cache',
                'error': str(e),
                'query_time_ms': int((time.time() - start_time) * 1000)
            }
    
    def store_with_context(self, 
                          query: str, 
                          sql_query: str, 
                          results: List[Dict[str, Any]], 
                          natural_language_response: str,
                          session_context: Dict[str, Any],
                          context_updates: Dict[str, Any] = None,
                          visualization_metadata: Dict[str, Any] = None,
                          execution_history: List[Dict[str, Any]] = None) -> bool:
        """
        Store a query result with session context.
        
        Args:
            query: Original natural language query
            sql_query: Generated SQL query
            results: Query execution results
            natural_language_response: Generated response
            session_context: Session context at time of query
            context_updates: Context updates from the query
            visualization_metadata: Visualization metadata
            execution_history: Execution history
            
        Returns:
            True if stored successfully, False otherwise
        """
        try:
            # Extract entities with context
            template_query, entity_map = self.entity_extractor.extract_with_context(
                query, session_context, []
            )
            
            # Check if template already exists
            for template in self.template_matcher.templates:
                if template['template_query'] == template_query:
                    # Update existing template with new data
                    logger.info(f"Updating existing template: {template_query}")
                    template['sample_results'] = results[:5]  # Store sample results
                    template['sample_response'] = natural_language_response
                    template['visualization_metadata'] = visualization_metadata or {}
                    template['usage_count'] = template.get('usage_count', 0) + 1
                    template['last_used'] = self._current_timestamp()
                    return True
            
            # Create SQL template with placeholders
            sql_template = sql_query
            for placeholder, entity_info in entity_map.items():
                normalized_value = entity_info.get('normalized', entity_info.get('value'))
                if normalized_value and normalized_value in sql_template:
                    sql_template = sql_template.replace(normalized_value, placeholder)
            
            # Add new template to the matcher
            self.template_matcher.add_template(
                template_query=template_query,
                sql_template=sql_template,
                entity_map=entity_map,
                metadata={
                    'source': 'chess_integration',
                    'timestamp': self._current_timestamp(),
                    'session_context': session_context,
                    'context_updates': context_updates or {},
                    'sample_results': results[:5],  # Store sample results
                    'sample_response': natural_language_response,
                    'visualization_metadata': visualization_metadata or {},
                    'execution_history': execution_history or []
                }
            )
            
            # Save templates if configured
            if 'templates_path' in self.config:
                self.save_state()
                
            logger.info(f"Stored new template with context: {template_query}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing result with context: {e}", exc_info=True)
            return False
    
    def _find_contextual_match(self, 
                              template_query: str, 
                              session_context: Dict[str, Any],
                              entity_map: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Find matching template considering conversation context.
        
        Args:
            template_query: Template query to match
            session_context: Current session context
            entity_map: Extracted entities
            
        Returns:
            Best matching template or None
        """
        best_match = None
        best_score = 0
        
        # Get recent entities from session context
        recent_entities = session_context.get('recent_entities', {})
        referenced_tables = set(session_context.get('referenced_tables', []))
        
        for template in self.template_matcher.templates:
            # Calculate base similarity
            base_similarity = self.template_matcher.calculate_similarity(
                template_query, template['template_query']
            )
            
            if base_similarity < self.config.get('similarity_threshold', 0.75):
                continue
            
            # Apply context boost
            context_boost = 0
            
            # Boost if template uses tables referenced in session
            template_metadata = template.get('metadata', {})
            template_context_updates = template_metadata.get('context_updates', {})
            template_tables = set(template_context_updates.get('tables_used', []))
            
            if template_tables & referenced_tables:
                context_boost += 0.1
                logger.debug(f"Table context boost for template: {template['template_query']}")
            
            # Boost if template entities match recent conversation entities
            template_entity_types = set()
            for entity_info in template['entity_map'].values():
                template_entity_types.add(entity_info['type'])
            
            current_entity_types = set()
            for entity_info in entity_map.values():
                current_entity_types.add(entity_info['type'])
            
            entity_overlap = len(template_entity_types & current_entity_types)
            if entity_overlap > 0:
                context_boost += entity_overlap * 0.05
                logger.debug(f"Entity type boost for template: {template['template_query']}")
            
            final_score = base_similarity + context_boost
            
            if final_score > best_score:
                best_score = final_score
                best_match = template.copy()
                best_match['similarity_score'] = final_score
        
        if best_match:
            logger.info(f"Best contextual match: {best_match['template_query']} (score: {best_score:.3f})")
        
        return best_match
    
    def _resolve_entities_with_context(self, 
                                     template: Dict[str, Any], 
                                     entity_map: Dict[str, Any],
                                     session_context: Dict[str, Any]) -> str:
        """
        Resolve entities in template considering session context.
        
        Args:
            template: Matching template
            entity_map: Current entities
            session_context: Session context
            
        Returns:
            SQL query with resolved entities
        """
        # Import date handler for time period resolution
        from ..utils.date_handler import get_date_range
        
        sql_template = template['sql_template']
        sql_query = sql_template
        
        # Create mapping from entity types to values
        type_to_entities = {}
        for placeholder, entity_info in entity_map.items():
            entity_type = entity_info['type']
            if entity_type not in type_to_entities:
                type_to_entities[entity_type] = []
            type_to_entities[entity_type].append((placeholder, entity_info))
        
        # Replace placeholders with actual values
        for template_placeholder, template_entity in template['entity_map'].items():
            entity_type = template_entity['type']
            
            if entity_type in type_to_entities and type_to_entities[entity_type]:
                # Use the first available entity of this type
                placeholder, entity_info = type_to_entities[entity_type][0]
                type_to_entities[entity_type].pop(0)
                
                # Special handling for time_period entities
                if entity_type == 'time_period':
                    # Get the time period value
                    time_period_value = entity_info.get('normalized', entity_info.get('value', ''))
                    
                    # Check if SQL template uses _start/_end or -start/-end suffixes
                    if (f"{template_placeholder}_start" in sql_query or f"{template_placeholder}_end" in sql_query or
                        f"{template_placeholder}-start" in sql_query or f"{template_placeholder}-end" in sql_query):
                        # Resolve the time period to actual dates
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
                            
                            logger.debug(f"Resolved time period {time_period_value} to dates: {start_date} - {end_date}")
                        except Exception as e:
                            logger.warning(f"Failed to resolve time period {time_period_value}: {e}")
                            # Fall back to simple replacement
                            sql_query = sql_query.replace(template_placeholder, time_period_value)
                    else:
                        # Simple replacement for time periods not using _start/_end pattern
                        sql_query = sql_query.replace(template_placeholder, time_period_value)
                else:
                    # Use normalized value for replacement
                    replacement_value = entity_info.get('normalized', entity_info.get('value', ''))
                    sql_query = sql_query.replace(template_placeholder, replacement_value)
                
                logger.debug(f"Replaced {template_placeholder} with resolved value")
        
        return sql_query
    
    def add_template(self, template_query: str, sql_query: str, entity_map: Dict[str, Dict[str, str]], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Add a new template to the cache with SQL validation.
        
        Args:
            template_query: Template query with entity placeholders
            sql_query: SQL query with entity values
            entity_map: Dictionary mapping placeholders to entity information
            context: Optional context information
            
        Returns:
            Dictionary containing success status and any issues
        """
        try:
            # Check if an identical template already exists
            for template in self.template_matcher.templates:
                if template['template_query'] == template_query:
                    # Template already exists
                    logger.info(f"Template already exists: {template_query}")
                    return {'success': True, 'message': 'Template already exists'}
                    
            # Create SQL template by replacing entity values with placeholders
            sql_template = sql_query
            for placeholder, entity_info in entity_map.items():
                normalized_value = entity_info.get('normalized', entity_info.get('value'))
                if normalized_value:
                    sql_template = sql_template.replace(normalized_value, placeholder)
            
            # Validate SQL before adding
            is_valid, issues = self.sql_validator.validate_sql_template(sql_template, entity_map)
            
            if not is_valid:
                logger.warning(f"SQL validation failed for template: {issues}")
                return {
                    'success': False,
                    'error': 'SQL validation failed',
                    'issues': issues
                }
            
            # Extract metadata from SQL
            tables_used = self.sql_validator.extract_table_references(sql_template)
            
            # Create metadata
            metadata = {
                'source': 'external',
                'timestamp': self._current_timestamp(),
                'created_at': time.time(),
                'last_used': time.time(),
                'intent': self.template_matcher.classify_query_intent(template_query),
                'tables_used': tables_used,
                'context_updates': context or {}
            }
            
            # Add template to the matcher
            self.template_matcher.add_template(
                template_query=template_query,
                sql_template=sql_template,
                entity_map=entity_map,
                metadata=metadata
            )
            
            # Save templates if we have a path configured
            if 'templates_path' in self.config:
                self.save_state()
                
            return {'success': True, 'message': 'Template added successfully'}
        except Exception as e:
            logger.error(f"Error adding template: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}
    
    def record_template_feedback(self, template_id: int, success: bool, execution_time: float = None, error: str = None):
        """Record feedback for template usage to improve learning."""
        self.template_learning.record_feedback(template_id, success, execution_time, error)
    
    def get_learning_insights(self) -> Dict[str, Any]:
        """Get insights about template learning and performance."""
        return self.template_learning.get_learning_insights()
    
    def suggest_template_improvements(self, template_id: int) -> List[Dict[str, Any]]:
        """Get suggestions for improving a specific template."""
        return self.template_learning.suggest_template_improvements(template_id)
    
    def auto_tune_thresholds(self) -> Dict[str, float]:
        """Get automatically tuned similarity thresholds based on performance."""
        return self.template_learning.auto_tune_thresholds()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        hit_rate = 0
        if self.metrics['total_requests'] > 0:
            hit_rate = self.metrics['cache_hits'] / self.metrics['total_requests']
        
        return {
            'total_requests': self.metrics['total_requests'],
            'cache_hits': self.metrics['cache_hits'],
            'cache_misses': self.metrics['cache_misses'],
            'hit_rate': hit_rate,
            'avg_response_time_ms': int(self.metrics['avg_response_time'] * 1000),
            'template_count': len(self.template_matcher.templates)
        }
    
    def save_state(self) -> None:
        """Save the current state of the cache service."""
        if 'templates_path' in self.config:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.config['templates_path']), exist_ok=True)
            self.template_matcher.save_templates(self.config['templates_path'])
            logger.info(f"Saved templates to {self.config['templates_path']}")
        
        if 'entity_dictionary_path' in self.config:
            os.makedirs(os.path.dirname(self.config['entity_dictionary_path']), exist_ok=True)
            with open(self.config['entity_dictionary_path'], 'w', encoding='utf-8') as f:
                json.dump(self.entity_dictionary, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved entity dictionary to {self.config['entity_dictionary_path']}")
        
        # Save alias mappings if multilingual is enabled
        if self.alias_mapper and 'alias_mapping_path' in self.config:
            self.alias_mapper.save_aliases(self.config['alias_mapping_path'])
    
    def add_alias(self, canonical: str, alias: str, entity_type: str) -> bool:
        """
        Add a new alias mapping for multilingual support.
        
        Args:
            canonical: Canonical entity name
            alias: Alias to map to the canonical name
            entity_type: Type of the entity
            
        Returns:
            True if added successfully, False otherwise
        """
        if not self.alias_mapper:
            logger.warning("Alias mapper not available - multilingual support not enabled")
            return False
        
        try:
            self.alias_mapper.add_alias(canonical, alias, entity_type)
            return True
        except Exception as e:
            logger.error(f"Error adding alias: {e}")
            return False
    
    def get_alias_mappings(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Get all alias mappings.
        
        Returns:
            Dictionary of alias mappings by entity type
        """
        if not self.alias_mapper:
            return {}
        return self.alias_mapper.alias_mappings
    
    def normalize_entity(self, entity: str, entity_type: str) -> str:
        """
        Normalize an entity using the alias mapper.
        
        Args:
            entity: Entity to normalize
            entity_type: Type of the entity
            
        Returns:
            Normalized entity name
        """
        if not self.alias_mapper:
            return entity
        return self.alias_mapper.normalize_entity(entity, entity_type)
    
    def get_entity_variations(self, canonical: str, entity_type: str) -> List[str]:
        """
        Get all variations of a canonical entity.
        
        Args:
            canonical: Canonical entity name
            entity_type: Type of the entity
            
        Returns:
            List of all variations including the canonical form
        """
        if not self.alias_mapper:
            return [canonical]
        return self.alias_mapper.get_all_variations(canonical, entity_type)
    
    def get_multilingual_stats(self) -> Dict[str, Any]:
        """
        Get statistics about multilingual support.
        
        Returns:
            Dictionary containing multilingual statistics
        """
        stats = {
            'multilingual_enabled': bool(self.alias_mapper),
            'extractor_type': type(self.entity_extractor).__name__,
        }
        
        if self.alias_mapper:
            stats.update(self.alias_mapper.get_stats())
        
        return stats


# Example usage
if __name__ == "__main__":
    # Create a query cache service
    cache_service = QueryCache()
    
    # Test with a query
    query = "How many hours did Bob Johnson work on the Mobile App project in Q3 2023?"
    result = cache_service.process_query(query)
    print(f"Query: {query}")
    print(f"Result: {result}")
    
    # Test adding a template
    template_query = "How many hours did {employee_0} work on the {project_0} project in {time_period_0}?"
    sql_query = "SELECT SUM(hours) FROM work_hours WHERE employee = 'Bob Johnson' AND project = 'Mobile App' AND period = 'Q3 2023'"
    entity_map = {
        '{employee_0}': {'type': 'employee', 'value': 'Bob Johnson', 'normalized': 'Bob Johnson'},
        '{project_0}': {'type': 'project', 'value': 'Mobile App', 'normalized': 'Mobile App'},
        '{time_period_0}': {'type': 'time_period', 'value': 'Q3 2023', 'normalized': 'Q3 2023'}
    }
    cache_service.add_template(template_query, sql_query, entity_map)
    
    # Test again with a similar query
    query2 = "How many hours did Jane Doe work on the Website Redesign project in Q4 2023?"
    result2 = cache_service.process_query(query2)
    print(f"\nQuery: {query2}")
    print(f"Result: {result2}")
    print(f"Metrics: {cache_service.get_metrics()}")