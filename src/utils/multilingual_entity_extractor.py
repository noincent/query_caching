"""
Multilingual Entity Extraction Module

This module extends the base EntityExtractor to support multilingual entity
extraction with smart alias mapping and cross-language normalization.
"""

import re
import logging
from typing import Dict, List, Tuple, Optional, Set, Any
from .entity_extractor import EntityExtractor
from .alias_mapper import AliasMapper

logger = logging.getLogger(__name__)

# Initialize Chinese processing components
JIEBA_AVAILABLE = False
LANGDETECT_AVAILABLE = False

try:
    import jieba
    JIEBA_AVAILABLE = True
    logger.info("Jieba Chinese segmentation available")
except ImportError:
    logger.warning("Jieba not available - Chinese text processing will be limited")

try:
    from langdetect import detect, LangDetectException
    LANGDETECT_AVAILABLE = True
    logger.info("Language detection available")
except ImportError:
    logger.warning("Language detection not available")


class MultilingualEntityExtractor(EntityExtractor):
    """Enhanced entity extractor with multilingual support and alias mapping."""
    
    def __init__(self, entity_dictionary: Dict[str, List[str]], 
                 alias_mapper: Optional[AliasMapper] = None):
        """
        Initialize multilingual entity extractor.
        
        Args:
            entity_dictionary: Dictionary mapping entity types to known entities
            alias_mapper: Optional alias mapper for entity normalization
        """
        super().__init__(entity_dictionary)
        self.alias_mapper = alias_mapper or AliasMapper()
        
        # Enhanced Chinese patterns for entity recognition
        self.chinese_patterns = {
            'time_period': [
                r'第[一二三四]季度',
                r'[今上下本]个?[月季年]',
                r'\d{4}年第?[一二三四]季度',
                r'\d{4}年\d{1,2}月',
                r'[一二三四五六七八九十十一十二]+月份?',
                r'Q[1-4]\s*\d{4}',
                r'Q[1-4]',
            ],
            'department': [
                r'[\u4e00-\u9fa5]+(?:管理)?(?:部|中心|科|室|组|办)',
                r'[\u4e00-\u9fa5]+(?:设计|工程|营销|财务|行政)[\u4e00-\u9fa5]*',
            ],
            'employee': [
                r'[\u4e00-\u9fa5]{2,4}(?:经理|总监|主任|助理|工程师|设计师|顾问|专员|分析师)',
                r'[\u4e00-\u9fa5]{2,4}',  # Chinese names
            ],
            'work_type': [
                r'[\u4e00-\u9fa5]+(?:工作|任务|事项|配合|对接|审核|检查|管理|设计|施工)',
                r'(?:现场|材料|施工|设计|预算|客户)[\u4e00-\u9fa5]*',
            ],
            'project': [
                r'[\u4e00-\u9fa5]+(?:项目|工程|建设|改造|升级)',
                r'[\u4e00-\u9fa5]+(?:大厦|大楼|园区|广场|中心)',
            ]
        }
        
        # Initialize jieba with domain-specific terms
        if JIEBA_AVAILABLE:
            self._init_jieba()
        
    def _init_jieba(self):
        """Initialize jieba with domain-specific terms for better segmentation."""
        try:
            # Add known entities to jieba dictionary for better segmentation
            for entity_type, entities in self.entity_dictionary.items():
                for entity in entities:
                    if self._is_chinese(entity):
                        jieba.add_word(entity)
                        
            # Add common domain terms
            domain_terms = [
                "物业管理", "财务管理", "营销策略", "行政服务", "综合管理",
                "金尚设计", "品牌营销", "设计管理", "工程营造", "业务拓展",
                "售后服务", "预算成本", "施工现场", "材料对接", "客户沟通",
                "现场管理", "设计结算", "员工关系"
            ]
            
            for term in domain_terms:
                jieba.add_word(term)
                
            logger.info("Jieba dictionary initialized with domain-specific terms")
            
        except Exception as e:
            logger.warning(f"Error initializing jieba: {e}")
    
    def _is_chinese(self, text: str) -> bool:
        """Check if text contains Chinese characters."""
        return bool(re.search(r'[\u4e00-\u9fa5]', text))
    
    def _detect_language(self, query: str) -> str:
        """
        Detect the primary language of the query.
        
        Args:
            query: Text to analyze
            
        Returns:
            Language code ('zh' for Chinese, 'en' for English, etc.)
        """
        if not LANGDETECT_AVAILABLE:
            # Fallback: check for Chinese characters
            if self._is_chinese(query):
                return 'zh'
            return 'en'
        
        try:
            lang = detect(query)
            return lang
        except (LangDetectException, Exception):
            # Fallback: check for Chinese characters
            if self._is_chinese(query):
                return 'zh'
            return 'en'
    
    def extract_chinese_entities(self, query: str) -> Dict[str, List[str]]:
        """
        Extract entities from Chinese text using pattern matching and segmentation.
        
        Args:
            query: Chinese text query
            
        Returns:
            Dictionary mapping entity types to extracted entities
        """
        entities = {}
        
        # Use jieba for word segmentation if available
        if JIEBA_AVAILABLE:
            try:
                words = list(jieba.cut(query))
                logger.debug(f"Jieba segmentation: {' / '.join(words)}")
                
                # Match segmented words against known entities
                for word in words:
                    word_stripped = word.strip()
                    if len(word_stripped) < 2:  # Skip single characters and punctuation
                        continue
                        
                    for entity_type, known_entities in self.entity_dictionary.items():
                        for known_entity in known_entities:
                            if (word_stripped == known_entity or 
                                word_stripped in known_entity or 
                                known_entity in word_stripped):
                                if entity_type not in entities:
                                    entities[entity_type] = []
                                if known_entity not in entities[entity_type]:
                                    entities[entity_type].append(known_entity)
                                    
            except Exception as e:
                logger.warning(f"Error in jieba segmentation: {e}")
        
        # Extract based on Chinese patterns
        for entity_type, patterns in self.chinese_patterns.items():
            for pattern in patterns:
                try:
                    matches = re.findall(pattern, query)
                    for match in matches:
                        if isinstance(match, tuple):
                            # Handle tuple matches by taking the first non-empty group
                            match = next((m for m in match if m), match[0] if match else "")
                        
                        if match and len(match.strip()) >= 2:
                            if entity_type not in entities:
                                entities[entity_type] = []
                            if match not in entities[entity_type]:
                                entities[entity_type].append(match)
                                
                except Exception as e:
                    logger.warning(f"Error matching Chinese pattern {pattern}: {e}")
        
        # Direct entity dictionary matching
        for entity_type, known_entities in self.entity_dictionary.items():
            for entity in known_entities:
                if self._is_chinese(entity) and entity in query:
                    if entity_type not in entities:
                        entities[entity_type] = []
                    if entity not in entities[entity_type]:
                        entities[entity_type].append(entity)
        
        return entities
    
    def extract_multilingual_entities(self, query: str) -> Dict[str, List[str]]:
        """
        Extract entities from multilingual text (mixed Chinese/English).
        
        Args:
            query: Mixed language query
            
        Returns:
            Dictionary mapping entity types to extracted entities
        """
        # Start with base English extraction
        entities = super().extract_entities(query)
        
        # Add Chinese entity extraction
        chinese_entities = self.extract_chinese_entities(query)
        
        # Merge entities
        for entity_type, chinese_list in chinese_entities.items():
            if entity_type not in entities:
                entities[entity_type] = []
            
            for entity in chinese_list:
                if entity not in entities[entity_type]:
                    entities[entity_type].append(entity)
        
        # Enhanced pattern matching for mixed queries
        mixed_patterns = {
            'time_period': [
                r'Q[1-4]\s*\d{4}年?',
                r'\d{4}年?Q[1-4]',
                r'(January|February|March|April|May|June|July|August|September|October|November|December)\s*\d{4}年?'
            ],
            'employee': [
                r'[\u4e00-\u9fa5]{2,4}\s+(Manager|Engineer|Director|Supervisor)',
                r'(Manager|Engineer|Director|Supervisor)\s+[\u4e00-\u9fa5]{2,4}'
            ]
        }
        
        for entity_type, patterns in mixed_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, query, re.IGNORECASE)
                for match in matches:
                    if isinstance(match, tuple):
                        match = ' '.join(filter(None, match))
                    
                    if match and match.strip():
                        if entity_type not in entities:
                            entities[entity_type] = []
                        if match not in entities[entity_type]:
                            entities[entity_type].append(match)
        
        return entities
    
    def extract_entities(self, query: str) -> Dict[str, List[str]]:
        """
        Enhanced entity extraction with multilingual support.
        
        Args:
            query: Natural language query string
            
        Returns:
            Dictionary mapping entity types to extracted entities
        """
        # Detect primary language
        lang = self._detect_language(query)
        logger.debug(f"Detected language: {lang} for query: {query[:50]}...")
        
        # Extract entities based on detected language and content
        if lang == 'zh' or self._is_chinese(query):
            # Primarily Chinese or contains Chinese
            if self._is_chinese(query) and not re.search(r'[a-zA-Z]', query):
                # Pure Chinese
                entities = self.extract_chinese_entities(query)
            else:
                # Mixed language
                entities = self.extract_multilingual_entities(query)
        else:
            # Primarily English, but check for Chinese content
            if self._is_chinese(query):
                entities = self.extract_multilingual_entities(query)
            else:
                entities = super().extract_entities(query)
        
        # Validate and clean entities
        cleaned_entities = {}
        for entity_type, entity_list in entities.items():
            cleaned_list = []
            for entity in entity_list:
                if self.validate_entity(entity, entity_type):
                    cleaned_list.append(entity)
            
            if cleaned_list:
                cleaned_entities[entity_type] = cleaned_list
        
        return cleaned_entities
    
    def normalize_entity(self, entity: str, entity_type: str) -> str:
        """
        Enhanced entity normalization using alias mapping.
        
        Args:
            entity: Entity string to normalize
            entity_type: Type of the entity
            
        Returns:
            Normalized entity string
        """
        # First try alias mapping
        normalized = self.alias_mapper.normalize_entity(entity, entity_type)
        
        # If no alias mapping, fall back to base normalization
        if normalized == entity:
            normalized = super().normalize_entity(entity, entity_type)
        
        return normalized
    
    def extract_and_normalize(self, query: str) -> Tuple[str, Dict[str, Dict[str, str]]]:
        """
        Extract and normalize entities with multilingual support and alias mapping.
        
        Args:
            query: Natural language query string
            
        Returns:
            Tuple containing:
                - Query with entities replaced by placeholders
                - Dictionary mapping placeholders to entity information
        """
        # Extract entities using multilingual extraction
        entities = self.extract_entities(query)
        template_query = query
        entity_map = {}
        
        logger.debug(f"Extracted entities: {entities}")
        
        # Process entities by type priority to avoid overlapping replacements
        entity_type_priority = ['project', 'employee', 'department', 'work_type', 'time_period', 'date', 'project_id', 'email', 'percentage', 'number']
        
        for entity_type in entity_type_priority:
            if entity_type not in entities:
                continue
                
            # Sort entities by length (descending) to replace longer entities first
            entity_list = entities[entity_type]
            sorted_entities = sorted([str(e) for e in entity_list if e], key=len, reverse=True)
            
            for i, entity in enumerate(sorted_entities):
                if not entity or not entity.strip():
                    continue
                
                placeholder = f"{{{entity_type}_{i}}}"
                
                # Create regex pattern for whole word/phrase matching
                # Handle both Chinese and English boundaries
                if self._is_chinese(entity):
                    # For Chinese, use the entity as-is since Chinese doesn't use word boundaries
                    pattern = re.escape(entity)
                else:
                    # For English, use word boundaries
                    pattern = r'\b' + re.escape(entity) + r'\b'
                
                # Replace in template query
                template_query = re.sub(pattern, placeholder, template_query, flags=re.IGNORECASE)
                
                # Normalize entity using alias mapper
                normalized_entity = self.normalize_entity(entity, entity_type)
                
                # Get all variations for this entity
                variations = self.alias_mapper.get_all_variations(normalized_entity, entity_type)
                
                # Replace all variations in the template
                for variation in variations:
                    if variation != entity:  # Don't double-replace
                        if self._is_chinese(variation):
                            var_pattern = re.escape(variation)
                        else:
                            var_pattern = r'\b' + re.escape(variation) + r'\b'
                        template_query = re.sub(var_pattern, placeholder, template_query, flags=re.IGNORECASE)
                
                entity_map[placeholder] = {
                    'value': entity,
                    'type': entity_type,
                    'normalized': normalized_entity,
                    'variations': variations
                }
        
        logger.debug(f"Generated template: {template_query}")
        logger.debug(f"Entity map: {entity_map}")
        
        return template_query, entity_map
    
    def extract_with_context(self, 
                            text: str, 
                            session_context: Dict[str, Any] = None, 
                            chat_history: List[Dict[str, Any]] = None) -> Tuple[str, Dict[str, Dict[str, str]]]:
        """
        Extract entities with multilingual context awareness.
        
        Args:
            text: Input text to process
            session_context: Current session context
            chat_history: Recent chat history
            
        Returns:
            Tuple of (template_query, entity_map)
        """
        if session_context is None:
            session_context = {}
        if chat_history is None:
            chat_history = []
        
        # Use multilingual extraction as base
        template_query, entity_map = self.extract_and_normalize(text)
        
        # Enhance with context (language-aware)
        self._enhance_multilingual_context(entity_map, session_context, chat_history)
        
        return template_query, entity_map
    
    def _enhance_multilingual_context(self, 
                                    entity_map: Dict[str, Dict[str, str]], 
                                    session_context: Dict[str, Any], 
                                    chat_history: List[Dict[str, Any]]) -> None:
        """
        Enhance entity extraction using multilingual session context.
        
        Args:
            entity_map: Current entity map to enhance
            session_context: Session context information
            chat_history: Recent chat history
        """
        # Extract entities from recent multilingual conversation
        recent_entities = {}
        for msg in chat_history[-3:]:  # Last 3 messages
            msg_content = msg.get('content', '')
            if msg_content:
                _, msg_entities = self.extract_and_normalize(msg_content)
                for placeholder, entity_info in msg_entities.items():
                    entity_type = entity_info['type']
                    if entity_type not in recent_entities:
                        recent_entities[entity_type] = []
                    recent_entities[entity_type].append(entity_info)
        
        # Update session context with recent entities
        session_context['recent_entities'] = recent_entities
        
        # Resolve cross-language references
        self._resolve_multilingual_references(entity_map, recent_entities)
    
    def _resolve_multilingual_references(self, 
                                       entity_map: Dict[str, Dict[str, str]], 
                                       recent_entities: Dict[str, List[Dict[str, str]]]) -> None:
        """
        Resolve references across languages using alias mappings.
        
        Args:
            entity_map: Current entity map
            recent_entities: Recently mentioned entities by type
        """
        # Cross-language pronouns and references
        reference_patterns = {
            'employee': {
                'chinese': ['他', '她', '这个人', '那个人', '同事'],
                'english': ['he', 'she', 'they', 'him', 'her', 'them', 'this person', 'that person']
            },
            'project': {
                'chinese': ['这个项目', '那个项目', '该项目'],
                'english': ['it', 'this', 'that', 'this project', 'that project']
            },
            'department': {
                'chinese': ['这个部门', '那个部门', '该部门'],
                'english': ['this department', 'that department', 'the department']
            }
        }
        
        for placeholder, entity_info in entity_map.items():
            entity_value = entity_info.get('value', '').lower()
            entity_type = entity_info['type']
            
            # Check if this is a reference that needs resolution
            if entity_type in reference_patterns:
                patterns = reference_patterns[entity_type]
                is_reference = False
                
                # Check Chinese references
                for ref in patterns.get('chinese', []):
                    if ref in entity_value:
                        is_reference = True
                        break
                
                # Check English references
                if not is_reference:
                    for ref in patterns.get('english', []):
                        if ref in entity_value:
                            is_reference = True
                            break
                
                # If this is a reference, try to resolve it
                if is_reference and entity_type in recent_entities and recent_entities[entity_type]:
                    recent_entity = recent_entities[entity_type][-1]  # Most recent
                    resolved_entity = recent_entity.get('normalized', recent_entity.get('value'))
                    
                    if resolved_entity:
                        entity_info['normalized'] = resolved_entity
                        entity_info['resolved_from'] = 'multilingual_reference'
                        entity_info['original_reference'] = entity_value
                        logger.debug(f"Resolved reference '{entity_value}' to '{resolved_entity}'")


# Example usage and testing
if __name__ == "__main__":
    from .alias_mapper import AliasMapper
    
    # Create test entity dictionary
    test_entities = {
        'employee': ['田树君', '张三', '李四', 'John Smith', 'Jane Doe'],
        'department': ['物业管理部', '财务管理中心', 'Engineering', 'Marketing'],
        'project': ['金尚府项目', 'Website Redesign', 'Mobile App'],
        'work_type': ['施工现场配合', '材料下单', 'Meeting', 'Design']
    }
    
    # Create alias mapper
    alias_mapper = AliasMapper()
    
    # Create multilingual extractor
    extractor = MultilingualEntityExtractor(test_entities, alias_mapper)
    
    # Test queries
    test_queries = [
        "田树君在Q1 2025工作了多少小时？",
        "How many hours did Manager Tian work in Q1 2025?",
        "财务中心和Engineering部门的工作时间比较",
        "Compare hours between Finance Center and Engineering",
        "田经理在金尚项目上的工作统计",
        "Manager Tian's work stats on Jinshang project"
    ]
    
    print("Testing multilingual entity extraction:")
    for query in test_queries:
        print(f"\nQuery: {query}")
        template, entity_map = extractor.extract_and_normalize(query)
        print(f"Template: {template}")
        print(f"Entities: {entity_map}")
        
        # Show normalized forms
        for placeholder, info in entity_map.items():
            if info['value'] != info['normalized']:
                print(f"  Normalized: {info['value']} → {info['normalized']}")