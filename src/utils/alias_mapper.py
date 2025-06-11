"""
Alias Mapping System for Multilingual Entity Normalization

This module provides functionality for mapping entity aliases across different
languages to canonical forms, enabling consistent entity recognition regardless
of the language or variation used in queries.
"""

import json
import logging
from typing import Dict, List, Set, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class AliasMapper:
    """Handles multilingual aliases and entity normalization."""
    
    def __init__(self, alias_path: str = "data/alias_mappings.json"):
        """
        Initialize the alias mapper with mappings from file or defaults.
        
        Args:
            alias_path: Path to the alias mappings JSON file
        """
        self.alias_path = alias_path
        self.alias_mappings = self._load_aliases(alias_path)
        self.reverse_mappings = self._build_reverse_mappings()
        logger.info(f"Loaded {len(self.alias_mappings)} entity types with aliases")
        
    def _load_aliases(self, path: str) -> Dict[str, Dict[str, List[str]]]:
        """Load alias mappings from JSON file."""
        if Path(path).exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    mappings = json.load(f)
                logger.info(f"Loaded alias mappings from {path}")
                return mappings
            except Exception as e:
                logger.warning(f"Error loading alias mappings from {path}: {e}")
        
        logger.info("Using default alias mappings")
        return self._get_default_aliases()
    
    def _get_default_aliases(self) -> Dict[str, Dict[str, List[str]]]:
        """Default alias mappings for common entities."""
        return {
            "employee": {
                "田树君": ["Tian Shujun", "田经理", "Manager Tian", "Tian Manager"],
                "梅帅男": ["Mei Shuainan", "梅工程师", "Engineer Mei"],
                "徐佳宁": ["Xu Jianing", "徐总监", "Director Xu"],
                "柴铠": ["Chai Kai", "柴主任", "Supervisor Chai"],
                "路佳": ["Lu Jia", "路助理", "Assistant Lu"],
                "李慧": ["Li Hui", "李顾问", "Consultant Li"],
                "杨舒然": ["Yang Shuran", "杨专员", "Specialist Yang"],
                "徐思迪": ["Xu Sidi", "徐分析师", "Analyst Xu"],
                "陈亮": ["Chen Liang", "陈经理", "Manager Chen"],
                "孙碧云": ["Sun Biyun", "孙主管", "Supervisor Sun"],
                "张小雨": ["Zhang Xiaoyu", "张助理", "Assistant Zhang"],
                "胡靖铖": ["Hu Jingcheng", "胡工程师", "Engineer Hu"],
                "于秀丽": ["Yu Xiuli", "于经理", "Manager Yu"],
                "王童": ["Wang Tong", "王专员", "Specialist Wang"],
                "任玲格": ["Ren Lingge", "任顾问", "Consultant Ren"],
                "张军": ["Zhang Jun", "张总监", "Director Zhang"],
                "修盼盼": ["Xiu Panpan", "修助理", "Assistant Xiu"],
                "陈希朝": ["Chen Xichao", "陈经理", "Manager Chen"],
                "刘鹏": ["Liu Peng", "刘工程师", "Engineer Liu"],
            },
            "department": {
                "物业管理部": ["Property Management", "Property Dept", "物业部", "PM Dept", "Property Management Department"],
                "财务管理中心": ["Finance Center", "财务中心", "Financial Management", "财务部", "Finance Department"],
                "营销策略中心": ["Marketing Strategy Center", "营销中心", "Marketing", "Marketing Center", "MSC"],
                "行政服务中心": ["Administrative Service Center", "行政中心", "Admin Center", "ASC"],
                "综合管理部": ["General Management", "综合部", "GM Dept", "General Management Department"],
                "金尚设计部": ["Jinshang Design", "金尚设计", "JS Design", "Jinshang Design Department"],
                "品牌营销中心": ["Brand Marketing Center", "品牌中心", "Brand Center", "BMC"],
                "设计管理部": ["Design Management", "设计部", "Design Dept", "Design Management Department"],
                "工程营造部": ["Engineering Construction", "工程部", "Eng Dept", "Engineering Department"],
                "金尚工程部": ["Jinshang Engineering", "金尚工程", "JS Engineering"],
                "业务拓展部": ["Business Development", "拓展部", "BD Dept", "Business Development Department"],
                "售后服务组": ["After-sales Service", "售后组", "Service Group", "After-sales Group"],
                "软通组": ["Softcom Group", "软通", "SC Group"],
                "采集购买部": ["Procurement Department", "采购部", "Procurement", "Purchase Department"],
                "预算成本部": ["Budget Cost Department", "预算部", "Budget Dept", "Cost Department"],
            },
            "work_type": {
                "施工现场配合": ["On-site Coordination", "现场配合", "Site Support", "Construction Support"],
                "材料对接": ["Material Coordination", "材料配合", "Material Support"],
                "材料下单": ["Material Ordering", "材料采购", "Material Purchase", "Ordering Materials"],
                "预算部门会议": ["Budget Meeting", "预算会议", "Budget Department Meeting"],
                "现场管理": ["Site Management", "现场监管", "On-site Management"],
                "施工图商务邮件": ["Construction Drawing Business Email", "图纸邮件", "Drawing Email"],
                "客户沟通": ["Client Communication", "客户交流", "Customer Communication"],
                "平面方案": ["Floor Plan", "平面设计", "Layout Design"],
                "水电现场勘探": ["MEP Site Survey", "水电勘探", "Utilities Survey"],
                "施工图绘制": ["Construction Drawing", "图纸绘制", "Drawing Production"],
                "弱电图纸": ["Low Voltage Drawing", "弱电设计", "LV Design"],
                "员工关系": ["Employee Relations", "人员关系", "HR Relations"],
                "施工配合": ["Construction Coordination", "施工支持", "Construction Support"],
                "设计结算": ["Design Settlement", "设计收尾", "Design Finalization"],
                "收款": ["Payment Collection", "款项收取", "Revenue Collection"],
            },
            "time_period": {
                "第一季度": ["Q1", "Q1季度", "first quarter", "1st quarter"],
                "第二季度": ["Q2", "Q2季度", "second quarter", "2nd quarter"],
                "第三季度": ["Q3", "Q3季度", "third quarter", "3rd quarter"],
                "第四季度": ["Q4", "Q4季度", "fourth quarter", "4th quarter"],
                "今年": ["this year", "本年度", "current year"],
                "去年": ["last year", "上年度", "previous year"],
                "本月": ["this month", "当月", "current month"],
                "上月": ["last month", "上个月", "previous month"],
                "Q1 2025": ["2025年第一季度", "2025 Q1", "2025年一季度"],
                "Q2 2025": ["2025年第二季度", "2025 Q2", "2025年二季度"],
                "Q3 2025": ["2025年第三季度", "2025 Q3", "2025年三季度"],
                "Q4 2025": ["2025年第四季度", "2025 Q4", "2025年四季度"],
            }
        }
    
    def _build_reverse_mappings(self) -> Dict[str, Dict[str, str]]:
        """Build reverse mappings for quick lookup."""
        reverse = {}
        
        for entity_type, mappings in self.alias_mappings.items():
            reverse[entity_type] = {}
            for canonical, aliases in mappings.items():
                # Map canonical to itself
                reverse[entity_type][canonical.lower()] = canonical
                # Map each alias to canonical
                for alias in aliases:
                    reverse[entity_type][alias.lower()] = canonical
        
        return reverse
    
    def normalize_entity(self, entity: str, entity_type: str) -> str:
        """
        Normalize an entity to its canonical form.
        
        Args:
            entity: Entity string to normalize
            entity_type: Type of the entity
            
        Returns:
            Canonical form of the entity
        """
        if not entity or not entity_type:
            return entity
            
        if entity_type not in self.reverse_mappings:
            return entity
            
        entity_lower = entity.lower().strip()
        canonical = self.reverse_mappings[entity_type].get(entity_lower, entity)
        
        if canonical != entity:
            logger.debug(f"Normalized '{entity}' to '{canonical}' for type {entity_type}")
        
        return canonical
    
    def get_all_variations(self, canonical: str, entity_type: str) -> List[str]:
        """
        Get all variations of a canonical entity.
        
        Args:
            canonical: Canonical entity name
            entity_type: Type of the entity
            
        Returns:
            List of all variations including the canonical form
        """
        if entity_type not in self.alias_mappings:
            return [canonical]
            
        variations = [canonical]
        if canonical in self.alias_mappings[entity_type]:
            variations.extend(self.alias_mappings[entity_type][canonical])
            
        return variations
    
    def add_alias(self, canonical: str, alias: str, entity_type: str):
        """
        Add a new alias mapping.
        
        Args:
            canonical: Canonical entity name
            alias: Alias to map to the canonical name
            entity_type: Type of the entity
        """
        if entity_type not in self.alias_mappings:
            self.alias_mappings[entity_type] = {}
        
        if canonical not in self.alias_mappings[entity_type]:
            self.alias_mappings[entity_type][canonical] = []
            
        if alias not in self.alias_mappings[entity_type][canonical]:
            self.alias_mappings[entity_type][canonical].append(alias)
            # Update reverse mapping
            if entity_type not in self.reverse_mappings:
                self.reverse_mappings[entity_type] = {}
            self.reverse_mappings[entity_type][alias.lower()] = canonical
            logger.info(f"Added alias '{alias}' for '{canonical}' (type: {entity_type})")
    
    def remove_alias(self, canonical: str, alias: str, entity_type: str) -> bool:
        """
        Remove an alias mapping.
        
        Args:
            canonical: Canonical entity name
            alias: Alias to remove
            entity_type: Type of the entity
            
        Returns:
            True if removed successfully, False if not found
        """
        if (entity_type in self.alias_mappings and 
            canonical in self.alias_mappings[entity_type] and
            alias in self.alias_mappings[entity_type][canonical]):
            
            self.alias_mappings[entity_type][canonical].remove(alias)
            
            # Update reverse mapping
            if (entity_type in self.reverse_mappings and 
                alias.lower() in self.reverse_mappings[entity_type]):
                del self.reverse_mappings[entity_type][alias.lower()]
            
            logger.info(f"Removed alias '{alias}' for '{canonical}' (type: {entity_type})")
            return True
        
        return False
    
    def get_canonical_entities(self, entity_type: str) -> List[str]:
        """
        Get all canonical entities for a given type.
        
        Args:
            entity_type: Type of entities to retrieve
            
        Returns:
            List of canonical entity names
        """
        if entity_type not in self.alias_mappings:
            return []
        return list(self.alias_mappings[entity_type].keys())
    
    def get_entity_types(self) -> List[str]:
        """Get all available entity types."""
        return list(self.alias_mappings.keys())
    
    def find_best_match(self, entity: str, entity_type: str, threshold: float = 0.8) -> Optional[str]:
        """
        Find the best matching canonical entity using fuzzy matching.
        
        Args:
            entity: Entity to match
            entity_type: Type of the entity
            threshold: Similarity threshold for matching
            
        Returns:
            Best matching canonical entity or None
        """
        if entity_type not in self.alias_mappings:
            return None
        
        entity_lower = entity.lower().strip()
        
        # First try exact match
        if entity_lower in self.reverse_mappings.get(entity_type, {}):
            return self.reverse_mappings[entity_type][entity_lower]
        
        # Try fuzzy matching (simple substring matching for now)
        best_match = None
        best_score = 0
        
        for canonical, aliases in self.alias_mappings[entity_type].items():
            all_variations = [canonical] + aliases
            
            for variation in all_variations:
                variation_lower = variation.lower()
                
                # Simple similarity scoring
                if entity_lower in variation_lower or variation_lower in entity_lower:
                    score = min(len(entity_lower), len(variation_lower)) / max(len(entity_lower), len(variation_lower))
                    if score > best_score and score >= threshold:
                        best_score = score
                        best_match = canonical
        
        if best_match:
            logger.debug(f"Fuzzy matched '{entity}' to '{best_match}' (score: {best_score:.3f})")
        
        return best_match
    
    def save_aliases(self, path: str = None):
        """
        Save alias mappings to file.
        
        Args:
            path: Path to save to (defaults to the original path)
        """
        save_path = path or self.alias_path
        
        try:
            # Ensure directory exists
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(self.alias_mappings, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Saved alias mappings to {save_path}")
        except Exception as e:
            logger.error(f"Error saving alias mappings to {save_path}: {e}")
    
    def get_stats(self) -> Dict[str, int]:
        """Get statistics about the alias mappings."""
        stats = {
            'entity_types': len(self.alias_mappings),
            'total_canonical_entities': 0,
            'total_aliases': 0
        }
        
        for entity_type, mappings in self.alias_mappings.items():
            stats['total_canonical_entities'] += len(mappings)
            for aliases in mappings.values():
                stats['total_aliases'] += len(aliases)
        
        return stats


# Example usage and testing
if __name__ == "__main__":
    # Create mapper with test data
    mapper = AliasMapper()
    
    # Test normalization
    test_cases = [
        ("财务中心", "department", "财务管理中心"),
        ("Finance Center", "department", "财务管理中心"),
        ("Q1", "time_period", "第一季度"),
        ("田经理", "employee", "田树君"),
        ("Property Management", "department", "物业管理部"),
    ]
    
    print("Testing alias normalization:")
    for entity, entity_type, expected in test_cases:
        result = mapper.normalize_entity(entity, entity_type)
        status = "✓" if result == expected else "✗"
        print(f"{status} {entity} ({entity_type}) → {result} (expected: {expected})")
    
    # Test variations
    print(f"\nVariations for '物业管理部': {mapper.get_all_variations('物业管理部', 'department')}")
    
    # Test adding new alias
    mapper.add_alias("田树君", "Tian", "employee")
    print(f"田树君 variations after adding 'Tian': {mapper.get_all_variations('田树君', 'employee')}")
    
    # Test stats
    print(f"\nAlias mapper statistics: {mapper.get_stats()}")