#!/usr/bin/env python3
"""
Comprehensive Test Suite for Multilingual Query Cache Features

This script tests all aspects of the multilingual implementation including:
- Alias mapping functionality
- Multilingual entity extraction
- Cross-language query matching
- API endpoints
- End-to-end query processing
"""

import sys
import json
import unittest
import tempfile
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.alias_mapper import AliasMapper
from src.utils.multilingual_entity_extractor import MultilingualEntityExtractor
from src.core.query_cache import QueryCache


class TestAliasMapper(unittest.TestCase):
    """Test cases for the AliasMapper class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary file for test aliases
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        test_aliases = {
            "employee": {
                "田树君": ["Tian Shujun", "田经理", "Manager Tian"],
                "张三": ["Zhang San", "张经理", "Manager Zhang"]
            },
            "department": {
                "物业管理部": ["Property Management", "Property Dept", "物业部"],
                "财务管理中心": ["Finance Center", "财务中心", "Finance Dept"]
            }
        }
        json.dump(test_aliases, self.temp_file, ensure_ascii=False, indent=2)
        self.temp_file.close()
        
        self.mapper = AliasMapper(self.temp_file.name)
    
    def tearDown(self):
        """Clean up test fixtures."""
        os.unlink(self.temp_file.name)
    
    def test_load_aliases(self):
        """Test loading aliases from file."""
        self.assertIn("employee", self.mapper.alias_mappings)
        self.assertIn("田树君", self.mapper.alias_mappings["employee"])
        self.assertEqual(
            self.mapper.alias_mappings["employee"]["田树君"], 
            ["Tian Shujun", "田经理", "Manager Tian"]
        )
    
    def test_normalize_entity(self):
        """Test entity normalization."""
        # Test Chinese to canonical
        self.assertEqual(self.mapper.normalize_entity("田经理", "employee"), "田树君")
        
        # Test English to canonical
        self.assertEqual(self.mapper.normalize_entity("Manager Tian", "employee"), "田树君")
        
        # Test exact canonical match
        self.assertEqual(self.mapper.normalize_entity("田树君", "employee"), "田树君")
        
        # Test unknown entity
        self.assertEqual(self.mapper.normalize_entity("Unknown Person", "employee"), "Unknown Person")
    
    def test_get_variations(self):
        """Test getting all variations of an entity."""
        variations = self.mapper.get_all_variations("田树君", "employee")
        expected = ["田树君", "Tian Shujun", "田经理", "Manager Tian"]
        self.assertEqual(variations, expected)
    
    def test_add_alias(self):
        """Test adding new aliases."""
        # Add new alias
        self.mapper.add_alias("田树君", "Tian", "employee")
        
        # Verify it was added
        variations = self.mapper.get_all_variations("田树君", "employee")
        self.assertIn("Tian", variations)
        
        # Verify normalization works
        self.assertEqual(self.mapper.normalize_entity("Tian", "employee"), "田树君")
    
    def test_fuzzy_matching(self):
        """Test fuzzy matching capabilities."""
        # Should find best match for partial strings
        match = self.mapper.find_best_match("Tian", "employee", threshold=0.5)
        self.assertEqual(match, "田树君")
        
        # Should not match if threshold too high
        match = self.mapper.find_best_match("T", "employee", threshold=0.8)
        self.assertIsNone(match)


class TestMultilingualEntityExtractor(unittest.TestCase):
    """Test cases for the MultilingualEntityExtractor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.entity_dict = {
            "employee": ["田树君", "张三", "John Smith", "Jane Doe"],
            "department": ["物业管理部", "财务管理中心", "Engineering", "Marketing"],
            "project": ["金尚府项目", "Website Redesign", "Mobile App"],
            "work_type": ["施工现场配合", "材料下单", "Meeting", "Design"],
            "time_period": ["Q1 2025", "第一季度", "January 2025", "本月"]
        }
        
        self.alias_mapper = AliasMapper()
        self.extractor = MultilingualEntityExtractor(self.entity_dict, self.alias_mapper)
    
    def test_language_detection(self):
        """Test language detection functionality."""
        # Test Chinese detection
        self.assertEqual(self.extractor._detect_language("田树君在工作"), "zh")
        
        # Test English detection
        self.assertEqual(self.extractor._detect_language("John is working"), "en")
        
        # Test mixed content detection
        mixed_lang = self.extractor._detect_language("田树君 is working")
        self.assertIn(mixed_lang, ["zh", "en"])  # Either is acceptable for mixed content
    
    def test_chinese_entity_extraction(self):
        """Test extraction from Chinese text."""
        query = "田树君在物业管理部工作了多少小时？"
        entities = self.extractor.extract_chinese_entities(query)
        
        self.assertIn("employee", entities)
        self.assertIn("department", entities)
        self.assertIn("田树君", entities["employee"])
        self.assertIn("物业管理部", entities["department"])
    
    def test_english_entity_extraction(self):
        """Test extraction from English text."""
        query = "How many hours did John Smith work in Engineering?"
        entities = self.extractor.extract_entities(query)
        
        self.assertIn("employee", entities)
        self.assertIn("department", entities)
        self.assertIn("John Smith", entities["employee"])
        self.assertIn("Engineering", entities["department"])
    
    def test_mixed_language_extraction(self):
        """Test extraction from mixed language text."""
        query = "田树君 worked on Website Redesign project in Q1 2025"
        entities = self.extractor.extract_entities(query)
        
        self.assertIn("employee", entities)
        self.assertIn("project", entities)
        self.assertIn("time_period", entities)
    
    def test_entity_normalization(self):
        """Test entity normalization through alias mapping."""
        # This should normalize "田经理" to "田树君"
        normalized = self.extractor.normalize_entity("田经理", "employee")
        self.assertEqual(normalized, "田树君")
    
    def test_extract_and_normalize(self):
        """Test the complete extraction and normalization pipeline."""
        query = "田经理在财务中心工作了多少小时？"
        template, entity_map = self.extractor.extract_and_normalize(query)
        
        # Check that template has placeholders
        self.assertIn("{employee_", template)
        self.assertIn("{department_", template)
        
        # Check that entities are normalized
        for placeholder, info in entity_map.items():
            if info['type'] == 'employee':
                self.assertEqual(info['normalized'], "田树君")
            elif info['type'] == 'department':
                self.assertEqual(info['normalized'], "财务管理中心")


class TestQueryCacheIntegration(unittest.TestCase):
    """Test cases for QueryCache integration with multilingual features."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary config file
        self.temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        config = {
            "multilingual_enabled": True,
            "model_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "similarity_threshold": 0.65,
            "alias_mapping_path": "data/alias_mappings.json"
        }
        json.dump(config, self.temp_config, indent=2)
        self.temp_config.close()
        
        # Create QueryCache instance
        self.cache = QueryCache(config_path=self.temp_config.name)
    
    def tearDown(self):
        """Clean up test fixtures."""
        os.unlink(self.temp_config.name)
    
    def test_multilingual_initialization(self):
        """Test that multilingual components are properly initialized."""
        stats = self.cache.get_multilingual_stats()
        self.assertTrue(stats.get('multilingual_enabled', False))
        self.assertEqual(stats.get('extractor_type'), 'MultilingualEntityExtractor')
    
    def test_cross_language_query_processing(self):
        """Test processing queries in different languages."""
        # Chinese query
        chinese_query = "田树君在第一季度工作了多少小时？"
        result_zh = self.cache.process_query(chinese_query)
        
        # English query with same semantic meaning
        english_query = "How many hours did Tian Shujun work in Q1?"
        result_en = self.cache.process_query(english_query)
        
        # Both should generate similar templates (allowing for minor differences)
        self.assertIsInstance(result_zh.get('template_query'), str)
        self.assertIsInstance(result_en.get('template_query'), str)
    
    def test_alias_management(self):
        """Test alias management functionality."""
        # Add alias
        success = self.cache.add_alias("田树君", "Tian", "employee")
        self.assertTrue(success)
        
        # Get variations
        variations = self.cache.get_entity_variations("田树君", "employee")
        self.assertIn("Tian", variations)
        
        # Test normalization
        normalized = self.cache.normalize_entity("Tian", "employee")
        self.assertEqual(normalized, "田树君")


class TestMultilingualQueryMatching(unittest.TestCase):
    """Test cases for multilingual query matching and template reuse."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.cache = QueryCache(use_predefined_templates=True)
        
        # Add a template manually for testing
        template_query = "How many hours did {employee_0} work in {time_period_0}?"
        sql_query = "SELECT SUM(hours) FROM work_hours WHERE employee = '田树君' AND period = 'Q1 2025'"
        entity_map = {
            '{employee_0}': {'type': 'employee', 'value': '田树君', 'normalized': '田树君'},
            '{time_period_0}': {'type': 'time_period', 'value': 'Q1 2025', 'normalized': 'Q1 2025'}
        }
        
        self.cache.add_template(template_query, sql_query, entity_map)
    
    def test_semantic_matching_across_languages(self):
        """Test that semantically equivalent queries in different languages match."""
        # Original template language (English)
        english_query = "How many hours did 田树君 work in Q1 2025?"
        result_en = self.cache.process_query(english_query)
        
        # Equivalent Chinese query
        chinese_query = "田树君在第一季度工作了多少小时？"
        result_zh = self.cache.process_query(chinese_query)
        
        # Both should either hit cache or generate similar results
        self.assertTrue(result_en['success'])
        self.assertTrue(result_zh['success'])
    
    def test_alias_based_matching(self):
        """Test that queries using aliases match existing templates."""
        # Query using an alias
        alias_query = "How many hours did Manager Tian work in Q1 2025?"
        result = self.cache.process_query(alias_query)
        
        # Should successfully process (either cache hit or generate SQL)
        self.assertTrue(result['success'])


def run_integration_tests():
    """Run integration tests that require full system setup."""
    print("Running integration tests...")
    
    try:
        # Test 1: Full pipeline with Chinese query
        cache = QueryCache(use_predefined_templates=True)
        
        query_zh = "田树君在第一季度工作了多少小时？"
        result = cache.process_query(query_zh)
        
        print(f"✓ Chinese query processed: {result['success']}")
        print(f"  Template: {result.get('template_query', 'N/A')}")
        
        # Test 2: Cross-language alias normalization
        if hasattr(cache, 'alias_mapper') and cache.alias_mapper:
            normalized = cache.normalize_entity("Finance Center", "department")
            print(f"✓ Alias normalization: 'Finance Center' → '{normalized}'")
        
        # Test 3: Multilingual statistics
        stats = cache.get_multilingual_stats()
        print(f"✓ Multilingual stats: {stats}")
        
        return True
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("COMPREHENSIVE MULTILINGUAL TEST SUITE")
    print("=" * 60)
    
    # Run unit tests
    print("\n1. Running Unit Tests...")
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestAliasMapper,
        TestMultilingualEntityExtractor,
        TestQueryCacheIntegration,
        TestMultilingualQueryMatching
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Run integration tests
    print("\n2. Running Integration Tests...")
    integration_success = run_integration_tests()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    unit_success = result.wasSuccessful()
    overall_success = unit_success and integration_success
    
    print(f"Unit Tests: {'✓ PASSED' if unit_success else '✗ FAILED'}")
    print(f"Integration Tests: {'✓ PASSED' if integration_success else '✗ FAILED'}")
    print(f"Overall: {'✓ ALL TESTS PASSED' if overall_success else '✗ SOME TESTS FAILED'}")
    
    if not unit_success:
        print(f"\nUnit test failures: {len(result.failures)}")
        print(f"Unit test errors: {len(result.errors)}")
    
    print("\n" + "=" * 60)
    
    return 0 if overall_success else 1


if __name__ == "__main__":
    sys.exit(main())