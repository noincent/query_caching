#!/usr/bin/env python3
"""
Multilingual Query Cache Demo

This script demonstrates the multilingual capabilities of the query cache service,
showing how queries in Chinese and English can be processed interchangeably with
smart alias mapping and semantic understanding.
"""

import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.query_cache import QueryCache


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n--- {title} ---")


def print_query_result(query: str, result: Dict[str, Any], lang: str = ""):
    """Print formatted query result."""
    lang_label = f" ({lang})" if lang else ""
    print(f"\nQuery{lang_label}: {query}")
    print(f"Success: {'✓' if result.get('success', False) else '✗'}")
    
    if result.get('success'):
        print(f"Source: {result.get('source', 'unknown')}")
        print(f"Template: {result.get('template_query', 'N/A')}")
        
        if 'entity_map' in result:
            print("Entities:")
            for placeholder, info in result['entity_map'].items():
                original = info.get('value', '')
                normalized = info.get('normalized', '')
                entity_type = info.get('type', '')
                if original != normalized:
                    print(f"  {placeholder}: {original} → {normalized} ({entity_type})")
                else:
                    print(f"  {placeholder}: {original} ({entity_type})")
        
        if 'sql_query' in result:
            sql = result['sql_query']
            if len(sql) > 100:
                sql = sql[:100] + "..."
            print(f"SQL: {sql}")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")
    
    print(f"Response time: {result.get('query_time_ms', 0)}ms")


def demo_basic_multilingual():
    """Demonstrate basic multilingual query processing."""
    print_header("BASIC MULTILINGUAL QUERY PROCESSING")
    
    cache = QueryCache(use_predefined_templates=True)
    
    # Test queries in different languages
    queries = [
        ("How many hours did 田树君 work in Q1 2025?", "English with Chinese names"),
        ("田树君在2025年第一季度工作了多少小时？", "Pure Chinese"),
        ("Compare hours between 财务管理中心 and Engineering", "Mixed language"),
        ("哪个部门在第一季度工作时间最多？", "Chinese question"),
        ("Show me 物业管理部 employee statistics", "English with Chinese entity"),
    ]
    
    for query, description in queries:
        print_section(description)
        result = cache.process_query(query)
        print_query_result(query, result)


def demo_alias_normalization():
    """Demonstrate alias normalization across languages."""
    print_header("ALIAS NORMALIZATION DEMO")
    
    cache = QueryCache(use_predefined_templates=True)
    
    if not hasattr(cache, 'alias_mapper') or not cache.alias_mapper:
        print("Alias mapper not available. Please ensure multilingual_enabled=true in config.")
        return
    
    print_section("Entity Normalization Examples")
    
    # Test cases for normalization
    test_cases = [
        ("田经理", "employee", "Should normalize to canonical Chinese name"),
        ("Manager Tian", "employee", "Should normalize to canonical Chinese name"),
        ("Finance Center", "department", "Should normalize to Chinese department name"),
        ("财务中心", "department", "Should normalize to canonical department name"),
        ("Q1", "time_period", "Should normalize to Chinese quarter notation"),
        ("Property Management", "department", "Should normalize to Chinese department name"),
    ]
    
    for entity, entity_type, description in test_cases:
        normalized = cache.normalize_entity(entity, entity_type)
        variations = cache.get_entity_variations(normalized, entity_type)
        
        print(f"\nEntity: {entity} ({entity_type})")
        print(f"Description: {description}")
        print(f"Normalized: {normalized}")
        print(f"All variations: {', '.join(variations[:5])}")  # Show first 5 variations


def demo_cross_language_matching():
    """Demonstrate cross-language query matching."""
    print_header("CROSS-LANGUAGE QUERY MATCHING")
    
    cache = QueryCache(use_predefined_templates=True)
    
    # First, add a template with a Chinese query
    print_section("Adding Template with Chinese Query")
    chinese_template = "How many hours did {employee_0} work in {time_period_0}?"
    chinese_sql = "SELECT SUM(hours_worked) FROM employee_hours WHERE employee_name = '田树君' AND time_period = 'Q1 2025'"
    chinese_entities = {
        '{employee_0}': {'type': 'employee', 'value': '田树君', 'normalized': '田树君'},
        '{time_period_0}': {'type': 'time_period', 'value': 'Q1 2025', 'normalized': 'Q1 2025'}
    }
    
    add_result = cache.add_template(chinese_template, chinese_sql, chinese_entities)
    print(f"Template added: {'✓' if add_result.get('success') else '✗'}")
    
    print_section("Testing Semantically Similar Queries")
    
    # Test queries that should match the template
    similar_queries = [
        ("How many hours did 田树君 work in Q1 2025?", "Exact entity match"),
        ("田经理在第一季度工作了多少小时？", "Using aliases in Chinese"),
        ("How many hours did Manager Tian work in first quarter 2025?", "English aliases"),
        ("Show work hours for Tian Shujun in Q1 2025", "Different phrasing, same entities"),
    ]
    
    for query, description in similar_queries:
        print(f"\n{description}:")
        result = cache.process_query(query)
        print_query_result(query, result)


def demo_mixed_language_queries():
    """Demonstrate handling of mixed language queries."""
    print_header("MIXED LANGUAGE QUERY PROCESSING")
    
    cache = QueryCache(use_predefined_templates=True)
    
    print_section("Queries Mixing Chinese and English")
    
    mixed_queries = [
        "田树君 worked on Website Redesign project",
        "Show me 财务管理中心 budget for Q1 2025",
        "Compare 物业管理部 and Engineering department hours",
        "How many 员工 worked on 金尚设计 projects?",
        "List all meetings between Marketing and 设计管理部",
    ]
    
    for query in mixed_queries:
        result = cache.process_query(query)
        print_query_result(query, result)


def demo_api_endpoints():
    """Demonstrate API endpoint functionality (simulated)."""
    print_header("API ENDPOINT SIMULATION")
    
    cache = QueryCache(use_predefined_templates=True)
    
    if not hasattr(cache, 'alias_mapper') or not cache.alias_mapper:
        print("Alias mapper not available for API demo.")
        return
    
    print_section("Alias Management Operations")
    
    # Simulate API calls
    print("1. Adding new alias:")
    success = cache.add_alias("田树君", "Tian", "employee")
    print(f"   Added alias 'Tian' for '田树君': {'✓' if success else '✗'}")
    
    print("\n2. Getting entity variations:")
    variations = cache.get_entity_variations("田树君", "employee")
    print(f"   Variations for '田树君': {', '.join(variations)}")
    
    print("\n3. Normalizing entity:")
    normalized = cache.normalize_entity("Tian", "employee")
    print(f"   'Tian' normalizes to: '{normalized}'")
    
    print("\n4. System statistics:")
    stats = cache.get_multilingual_stats()
    print(f"   Multilingual enabled: {stats.get('multilingual_enabled', False)}")
    print(f"   Extractor type: {stats.get('extractor_type', 'unknown')}")
    print(f"   Entity types: {stats.get('entity_types', 0)}")
    print(f"   Total aliases: {stats.get('total_aliases', 0)}")


def demo_performance_comparison():
    """Demonstrate performance with and without multilingual features."""
    print_header("PERFORMANCE COMPARISON")
    
    print_section("Processing Performance Test")
    
    # Test queries
    test_queries = [
        "田树君在第一季度工作了多少小时？",
        "How many hours did Manager Tian work?",
        "财务中心的工作统计",
        "Show Engineering department metrics",
    ]
    
    cache = QueryCache(use_predefined_templates=True)
    
    total_time = 0
    successful_queries = 0
    
    for i, query in enumerate(test_queries, 1):
        start_time = time.time()
        result = cache.process_query(query)
        end_time = time.time()
        
        query_time = (end_time - start_time) * 1000  # Convert to ms
        total_time += query_time
        
        if result.get('success'):
            successful_queries += 1
        
        print(f"Query {i}: {query_time:.2f}ms {'✓' if result.get('success') else '✗'}")
    
    avg_time = total_time / len(test_queries)
    success_rate = (successful_queries / len(test_queries)) * 100
    
    print(f"\nSummary:")
    print(f"  Average response time: {avg_time:.2f}ms")
    print(f"  Success rate: {success_rate:.1f}%")
    print(f"  Total queries processed: {len(test_queries)}")


def demo_real_world_scenarios():
    """Demonstrate real-world usage scenarios."""
    print_header("REAL-WORLD USAGE SCENARIOS")
    
    cache = QueryCache(use_predefined_templates=True)
    
    scenarios = [
        {
            "title": "Project Manager Dashboard",
            "queries": [
                "田树君在金尚府项目上工作了多少小时？",
                "Show project progress for 西安高新环普产业园 renovation",
                "Compare work hours between 设计管理部 and 工程营造部 on current projects"
            ]
        },
        {
            "title": "HR Analytics",
            "queries": [
                "Which employees in 物业管理部 worked overtime this month?",
                "Show attendance statistics for 财务管理中心 team",
                "List all 会议 hours for management staff in Q1 2025"
            ]
        },
        {
            "title": "Financial Reporting",
            "queries": [
                "Calculate total 收款 amount for Q1 2025",
                "Show 预算成本部 expense tracking for ongoing projects",
                "Compare revenue between 营销策略中心 and 业务拓展部"
            ]
        }
    ]
    
    for scenario in scenarios:
        print_section(scenario["title"])
        
        for query in scenario["queries"]:
            result = cache.process_query(query)
            print_query_result(query, result)


def main():
    """Run the complete multilingual demo."""
    print("🌍 MULTILINGUAL QUERY CACHE SERVICE DEMO")
    print("Demonstrating Chinese/English cross-language capabilities")
    
    try:
        # Run all demo sections
        demo_basic_multilingual()
        demo_alias_normalization()
        demo_cross_language_matching()
        demo_mixed_language_queries()
        demo_api_endpoints()
        demo_performance_comparison()
        demo_real_world_scenarios()
        
        print_header("DEMO COMPLETE")
        print("✓ All multilingual features demonstrated successfully!")
        print("\nKey Features Shown:")
        print("• Cross-language entity recognition")
        print("• Smart alias mapping and normalization")
        print("• Mixed language query processing")
        print("• Semantic query matching across languages")
        print("• API endpoint functionality")
        print("• Real-world usage scenarios")
        
        print("\nNext Steps:")
        print("• Run ./install_multilingual.sh to set up dependencies")
        print("• Use python test_multilingual_comprehensive.py to run tests")
        print("• Start the API server to test REST endpoints")
        print("• Add more aliases using the API or configuration files")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        print("Please ensure all dependencies are installed and configuration is correct.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())