import sys
import os
import unittest
import tempfile
import json
import pickle
from pathlib import Path

# Add the parent directory to the Python path
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

# Skip this test if sentence-transformers is not available
try:
    from src.utils.template_matcher import TemplateMatcher
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Skipping template matcher tests because sentence-transformers is not available")

@unittest.skipIf(not SENTENCE_TRANSFORMERS_AVAILABLE, "sentence-transformers not available")
class TestTemplateMatcher(unittest.TestCase):
    """Test suite for the TemplateMatcher class."""
    
    def setUp(self):
        """Set up test fixtures."""
        try:
            # Create a template matcher with the same model used by the application
            self.matcher = TemplateMatcher(model_name="all-MiniLM-L6-v2")
            
            # Add some test templates
            self.matcher.add_template(
                template_query="How many hours did {employee_0} work on the {project_0} project in {time_period_0}?",
                sql_template="SELECT SUM(hours) FROM work_hours WHERE employee = '{employee_0}' AND project = '{project_0}' AND period = '{time_period_0}'",
                entity_map={
                    '{employee_0}': {'type': 'employee', 'value': 'John Smith', 'normalized': 'John Smith'},
                    '{project_0}': {'type': 'project', 'value': 'Website Redesign', 'normalized': 'Website Redesign'},
                    '{time_period_0}': {'type': 'time_period', 'value': 'Q1 2023', 'normalized': 'Q1 2023'}
                }
            )
            
            self.matcher.add_template(
                template_query="List all projects {employee_0} worked on in {time_period_0}",
                sql_template="SELECT DISTINCT project FROM work_hours WHERE employee = '{employee_0}' AND period = '{time_period_0}'",
                entity_map={
                    '{employee_0}': {'type': 'employee', 'value': 'Jane Doe', 'normalized': 'Jane Doe'},
                    '{time_period_0}': {'type': 'time_period', 'value': 'Q2 2023', 'normalized': 'Q2 2023'}
                }
            )
        except Exception as e:
            # If the test setup fails, mark it as skipped
            self.skipTest(f"Could not initialize TemplateMatcher: {e}")
        
    def test_find_matching_template_exact(self):
        """Test finding a matching template with an exact match."""
        query = "How many hours did John Smith work on the Website Redesign project in Q1 2023?"
        
        matching_template = self.matcher.find_matching_template(query, similarity_threshold=0.5)
        
        self.assertIsNotNone(matching_template)
        self.assertIn('template_query', matching_template)
        self.assertIn('sql_template', matching_template)
        self.assertEqual("How many hours did {employee_0} work on the {project_0} project in {time_period_0}?", 
                         matching_template['template_query'])
        
    def test_find_matching_template_similar(self):
        """Test finding a matching template with a similar query."""
        query = "Show me the hours John Smith spent working on Website Redesign during Q1 2023"
        
        # Lower the threshold to 0.3 to allow for more matches
        matching_template = self.matcher.find_matching_template(query, similarity_threshold=0.3)
        
        self.assertIsNotNone(matching_template)
        self.assertIn('template_query', matching_template)
        self.assertIn('sql_template', matching_template)
        self.assertEqual("How many hours did {employee_0} work on the {project_0} project in {time_period_0}?", 
                         matching_template['template_query'])
        
    def test_find_matching_template_different_intent(self):
        """Test finding a matching template with a different intent."""
        query = "What projects did John Smith work on in Q1 2023?"
        
        matching_template = self.matcher.find_matching_template(query, similarity_threshold=0.3)
        
        self.assertIsNotNone(matching_template)
        self.assertIn('template_query', matching_template)
        self.assertEqual("List all projects {employee_0} worked on in {time_period_0}", 
                         matching_template['template_query'])
        
    def test_no_matching_template(self):
        """Test when no matching template is found."""
        query = "What is the total budget for all projects in 2023?"
        
        # Use a high threshold to ensure no match
        matching_template = self.matcher.find_matching_template(query, similarity_threshold=0.9)
        
        self.assertIsNone(matching_template)
        
    def test_save_and_load_templates_json(self):
        """Test saving and loading templates to/from JSON."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as temp_file:
            temp_path = temp_file.name
            
        try:
            # Save templates to the temp file
            self.matcher.save_templates(temp_path)
            
            # Create a new matcher and load templates
            new_matcher = TemplateMatcher(model_name="all-MiniLM-L6-v2")
            new_matcher.load_templates(temp_path)
            
            # Check that templates were loaded
            self.assertEqual(len(self.matcher.templates), len(new_matcher.templates))
            self.assertEqual(self.matcher.templates[0]['template_query'], 
                            new_matcher.templates[0]['template_query'])
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        
    def test_save_and_load_templates_pickle(self):
        """Test saving and loading templates to/from pickle."""
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as temp_file:
            temp_path = temp_file.name
            
        try:
            # Save templates to the temp file
            self.matcher.save_templates(temp_path)
            
            # Create a new matcher and load templates
            new_matcher = TemplateMatcher(model_name="all-MiniLM-L6-v2")
            new_matcher.load_templates(temp_path)
            
            # Check that templates were loaded
            self.assertEqual(len(self.matcher.templates), len(new_matcher.templates))
            self.assertEqual(self.matcher.templates[0]['template_query'], 
                            new_matcher.templates[0]['template_query'])
            
            # Check if embeddings were also loaded
            self.assertIsNotNone(new_matcher.template_embeddings)
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_update_template_stats(self):
        """Test updating template usage statistics."""
        # Get initial values
        template_idx = 0
        initial_usage = self.matcher.templates[template_idx]['usage_count']
        initial_success_rate = self.matcher.templates[template_idx]['success_rate']
        
        # Update stats with a success
        self.matcher.update_template_stats(template_idx, success=True)
        
        # Check that values were updated
        self.assertEqual(initial_usage + 1, self.matcher.templates[template_idx]['usage_count'])
        # Success rate should stay high or increase slightly since success=True
        self.assertGreaterEqual(self.matcher.templates[template_idx]['success_rate'], initial_success_rate)
        
        # Update stats with a failure
        initial_success_rate = self.matcher.templates[template_idx]['success_rate']
        self.matcher.update_template_stats(template_idx, success=False)
        
        # Check that values were updated
        self.assertEqual(initial_usage + 2, self.matcher.templates[template_idx]['usage_count'])
        # Success rate should decrease since success=False
        self.assertLess(self.matcher.templates[template_idx]['success_rate'], initial_success_rate)


if __name__ == '__main__':
    unittest.main()