"""
Simple test that doesn't require any dependencies to verify the testing setup works
"""

import unittest
import sys
import os
from pathlib import Path

# Add the parent directory to the Python path
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

class TestSimple(unittest.TestCase):
    """Simple test suite to verify the testing setup works."""
    
    def test_basic_functionality(self):
        """Test that the testing framework works."""
        self.assertTrue(True)
        self.assertEqual(1 + 1, 2)
        
    def test_path_setup(self):
        """Test that the path setup is correct."""
        # Check that the parent directory is in the Python path
        self.assertIn(str(parent_dir), sys.path)
        
    def test_file_structure(self):
        """Test that the file structure is correct."""
        # Check that the main directories exist
        src_dir = parent_dir / "src"
        self.assertTrue(src_dir.exists())
        self.assertTrue(src_dir.is_dir())
        
        # Check for main subdirectories
        self.assertTrue((src_dir / "api").exists())
        self.assertTrue((src_dir / "core").exists())
        self.assertTrue((src_dir / "utils").exists())


if __name__ == '__main__':
    unittest.main()