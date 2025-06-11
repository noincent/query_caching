#!/usr/bin/env python3
"""
Clear invalid cache entries with unresolved date placeholders.
"""

import requests
import sys

def clear_invalid_cache():
    """Clear cache entries with unresolved date placeholders."""
    try:
        response = requests.post('http://localhost:6000/clear_invalid_cache')
        
        if response.status_code == 200:
            result = response.json()
            print(f"Success: {result['message']}")
            print(f"Remaining valid templates: {result['remaining_templates']}")
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to cache service at localhost:6000")
        print("Make sure the cache service is running")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    clear_invalid_cache()