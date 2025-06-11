# Query Cache Service

A standalone service for caching structured SQL query templates based on natural language queries. This service works alongside the CHESS system to improve performance for frequently asked question patterns, with special integration for WTL database systems.

## Features

- Extracts entities from natural language queries (employee names, project names, dates, etc.)
- Enhanced entity recognition using spaCy large model (en_core_web_lg) for better accuracy
- Identifies query intent and matches to known templates using semantic similarity
- Stores and retrieves SQL query templates with placeholders for entities
- Provides a simple REST API for integration with existing systems
- Improved entity extraction for time periods and dates
- Predefined templates for common query patterns
- Enhanced template matching with type-based entity replacement
- WTL-specific templates and entity dictionary
- Advanced visualization capabilities with automatic chart selection
- Direct integration with WTL database systems

## Quick Start

```bash
# Setup environment (one-time)
./setup_env.sh

# Install spaCy large model for better entity recognition (recommended)
./install_spacy_lg.sh

# Initialize database with sample data (one-time)
./initialize_db.sh

# Start the demo server
./demo.sh
```

This will start a web server on port 7000. Access the demo at http://localhost:7000.

### Running in API-only mode

```bash
# Start the API server without web interface
./run.sh
```

### Testing

```bash
# Run all tests
./test.sh

# Run specific test groups
./test.sh --simple      # Basic tests
./test.sh --entity      # Entity extraction tests
./test.sh --template    # Template matching tests
./test.sh --cache       # Query cache tests
./test.sh --library     # Template library tests
```

## Architecture

The service follows a straightforward workflow:

1. **When a user submits a query**:
   - The main server sends the query to the cache service first
   - The cache service extracts entities and looks for matching templates
   - If a match is found, it returns the SQL with entities replaced
   - If no match is found, it returns the extracted template and entities

2. **When no cache match is found**:
   - The main server sends the query to CHESS
   - CHESS generates the SQL query
   - The main server sends the query, generated SQL, and extracted entities to the cache service
   - The cache service creates a new template and stores it for future use

## Recent Improvements

### 1. Improved Entity Extraction
- Enhanced time period recognition (quarters, months, years, relative periods)
- Entity prioritization to avoid overlapping replacements
- Robust normalization to handle edge cases
- Support for department and work type entities

### 2. Predefined Template Library
- 10 common query templates ready to use
- 12 WTL-specific templates for advanced queries
- Templates for hours by employee/project, comparing hours, top employees/projects
- Support for department-specific queries and metrics

### 3. Robust Template Matching
- Type-based entity matching for consistent replacements
- Better handling of entity map generation
- Improved error handling for missing or invalid entities
- Enhanced similarity scoring for better matching

### 4. Enhanced Integration
- Cleaner API response format
- Better handling of cache misses
- Entity validation to prevent template errors
- Direct WTL database integration with pymysql

### 5. Advanced Visualization
- Automatic chart type selection based on query and data
- Support for bar, line, pie, heatmap and table visualizations
- Data export to CSV and Excel
- Customizable styling and titles

## Configuration

The service is configured using a JSON file with the following options:

```json
{
  "templates_path": "data/templates.pkl",           // Path to store templates
  "entity_dictionary_path": "data/entity_dictionary.json", // Path to entity dictionary
  "similarity_threshold": 0.75,                     // Minimum similarity score (0-1)
  "max_templates": 1000,                            // Maximum number of templates to store
  "model_name": "all-MiniLM-L6-v2",                // Sentence transformer model
  "use_predefined_templates": true,                 // Whether to use predefined templates
  "use_wtl_templates": true,                        // Whether to use WTL-specific templates
  "wtl_database_config": {                          // WTL database connection details
    "host": "localhost",
    "port": 3306,
    "user": "wtl_user",
    "password": "your_password_here",
    "database": "work_tracking"
  }
}
```

## API Endpoints

### Check If Query Is Cached

```
POST /query
```

Request body:
```json
{
  "query": "How many hours did John Smith work on the Website Redesign project in Q1 2023?"
}
```

Response (cache hit):
```json
{
  "success": true,
  "source": "cache",
  "sql_query": "SELECT SUM(hours) FROM work_hours WHERE employee = 'John Smith' AND project = 'Website Redesign' AND period = 'Q1 2023'",
  "template_id": 1,
  "similarity_score": 0.92,
  "template_query": "How many hours did {employee_0} work on the {project_0} project in {time_period_0}?",
  "entity_map": {
    "{employee_0}": {"type": "employee", "value": "John Smith", "normalized": "John Smith"},
    "{project_0}": {"type": "project", "value": "Website Redesign", "normalized": "Website Redesign"},
    "{time_period_0}": {"type": "time_period", "value": "Q1 2023", "normalized": "Q1 2023"}
  },
  "matching_template": "How many hours did {employee_0} work on the {project_0} project in {time_period_0}?"
}
```

Response (cache miss):
```json
{
  "success": false,
  "source": "cache",
  "error": "No matching template found",
  "template_query": "How many hours did {employee_0} work on the {project_0} project in {time_period_0}?",
  "entity_map": {
    "{employee_0}": {"type": "employee", "value": "John Smith", "normalized": "John Smith"},
    "{project_0}": {"type": "project", "value": "Website Redesign", "normalized": "Website Redesign"},
    "{time_period_0}": {"type": "time_period", "value": "Q1 2023", "normalized": "Q1 2023"}
  }
}
```

### Add Template

```
POST /add
```

Request body:
```json
{
  "template_query": "How many hours did {employee_0} work on the {project_0} project in {time_period_0}?",
  "sql_query": "SELECT SUM(hours) FROM work_hours WHERE employee = 'John Smith' AND project = 'Website Redesign' AND period = 'Q1 2023'",
  "entity_map": {
    "{employee_0}": {"type": "employee", "value": "John Smith", "normalized": "John Smith"},
    "{project_0}": {"type": "project", "value": "Website Redesign", "normalized": "Website Redesign"},
    "{time_period_0}": {"type": "time_period", "value": "Q1 2023", "normalized": "Q1 2023"}
  }
}
```

Response:
```json
{
  "success": true,
  "message": "Template added successfully"
}
```

### Other Endpoints

- `GET /health` - Health check and metrics
- `GET /templates` - List all templates in cache
- `GET /metrics` - Get performance metrics
- `POST /save` - Save current cache state

## Demo Usage

The web demo provides a user interface to:
- Submit natural language queries
- View SQL generation and results from the actual database
- See visualizations of query results using real data
- Browse available templates
- Explore the entity dictionary
- Monitor service metrics

### Example Queries

Try these example queries in the web interface:

- "How many hours did the 物业管理部 department work in Q1 2025?"
- "How many hours were spent on 施工现场配合 tasks in 2025?"
- "Compare hours between 财务管理中心 and 营销策略中心 departments in Q3 2025"
- "How many hours did 田树君 spend on 材料下单 tasks in Q2 2025?"
- "Who worked the most hours in the 金尚设计部 department during Q4 2025?"
- "What types of tasks did the 设计管理部 department work on in Q3 2025?"
- "What projects did the 工程营造部 department work on in Q2 2025?"
- "What percentage of 行政服务中心 department hours were spent on the 杭州软通2025系列小改造 project in Q2 2025?"

## WTL Integration

The easiest way to use the service with WTL database is through the `wtl_integration.py` script:

```bash
# Run the demonstration with sample queries
python wtl_integration.py --demo

# Process a specific query with visualization
python wtl_integration.py --query "How many hours did the Engineering department work in Q1 2023?" --output "engineering_hours.png"
```

For direct integration in your code:

```python
from wtl_integration import WTLQueryProcessor

# Initialize the processor
processor = WTLQueryProcessor()

# Process a query with visualization
result = processor.process_query(
    "Compare hours between Engineering and Marketing departments in Q3 2023",
    visualize=True
)

# Access the results
if result['success']:
    sql_query = result['sql_query']
    data = result['results']
    viz_path = result['visualization']
    chart_type = result['chart_type']
    
    # You can use these results in your application
    print(f"SQL Query: {sql_query}")
    print(f"Found {len(data)} results")
    print(f"Visualization saved to {viz_path}")
```

## Integration with CHESS

The integration follows a "Cache-First, CHESS-Fallback" approach:

1. Natural language queries are first checked against the Query Cache
2. If a matching template is found (cache hit), the SQL is generated immediately
3. If no matching template is found (cache miss), the query is forwarded to CHESS
4. Successful CHESS results are added to the cache for future use

Example integration flow:

```python
# Initialize components
cache_client = QueryCacheClient()
chess_interface = CHESSInterface()

# Process a query
def process_query(query):
    # First, try the cache
    cache_result = cache_client.process_query(query)
    
    if cache_result['success']:
        # Cache hit - use the cached SQL
        return {
            'source': 'cache',
            'sql_query': cache_result['sql_query']
        }
    else:
        # Cache miss - use CHESS
        chess_result = chess_interface.chat_query(query)
        
        # Add the result to the cache for future use
        cache_client.add_template(
            template_query=cache_result['template_query'],
            sql_query=chess_result['sql_query'],
            entity_map=cache_result['entity_map']
        )
        
        return {
            'source': 'chess',
            'sql_query': chess_result['sql_query']
        }
```

To run the demo web server with the integration:

```bash
./run.sh --demo
```

## Custom Templates

You can add custom templates for your specific queries:

1. Add templates directly via the API:
   ```python
   response = requests.post("http://localhost:6000/add", json={
       "template_query": "Show me the {department_0} department budget usage for {time_period_0}",
       "sql_query": "SELECT project, SUM(cost) FROM budget_items WHERE department = 'Engineering' AND period = 'Q1 2023' GROUP BY project",
       "entity_map": {
           "{department_0}": {"type": "department", "value": "Engineering", "normalized": "Engineering"},
           "{time_period_0}": {"type": "time_period", "value": "Q1 2023", "normalized": "Q1 2023"}
       }
   })
   ```

2. Add templates to the WTL template library in `src/core/wtl_templates.py`:
   ```python
   # Create a new template
   templates.append({
       'template_query': "Show me the {department_0} department budget usage for {time_period_0}",
       'sql_template': "SELECT project, SUM(cost) FROM budget_items WHERE department = '{department_0}' AND period = '{time_period_0}' GROUP BY project",
       'entity_map': {
           '{department_0}': {'type': 'department', 'value': 'PLACEHOLDER', 'normalized': 'PLACEHOLDER'},
           '{time_period_0}': {'type': 'time_period', 'value': 'PLACEHOLDER', 'normalized': 'PLACEHOLDER'}
       },
       'metadata': {
           'source': 'wtl_specialized',
           'description': 'Query for budget usage by department in a time period'
       }
   })
   ```

## Visualization Integration

The system includes a visualization module that automatically creates charts and graphs for query results. To integrate with your frontend:

1. **Use the file paths**
   The service generates visualization files which you can serve statically or embed in web pages:
   ```python
   result = processor.process_query("How many hours did each department work in Q1 2023?")
   visualization_path = result['visualization']
   # Then serve this file to the frontend
   ```

2. **Embed in web application**
   For a web application, add an endpoint that serves the visualization:
   ```python
   @app.route('/visualization/<query_id>')
   def show_visualization(query_id):
       # Get visualization path for this query
       viz_path = get_viz_path_for_query(query_id)
       return send_file(viz_path)
   ```

3. **Export data for frontend visualization**
   ```python
   result = processor.process_query("How many hours did each department work in Q1 2023?")
   data = result['results']
   # Send raw data to frontend for client-side visualization
   return jsonify(data)
   ```

## Troubleshooting

Common issues and solutions:

1. **Entity extraction issues**: 
   - Use `debug_entity_extraction.py` to see how entities are extracted
   - Add known entities to the entity dictionary for improved recognition

2. **Template matching problems**:
   - Adjust the similarity threshold (lower for more matches, higher for better precision)
   - Use `test_template_library.py` to test template matching

3. **Integration failures**:
   - Ensure entity maps are properly formatted
   - Check for empty or invalid normalized values

4. **Database connection issues**:
   - Verify database credentials in `config.json`
   - Check if database is accessible from your network
   - Ensure required tables exist with correct schema

5. **Demo mode issues**:
   - If port 7000 is in use, modify the port with `--port` option
   - If visualization issues occur, check that the `visualizations` directory exists and is writable
   - If database connection fails, the system will automatically fall back to demo mode with mock data

## Performance Optimization

For best performance:

1. Pre-populate the entity dictionary with known entities from your database
2. Enable predefined templates with `use_predefined_templates: true`
3. Run the service for a period to build up templates, then save the state
4. Periodically review and clean up low-quality templates
5. Adjust the similarity threshold based on your accuracy needs