"""
Database Initialization Script

This script creates the necessary database tables and populates them with
sample data for the WTL Query Cache Service demo.
"""

import os
import sys
import json
import logging
import pymysql
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load config
config_path = "config.json"
if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
else:
    logger.error(f"Config file not found: {config_path}")
    sys.exit(1)

# Database configuration
db_config = config.get('wtl_database_config', {})
host = db_config.get('host', 'localhost')
port = db_config.get('port', 3306)
user = db_config.get('user', 'root')
password = db_config.get('password', '')
database = db_config.get('database', 'work_tracking')

# Sample data
employees = [
    "John Smith", "Jane Doe", "Bob Johnson", "Sarah Williams", 
    "Michael Chen", "Lisa Rodriguez", "David Wilson", "Emma Brown"
]

projects = [
    "Mobile App", "Website Redesign", "Database Migration", "API Development",
    "UI/UX Improvements", "Security Audit", "Performance Optimization", "Cloud Migration"
]

departments = [
    "Engineering", "Marketing", "Sales", "Finance", 
    "Human Resources", "Product", "Design", "Customer Support"
]

work_types = [
    "Development", "Testing", "Documentation", "Meeting",
    "Planning", "Research", "Design", "Review"
]

periods = [
    "Q1 2025", "Q2 2025", "Q3 2025", "Q4 2025"
]

# SQL statements
create_db_sql = f"CREATE DATABASE IF NOT EXISTS {database}"

# We'll adapt to the existing schema instead of creating new tables
# Get existing tables schema to work with them
describe_tables_sql = {
    "employee": "DESCRIBE employee",
    "project": "DESCRIBE project",
    "work_hour": "DESCRIBE work_hour",
    "team": "DESCRIBE team",
    "client": "DESCRIBE client",
    "users": "DESCRIBE users"
}

# Connect to MySQL and create the database
try:
    logger.info(f"Connecting to MySQL at {host}:{port} as {user}")
    conn = pymysql.connect(
        host=host,
        port=port,
        user=user,
        password=password
    )
    
    with conn.cursor() as cursor:
        # Create database if it doesn't exist
        logger.info(f"Creating database if not exists: {database}")
        cursor.execute(create_db_sql)
        conn.commit()
    
    conn.close()
    
    # Connect to the database
    logger.info(f"Connecting to database: {database}")
    conn = pymysql.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database=database,
        charset='utf8mb4'
    )
    
    with conn.cursor() as cursor:
        # Create tables
        logger.info("Creating work_hours table if not exists")
        cursor.execute(create_table_sql)
        conn.commit()
        
        # Check if data already exists
        cursor.execute("SELECT COUNT(*) FROM work_hours")
        count = cursor.fetchone()[0]
        
        if count > 0:
            logger.info(f"Data already exists in work_hours table ({count} records). Skipping data insertion.")
        else:
            # Generate sample data
            logger.info("Generating sample data...")
            
            import random
            
            # Map employees to departments
            employee_departments = {}
            for emp in employees:
                employee_departments[emp] = random.choice(departments)
            
            # Insert sample data
            insert_sql = """
            INSERT INTO work_hours (employee, project, department, work_type, period, hours)
            VALUES (%s, %s, %s, %s, %s, %s)
            """
            
            # Generate records
            records = []
            for period in periods:
                # Each employee works on 1-3 projects in each quarter
                for employee in employees:
                    department = employee_departments[employee]
                    emp_projects = random.sample(projects, random.randint(1, 3))
                    
                    for project in emp_projects:
                        # Each project involves 1-3 work types
                        emp_work_types = random.sample(work_types, random.randint(1, 3))
                        
                        for work_type in emp_work_types:
                            # Random hours between 10 and 100
                            hours = random.randint(10, 100)
                            records.append((employee, project, department, work_type, period, hours))
            
            # Insert records in batches
            batch_size = 100
            for i in range(0, len(records), batch_size):
                batch = records[i:i+batch_size]
                cursor.executemany(insert_sql, batch)
                conn.commit()
                logger.info(f"Inserted batch of {len(batch)} records ({i+len(batch)}/{len(records)})")
            
            logger.info(f"Successfully inserted {len(records)} records into work_hours table")
    
    logger.info("Database initialization completed successfully")
    
except Exception as e:
    logger.error(f"Error initializing database: {e}")
    sys.exit(1)
finally:
    if 'conn' in locals() and conn:
        conn.close()