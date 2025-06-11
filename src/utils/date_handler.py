"""
Date handling utility for the Query Cache Service.

This module provides helpers to convert time period strings into date ranges
for SQL queries.
"""

import logging
import re
from datetime import datetime, timedelta
from typing import Tuple, Dict

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_date_range(time_period: str) -> Tuple[str, str]:
    """
    Convert a time period string into a start and end date.
    
    Args:
        time_period: A string representing a time period (e.g., "Q1 2025", "January 2025", etc.)
        
    Returns:
        A tuple of (start_date, end_date) in YYYY-MM-DD format
    """
    current_year = datetime.now().year
    current_month = datetime.now().month
    current_day = datetime.now().day
    
    # Week range pattern (e.g., "2024-11-11 - 2024-11-17")
    week_match = re.match(r'(\d{4}-\d{2}-\d{2})\s*-\s*(\d{4}-\d{2}-\d{2})', time_period)
    if week_match:
        start_date = week_match.group(1)
        end_date = week_match.group(2)
        # Validate dates
        try:
            datetime.strptime(start_date, "%Y-%m-%d")
            datetime.strptime(end_date, "%Y-%m-%d")
            return start_date, end_date
        except ValueError:
            # If date validation fails, continue with other patterns
            pass
    
    # Quarter pattern (e.g., "Q1 2025")
    quarter_match = re.match(r'Q([1-4])\s+(\d{4})', time_period)
    if quarter_match:
        quarter = int(quarter_match.group(1))
        year = int(quarter_match.group(2))
        
        if quarter == 1:
            return f"{year}-01-01", f"{year}-03-31"
        elif quarter == 2:
            return f"{year}-04-01", f"{year}-06-30"
        elif quarter == 3:
            return f"{year}-07-01", f"{year}-09-30"
        elif quarter == 4:
            return f"{year}-10-01", f"{year}-12-31"
    
    # Month pattern (e.g., "January 2025")
    month_names = {
        "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
        "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12
    }
    
    for month_name, month_num in month_names.items():
        if month_name in time_period.lower():
            year_match = re.search(r'\b\d{4}\b', time_period)
            if year_match:
                year = int(year_match.group(0))
                
                # Calculate start and end dates for the month
                start_date = f"{year}-{month_num:02d}-01"
                
                # For end date, use the last day of the month
                if month_num == 12:
                    end_date = f"{year+1}-01-01"
                else:
                    end_date = f"{year}-{month_num+1:02d}-01"
                end_date = (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
                
                return start_date, end_date
    
    # Year pattern (e.g., "2025")
    year_match = re.match(r'(\d{4})', time_period)
    if year_match:
        year = int(year_match.group(1))
        return f"{year}-01-01", f"{year}-12-31"
    
    # Week number pattern (e.g., "Week 23 2025")
    week_num_match = re.match(r'week\s+(\d{1,2})(?:\s+of)?\s+(\d{4})', time_period.lower())
    if week_num_match:
        week_num = int(week_num_match.group(1))
        year = int(week_num_match.group(2))
        
        # Calculate the first day of the year
        first_day = datetime(year, 1, 1)
        
        # Calculate the first day of the week (Monday)
        first_monday = first_day + timedelta(days=(7 - first_day.weekday()) % 7)
        
        # Calculate the start day of the requested week
        start_date = first_monday + timedelta(weeks=week_num-1)
        
        # End date is 6 days after the start date (Sunday)
        end_date = start_date + timedelta(days=6)
        
        return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
    
    # Relative time periods
    if time_period.lower() == "this month":
        start = datetime(current_year, current_month, 1)
        if current_month == 12:
            end = datetime(current_year + 1, 1, 1) - timedelta(days=1)
        else:
            end = datetime(current_year, current_month + 1, 1) - timedelta(days=1)
        return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
        
    if time_period.lower() == "last month":
        if current_month == 1:
            start = datetime(current_year - 1, 12, 1)
            end = datetime(current_year, 1, 1) - timedelta(days=1)
        else:
            start = datetime(current_year, current_month - 1, 1)
            end = datetime(current_year, current_month, 1) - timedelta(days=1)
        return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
    
    if time_period.lower() == "this quarter":
        current_quarter = (current_month - 1) // 3 + 1
        if current_quarter == 1:
            return f"{current_year}-01-01", f"{current_year}-03-31"
        elif current_quarter == 2:
            return f"{current_year}-04-01", f"{current_year}-06-30"
        elif current_quarter == 3:
            return f"{current_year}-07-01", f"{current_year}-09-30"
        elif current_quarter == 4:
            return f"{current_year}-10-01", f"{current_year}-12-31"
    
    if time_period.lower() == "last quarter":
        current_quarter = (current_month - 1) // 3 + 1
        if current_quarter == 1:
            return f"{current_year-1}-10-01", f"{current_year-1}-12-31"
        elif current_quarter == 2:
            return f"{current_year}-01-01", f"{current_year}-03-31"
        elif current_quarter == 3:
            return f"{current_year}-04-01", f"{current_year}-06-30"
        elif current_quarter == 4:
            return f"{current_year}-07-01", f"{current_year}-09-30"
    
    if time_period.lower() == "this year":
        return f"{current_year}-01-01", f"{current_year}-12-31"
        
    if time_period.lower() == "last year":
        return f"{current_year-1}-01-01", f"{current_year-1}-12-31"
    
    # Default to current month if no match
    logger.warning(f"Could not parse time period: {time_period}. Using current month.")
    start = datetime(current_year, current_month, 1)
    if current_month == 12:
        end = datetime(current_year + 1, 1, 1) - timedelta(days=1)
    else:
        end = datetime(current_year, current_month + 1, 1) - timedelta(days=1)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def replace_time_period_placeholders(sql_query: str, time_periods: Dict[str, str]) -> str:
    """
    Replace time period placeholders in SQL with actual date ranges.
    
    Args:
        sql_query: SQL query with time period placeholders
        time_periods: Dictionary mapping time period placeholders to time period strings
        
    Returns:
        SQL query with time period placeholders replaced by actual date ranges
    """
    result = sql_query
    
    for placeholder, period in time_periods.items():
        start_date, end_date = get_date_range(period)
        
        # Replace start and end date placeholders
        result = result.replace(f"{placeholder}-start", start_date)
        result = result.replace(f"{placeholder}-end", end_date)
        
    return result