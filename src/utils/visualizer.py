"""
Query Result Visualizer

This module provides advanced visualization capabilities for SQL query results,
supporting various chart types and automatic visualization selection based on
query content and result structure.
"""

import os
import time
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set seaborn style
sns.set_theme(style="whitegrid")
sns.set_context("paper")

class QueryVisualizer:
    """
    Advanced query result visualization with automatic chart selection
    and multiple output formats.
    """
    
    def __init__(self, output_dir: str = "visualizations"):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Set default color palette
        self.color_palette = "viridis"
        sns.set_palette(self.color_palette)
        
    def to_dataframe(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Convert database results to DataFrame.
        
        Args:
            results: List of dictionaries from database query
            
        Returns:
            Pandas DataFrame
        """
        if not results:
            return pd.DataFrame()
            
        df = pd.DataFrame(results)
        
        # Convert date columns
        for col in df.columns:
            if col.lower() in ('date', 'datetime', 'time', 'timestamp', 'created_at', 'updated_at'):
                try:
                    df[col] = pd.to_datetime(df[col])
                except:
                    pass
                    
        return df
    
    def visualize(self, 
                  df: pd.DataFrame, 
                  query_type: str = None, 
                  query_text: str = None,
                  chart_type: str = None,
                  title: str = None,
                  output_file: str = None) -> Dict[str, Any]:
        """
        Generate visualizations based on data and query.
        
        Args:
            df: DataFrame with query results
            query_type: Type of query (hoursByPerson, hoursByProject, etc.)
            query_text: Original natural language query
            chart_type: Specific chart type to use (bar, line, pie, etc.)
            title: Custom title for the visualization
            output_file: Optional path to save visualization
            
        Returns:
            Dictionary with visualization info
        """
        if df.empty:
            return {"success": False, "error": "No data to visualize"}
        
        # Determine best chart type if not specified
        if not chart_type:
            chart_type = self._determine_chart_type(df, query_type, query_text)
            
        # Generate title if not provided
        if not title:
            title = self._generate_title(query_type, query_text)
            
        # Create output path if not specified
        if not output_file:
            timestamp = int(time.time())
            output_file = os.path.join(self.output_dir, f"viz_{chart_type}_{timestamp}.png")
            
        # Create visualization based on chart type
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if chart_type == "bar":
                self._create_bar_chart(df, ax, title)
            elif chart_type == "horizontal_bar":
                self._create_horizontal_bar_chart(df, ax, title)
            elif chart_type == "line":
                self._create_line_chart(df, ax, title)
            elif chart_type == "pie":
                self._create_pie_chart(df, fig, title)
            elif chart_type == "stacked_bar":
                self._create_stacked_bar_chart(df, ax, title)
            elif chart_type == "heatmap":
                self._create_heatmap(df, ax, title)
            elif chart_type == "scatter":
                self._create_scatter_plot(df, ax, title)
            elif chart_type == "table":
                self._create_table_view(df, ax, title)
            else:
                # Default to bar chart
                self._create_bar_chart(df, ax, title)
                
            # Save visualization
            plt.tight_layout()
            plt.savefig(output_file, dpi=300)
            plt.close(fig)
            
            logger.info(f"Visualization saved to {output_file}")
            
            return {
                "success": True,
                "chart_type": chart_type,
                "title": title,
                "file_path": output_file,
                "data_shape": df.shape
            }
            
        except Exception as e:
            logger.error(f"Error creating visualization: {e}")
            return {
                "success": False,
                "error": str(e),
                "chart_type": chart_type
            }
    
    def _determine_chart_type(self, df: pd.DataFrame, query_type: str = None, query_text: str = None) -> str:
        """
        Determine the best chart type based on data and query.
        
        Args:
            df: DataFrame with data
            query_type: Type of query if known
            query_text: Original query text
            
        Returns:
            Chart type string
        """
        # Number of columns and rows
        num_cols = df.shape[1]
        num_rows = df.shape[0]
        
        # Check query text keywords if available
        if query_text:
            query_text = query_text.lower()
            
            # Distribution or percentage queries
            if any(word in query_text for word in ["distribution", "breakdown", "percentage", "percent", "proportion", "share"]):
                return "pie" if num_rows <= 7 else "horizontal_bar"
                
            # Time series queries
            if any(word in query_text for word in ["trend", "over time", "by month", "by year", "by quarter"]):
                return "line"
                
            # Comparison queries
            if any(word in query_text for word in ["compare", "comparison", "versus", "vs"]):
                return "bar" if num_rows <= 10 else "horizontal_bar"
                
            # Ranking queries
            if any(word in query_text for word in ["top", "bottom", "rank", "ranking", "most", "least"]):
                return "horizontal_bar" if num_rows > 6 else "bar"
                
        # Check query type if available
        if query_type:
            query_type = query_type.lower()
            
            if "distribution" in query_type or "percentage" in query_type:
                return "pie" if num_rows <= 7 else "horizontal_bar"
                
            if "trend" in query_type or "overtime" in query_type:
                return "line"
                
            if "compare" in query_type:
                return "bar" if num_rows <= 10 else "horizontal_bar"
                
            if "rank" in query_type or "top" in query_type:
                return "horizontal_bar" if num_rows > 6 else "bar"
                
        # Determine by data structure
        if num_cols == 2:
            # Two columns typically means a category and a value
            if num_rows > 10:
                return "horizontal_bar"  # Better for many categories
            else:
                return "bar"
                
        elif num_cols > 3 and num_rows > 10:
            # Many columns and rows, possibly a complex relationship
            return "heatmap"
            
        elif num_cols > 3 and num_rows <= 10:
            # Several metrics across a few categories
            return "stacked_bar"
            
        elif num_cols >= 3 and any(isinstance(col, pd.DatetimeIndex) for col in [df.index] + [df[col] for col in df.columns if df[col].dtype == 'datetime64[ns]']):
            # Time series with multiple metrics
            return "line"
            
        elif num_rows > 20 or num_cols > 5:
            # Large datasets are hard to visualize, use a table
            return "table"
            
        # Default to bar for simplicity
        return "bar"
    
    def _generate_title(self, query_type: str = None, query_text: str = None) -> str:
        """
        Generate a chart title based on query information.
        
        Args:
            query_type: Type of query if known
            query_text: Original query text
            
        Returns:
            Generated title
        """
        # Use query text if available
        if query_text:
            # Capitalize first letter
            title = query_text[0].upper() + query_text[1:]
            
            # Remove question mark if present
            if title.endswith('?'):
                title = title[:-1]
                
            # Truncate if too long
            if len(title) > 60:
                title = title[:57] + "..."
                
            return title
        
        # Use query type if available
        if query_type:
            # Convert camelCase to Title Case with spaces
            import re
            words = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)', query_type)
            title = ' '.join(words).title()
            return title
        
        # Default title
        return "Query Results"
    
    def _create_bar_chart(self, df: pd.DataFrame, ax: plt.Axes, title: str) -> None:
        """
        Create a bar chart visualization.
        
        Args:
            df: DataFrame with data
            ax: Matplotlib axes
            title: Chart title
        """
        # For bar charts, we typically need one category column and one value column
        if df.shape[1] < 2:
            raise ValueError("Bar chart requires at least two columns (category and value)")
            
        # Identify category and value columns
        category_col = df.columns[0]  # Assume first column is category
        value_col = df.columns[1]     # Assume second column is value
        
        # Sort by value descending
        df_sorted = df.sort_values(by=value_col, ascending=False)
        
        # Create the bar chart
        sns.barplot(x=category_col, y=value_col, data=df_sorted, ax=ax)
        
        # Customize
        ax.set_title(title)
        ax.set_xlabel(category_col.replace('_', ' ').title())
        ax.set_ylabel(value_col.replace('_', ' ').title())
        
        # Rotate x-axis labels if needed
        if df.shape[0] > 5 or df[category_col].str.len().max() > 10:
            plt.xticks(rotation=45, ha='right')
    
    def _create_horizontal_bar_chart(self, df: pd.DataFrame, ax: plt.Axes, title: str) -> None:
        """
        Create a horizontal bar chart visualization.
        
        Args:
            df: DataFrame with data
            ax: Matplotlib axes
            title: Chart title
        """
        # For bar charts, we typically need one category column and one value column
        if df.shape[1] < 2:
            raise ValueError("Horizontal bar chart requires at least two columns (category and value)")
            
        # Identify category and value columns
        category_col = df.columns[0]  # Assume first column is category
        value_col = df.columns[1]     # Assume second column is value
        
        # Sort by value descending
        df_sorted = df.sort_values(by=value_col, ascending=True)  # Ascending for horizontal bars
        
        # Create the horizontal bar chart
        sns.barplot(y=category_col, x=value_col, data=df_sorted, ax=ax)
        
        # Customize
        ax.set_title(title)
        ax.set_ylabel(category_col.replace('_', ' ').title())
        ax.set_xlabel(value_col.replace('_', ' ').title())
    
    def _create_line_chart(self, df: pd.DataFrame, ax: plt.Axes, title: str) -> None:
        """
        Create a line chart visualization.
        
        Args:
            df: DataFrame with data
            ax: Matplotlib axes
            title: Chart title
        """
        # For line charts, we typically need one time column and one or more value columns
        if df.shape[1] < 2:
            raise ValueError("Line chart requires at least two columns (time and value)")
            
        # Try to identify time column
        time_col = None
        for col in df.columns:
            if col.lower() in ('date', 'datetime', 'time', 'timestamp', 'period', 'month', 'year', 'quarter'):
                time_col = col
                break
                
        if not time_col:
            # Assume first column is time/category
            time_col = df.columns[0]
            
        # Try to convert to datetime if not already
        if df[time_col].dtype != 'datetime64[ns]':
            try:
                df[time_col] = pd.to_datetime(df[time_col])
            except:
                pass  # Keep as is if conversion fails
        
        # Sort by time
        if df[time_col].dtype == 'datetime64[ns]':
            df = df.sort_values(by=time_col)
            
        # Get value columns
        value_cols = [col for col in df.columns if col != time_col]
        
        # Plot each value column
        for col in value_cols:
            ax.plot(df[time_col], df[col], marker='o', label=col.replace('_', ' ').title())
            
        # Customize
        ax.set_title(title)
        ax.set_xlabel(time_col.replace('_', ' ').title())
        ax.set_ylabel('Value')
        
        # Format x-axis if datetime
        if df[time_col].dtype == 'datetime64[ns]':
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45, ha='right')
            
        # Add legend if multiple value columns
        if len(value_cols) > 1:
            ax.legend()
    
    def _create_pie_chart(self, df: pd.DataFrame, fig: plt.Figure, title: str) -> None:
        """
        Create a pie chart visualization.
        
        Args:
            df: DataFrame with data
            fig: Matplotlib figure
            title: Chart title
        """
        # For pie charts, we typically need one category column and one value column
        if df.shape[1] < 2:
            raise ValueError("Pie chart requires at least two columns (category and value)")
            
        # Identify category and value columns
        category_col = df.columns[0]  # Assume first column is category
        value_col = df.columns[1]     # Assume second column is value
        
        # Sort by value descending
        df_sorted = df.sort_values(by=value_col, ascending=False)
        
        # If we have too many categories, group small ones
        if df.shape[0] > 7:
            # Keep top 6 and group the rest as "Other"
            top_6 = df_sorted.iloc[:6]
            other = pd.DataFrame({
                category_col: ['Other'],
                value_col: [df_sorted.iloc[6:][value_col].sum()]
            })
            df_plot = pd.concat([top_6, other])
        else:
            df_plot = df_sorted
            
        # Create the pie chart
        plt.pie(
            df_plot[value_col],
            labels=df_plot[category_col],
            autopct='%1.1f%%', 
            startangle=90,
            shadow=False
        )
        
        # Equal aspect ratio ensures that pie is drawn as a circle
        plt.axis('equal')
        
        # Add title
        plt.title(title)
    
    def _create_stacked_bar_chart(self, df: pd.DataFrame, ax: plt.Axes, title: str) -> None:
        """
        Create a stacked bar chart visualization.
        
        Args:
            df: DataFrame with data
            ax: Matplotlib axes
            title: Chart title
        """
        # For stacked bars, we need at least three columns
        if df.shape[1] < 3:
            raise ValueError("Stacked bar chart requires at least three columns")
            
        # Assume first column is the primary category
        category_col = df.columns[0]
        
        # Use all other columns as stacked values
        value_cols = [col for col in df.columns if col != category_col]
        
        # Create the stacked bar chart
        df.set_index(category_col)[value_cols].plot(kind='bar', stacked=True, ax=ax)
        
        # Customize
        ax.set_title(title)
        ax.set_xlabel(category_col.replace('_', ' ').title())
        ax.set_ylabel('Value')
        
        # Rotate x-axis labels if needed
        if df.shape[0] > 5 or df[category_col].str.len().max() > 10:
            plt.xticks(rotation=45, ha='right')
    
    def _create_heatmap(self, df: pd.DataFrame, ax: plt.Axes, title: str) -> None:
        """
        Create a heatmap visualization.
        
        Args:
            df: DataFrame with data
            ax: Matplotlib axes
            title: Chart title
        """
        # Try to identify if we have a natural pivot structure
        if df.shape[1] >= 3:
            # If we have three columns, try to pivot
            try:
                pivot_cols = df.columns[:3]
                pivot_df = df.pivot(index=pivot_cols[0], columns=pivot_cols[1], values=pivot_cols[2])
                sns.heatmap(pivot_df, annot=True, fmt=".1f", linewidths=.5, ax=ax, cmap="YlGnBu")
            except:
                # If pivot fails, just use the data as is
                sns.heatmap(df.select_dtypes(include=[np.number]), annot=True, fmt=".1f", linewidths=.5, ax=ax, cmap="YlGnBu")
        else:
            # Just use numeric columns
            numeric_df = df.select_dtypes(include=[np.number])
            if numeric_df.empty:
                raise ValueError("Heatmap requires numeric data")
            sns.heatmap(numeric_df, annot=True, fmt=".1f", linewidths=.5, ax=ax, cmap="YlGnBu")
        
        # Customize
        ax.set_title(title)
    
    def _create_scatter_plot(self, df: pd.DataFrame, ax: plt.Axes, title: str) -> None:
        """
        Create a scatter plot visualization.
        
        Args:
            df: DataFrame with data
            ax: Matplotlib axes
            title: Chart title
        """
        # For scatter plots, we need at least two numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            raise ValueError("Scatter plot requires at least two numeric columns")
            
        # Use first two numeric columns
        x_col = numeric_cols[0]
        y_col = numeric_cols[1]
        
        # Create the scatter plot
        sns.scatterplot(x=x_col, y=y_col, data=df, ax=ax)
        
        # Customize
        ax.set_title(title)
        ax.set_xlabel(x_col.replace('_', ' ').title())
        ax.set_ylabel(y_col.replace('_', ' ').title())
    
    def _create_table_view(self, df: pd.DataFrame, ax: plt.Axes, title: str) -> None:
        """
        Create a table visualization.
        
        Args:
            df: DataFrame with data
            ax: Matplotlib axes
            title: Chart title
        """
        # Create a table view (useful for complex data)
        ax.axis('off')
        ax.axis('tight')
        
        # Limit to first 15 rows and 8 columns for readability
        display_df = df.iloc[:15, :8].copy()
        
        # Truncate long strings
        for col in display_df.select_dtypes(include=['object']).columns:
            display_df[col] = display_df[col].astype(str).apply(lambda x: x[:20] + '...' if len(x) > 20 else x)
            
        # Create table
        table = ax.table(
            cellText=display_df.values,
            colLabels=display_df.columns,
            cellLoc='center', 
            loc='center'
        )
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Add title
        ax.set_title(title)
        
    def export_to_excel(self, df: pd.DataFrame, output_file: str = None) -> str:
        """
        Export query results to Excel.
        
        Args:
            df: DataFrame with data
            output_file: Optional output file path
            
        Returns:
            Path to saved Excel file
        """
        if output_file is None:
            timestamp = int(time.time())
            output_file = os.path.join(self.output_dir, f"export_{timestamp}.xlsx")
            
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # Save to Excel
            df.to_excel(output_file, index=False)
            logger.info(f"Data exported to Excel: {output_file}")
            
            return output_file
        except Exception as e:
            logger.error(f"Error exporting to Excel: {e}")
            return ""
            
    def export_to_csv(self, df: pd.DataFrame, output_file: str = None) -> str:
        """
        Export query results to CSV.
        
        Args:
            df: DataFrame with data
            output_file: Optional output file path
            
        Returns:
            Path to saved CSV file
        """
        if output_file is None:
            timestamp = int(time.time())
            output_file = os.path.join(self.output_dir, f"export_{timestamp}.csv")
            
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # Save to CSV
            df.to_csv(output_file, index=False)
            logger.info(f"Data exported to CSV: {output_file}")
            
            return output_file
        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
            return ""