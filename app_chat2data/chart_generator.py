# Fixed chart_generator.py - Corrected Pydantic field handling

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import uuid
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ChartGenerator:
    def __init__(self, upload_folder='uploads'):
        self.upload_folder = upload_folder
        os.makedirs(upload_folder, exist_ok=True)

        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        sns.set_style("whitegrid")
        plt.style.use('default')

    def create_pie_chart(self, data, labels, title="Pie Chart", colors=None):
        try:
            fig, ax = plt.subplots(figsize=(10, 8))

            if colors is None:
                colors = plt.cm.Set3(np.linspace(0, 1, len(data)))

            wedges, texts, autotexts = ax.pie(
                data,
                labels=labels,
                autopct='%1.1f%%',
                startangle=90,
                colors=colors,
                explode=[0.05 if i == data.index(max(data)) else 0 for i in range(len(data))]
            )

            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(10)

            ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
            ax.legend(wedges, labels, title="Categories", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

            plt.tight_layout()

            filename = f"pie_chart_{uuid.uuid4().hex[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            filepath = os.path.join(self.upload_folder, filename)

            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()

            logger.info(f"Pie chart saved: {filepath}")
            return filename

        except Exception as e:
            logger.error(f"Error creating pie chart: {e}")
            plt.close()
            return None

    def create_bar_chart(self, x_data, y_data, title="Bar Chart", xlabel="X-axis", ylabel="Y-axis", horizontal=False):
        try:
            fig, ax = plt.subplots(figsize=(12, 8))

            if horizontal:
                bars = ax.barh(x_data, y_data, color=plt.cm.viridis(np.linspace(0, 1, len(y_data))))
                ax.set_xlabel(ylabel)
                ax.set_ylabel(xlabel)
            else:
                bars = ax.bar(x_data, y_data, color=plt.cm.viridis(np.linspace(0, 1, len(y_data))))
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)

            ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

            for bar in bars:
                if horizontal:
                    width = bar.get_width()
                    ax.text(width, bar.get_y() + bar.get_height() / 2,
                            f'{width:.1f}', ha='left', va='center', fontweight='bold')
                else:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2, height,
                            f'{height:.1f}', ha='center', va='bottom', fontweight='bold')

            if not horizontal and max(len(str(x)) for x in x_data) > 10:
                plt.xticks(rotation=45, ha='right')

            plt.tight_layout()

            chart_type = "horizontal_bar" if horizontal else "bar"
            filename = f"{chart_type}_chart_{uuid.uuid4().hex[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            filepath = os.path.join(self.upload_folder, filename)

            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()

            logger.info(f"Bar chart saved: {filepath}")
            return filename

        except Exception as e:
            logger.error(f"Error creating bar chart: {e}")
            plt.close()
            return None

    def create_line_chart(self, x_data, y_data, title="Line Chart", xlabel="X-axis", ylabel="Y-axis",
                          multiple_lines=None):
        try:
            fig, ax = plt.subplots(figsize=(12, 8))

            if multiple_lines and isinstance(y_data, dict):
                for i, (line_name, line_data) in enumerate(y_data.items()):
                    ax.plot(x_data, line_data, marker='o', linewidth=2, label=line_name)
                ax.legend()
            else:
                ax.plot(x_data, y_data, marker='o', linewidth=3, markersize=6, color='#2E86AB')

            ax.set_xlabel(xlabel, fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
            ax.grid(True, alpha=0.3)

            if len(x_data) > 10:
                plt.xticks(rotation=45, ha='right')

            plt.tight_layout()

            filename = f"line_chart_{uuid.uuid4().hex[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            filepath = os.path.join(self.upload_folder, filename)

            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()

            logger.info(f"Line chart saved: {filepath}")
            return filename

        except Exception as e:
            logger.error(f"Error creating line chart: {e}")
            plt.close()
            return None


# Fixed LangChain tool implementation
try:
    from langchain_core.tools import BaseTool
    from langchain.tools import Tool
except ImportError:
    from langchain.tools import BaseTool, Tool

from typing import Optional, Type, Any, Dict
from pydantic import BaseModel, Field, ConfigDict
import json


class ChartGeneratorInput(BaseModel):
    """Input schema for chart generator tool."""
    chart_type: str = Field(description="Type of chart: 'pie', 'bar', 'horizontal_bar', or 'line'")
    data_query: str = Field(description="SQL query to get data for the chart")
    title: str = Field(description="Chart title")
    xlabel: str = Field(default="", description="X-axis label")
    ylabel: str = Field(default="", description="Y-axis label")


class ChartGeneratorTool(BaseTool):
    """Tool for generating charts from database query results."""

    name: str = "generate_chart"
    description: str = """
    Use this tool to create visual charts from database data. Call this tool when users ask for:
    - pie charts, bar charts, line charts, or any visualizations
    - phrases like "show me a chart", "create a graph", "visualize the data"
    - "plot", "graph", "chart", "visualization"

    Required inputs:
    - chart_type: "pie", "bar", "horizontal_bar", or "line"
    - data_query: SQL SELECT query to get the data
    - title: descriptive title for the chart
    - xlabel: label for x-axis (optional)
    - ylabel: label for y-axis (optional)
    """
    args_schema: Type[BaseModel] = ChartGeneratorInput

    # Fix: Use model_config instead of class Config for Pydantic v2
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, db_manager_class, base_url="", **kwargs):
        super().__init__(**kwargs)
        # Store the class reference, not as a Pydantic field
        self._db_manager_class = db_manager_class
        self.chart_generator = ChartGenerator()
        self._base_url = base_url

    def _run(self, chart_type: str, data_query: str, title: str, xlabel: str = "", ylabel: str = "",
             **kwargs: Any) -> str:
        """Execute the chart generation."""
        try:
            logger.info(f"Chart generation requested: {chart_type} - {title}")
            logger.info(f"Query: {data_query}")

            # Get database connection using the stored class reference
            connection = self._db_manager_class.get_connection()
            if not connection:
                return "Error: Could not connect to database for chart generation"

            # Execute query
            results, columns = self._db_manager_class.execute_query(connection, data_query)
            connection.close()

            if not results:
                return "Error: Query returned no results for chart generation"

            if len(columns) < 2:
                return "Error: Chart requires at least 2 columns (labels and values)"

            # Convert data
            df = pd.DataFrame(results, columns=columns)
            logger.info(f"Data shape: {df.shape}")

            # Generate chart based on type
            filename = None

            if chart_type.lower() == "pie":
                labels = df.iloc[:, 0].astype(str).tolist()
                values = pd.to_numeric(df.iloc[:, 1], errors='coerce').tolist()

                if any(pd.isna(values)):
                    return "Error: Pie chart values must be numeric"

                # Limit to top 10 for readability
                if len(labels) > 10:
                    df_sorted = df.sort_values(df.columns[1], ascending=False).head(10)
                    labels = df_sorted.iloc[:, 0].astype(str).tolist()
                    values = df_sorted.iloc[:, 1].tolist()

                filename = self.chart_generator.create_pie_chart(values, labels, title)

            elif chart_type.lower() in ["bar", "horizontal_bar"]:
                x_data = df.iloc[:, 0].astype(str).tolist()
                y_data = pd.to_numeric(df.iloc[:, 1], errors='coerce').tolist()

                if any(pd.isna(y_data)):
                    return "Error: Bar chart values must be numeric"

                horizontal = chart_type.lower() == "horizontal_bar"
                filename = self.chart_generator.create_bar_chart(
                    x_data, y_data, title, xlabel or columns[0], ylabel or columns[1], horizontal
                )

            elif chart_type.lower() == "line":
                x_data = df.iloc[:, 0].tolist()

                if len(columns) == 2:
                    # Single line
                    y_data = pd.to_numeric(df.iloc[:, 1], errors='coerce').tolist()
                    if any(pd.isna(y_data)):
                        return "Error: Line chart values must be numeric"

                    filename = self.chart_generator.create_line_chart(
                        x_data, y_data, title, xlabel or columns[0], ylabel or columns[1]
                    )
                else:
                    # Multiple lines
                    y_data_dict = {}
                    for i in range(1, len(columns)):
                        y_values = pd.to_numeric(df.iloc[:, i], errors='coerce').tolist()
                        if not any(pd.isna(y_values)):
                            y_data_dict[columns[i]] = y_values

                    if not y_data_dict:
                        return "Error: No valid numeric columns found for line chart"

                    filename = self.chart_generator.create_line_chart(
                        x_data, y_data_dict, title, xlabel or columns[0], ylabel, list(y_data_dict.keys())
                    )

            else:
                return f"Error: Unsupported chart type '{chart_type}'. Use: pie, bar, horizontal_bar, or line"

            if filename:
                chart_url = f"{self._base_url}/uploads/{filename}"
                logger.info(f"Chart generated successfully: {chart_url}")
                return f"Chart generated successfully!\n\n{title}\n\nView your {chart_type} chart here: {chart_url}\n\nThe chart shows data from {len(df)} records with columns: {', '.join(columns)}"
            else:
                return "Error: Failed to generate chart file"

        except Exception as e:
            logger.error(f"Chart generation error: {e}")
            import traceback
            traceback.print_exc()
            return f"Error generating chart: {str(e)}"

    async def _arun(self, *args, **kwargs):
        """Async version - just calls the sync version."""
        return self._run(*args, **kwargs)


# Alternative: Create a simple function-based tool (recommended)
def create_chart_tool(db_manager_class, base_url=""):
    """Create a function-based chart tool as an alternative."""

    chart_generator = ChartGenerator()

    def generate_chart_function(input_string: str) -> str:
        """Generate a chart from database data."""
        try:
            # Parse input
            if isinstance(input_string, str):
                params = json.loads(input_string)
            else:
                params = input_string

            logger.info(f"Function-based chart generation: {params}")

            # Get database connection
            connection = db_manager_class.get_connection()
            if not connection:
                return "Error: Could not connect to database"

            # Execute query
            results, columns = db_manager_class.execute_query(connection, params.get("data_query"))
            connection.close()

            if not results or len(columns) < 2:
                return "Error: Query returned insufficient data for chart"

            # Convert data
            df = pd.DataFrame(results, columns=columns)

            chart_type = params.get("chart_type")
            title = params.get("title", "Chart")

            filename = None

            if chart_type == "pie":
                labels = df.iloc[:, 0].astype(str).tolist()
                values = pd.to_numeric(df.iloc[:, 1], errors='coerce').tolist()
                filename = chart_generator.create_pie_chart(values, labels, title)

            elif chart_type in ["bar", "horizontal_bar"]:
                x_data = df.iloc[:, 0].astype(str).tolist()
                y_data = pd.to_numeric(df.iloc[:, 1], errors='coerce').tolist()
                horizontal = chart_type == "horizontal_bar"
                filename = chart_generator.create_bar_chart(x_data, y_data, title, horizontal=horizontal)

            elif chart_type == "line":
                x_data = df.iloc[:, 0].tolist()
                y_data = pd.to_numeric(df.iloc[:, 1], errors='coerce').tolist()
                filename = chart_generator.create_line_chart(x_data, y_data, title)

            if filename:
                chart_url = f"{base_url}/uploads/{filename}"
                return f"Chart generated: {chart_url}"
            else:
                return "Failed to generate chart"

        except Exception as e:
            logger.error(f"Function chart error: {e}")
            return f"Error: {str(e)}"

    return Tool(
        name="generate_chart",
        description="Create charts from database data. Input: JSON with chart_type, data_query, title",
        func=generate_chart_function
    )