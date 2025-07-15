from typing import Dict, List, Any, Optional, Sequence, Annotated
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.graph import Graph, StateGraph, END
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import pandas as pd
import seaborn as sns
import plotly.express as px
from io import BytesIO
import base64
import uuid
import operator
import io
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

class InsightOutput(BaseModel):
    insight_type: str = Field(description="Type of insight (e.g., trend, outlier, correlation, pattern)")
    description: str = Field(description="Detailed description of the insight")
    importance: str = Field(description="High/Medium/Low importance of the insight")
    python_code: str = Field(description="Python code to validate/reproduce this insight")
    visualization_type: Optional[str] = Field(description="Type of visualization if applicable")

class VisualizationOutput(BaseModel):
    title: str = Field(description="Title for the visualization")
    description: str = Field(description="Description of what the visualization shows")
    python_code: str = Field(description="Python code to generate the visualization")
    plot_type: str = Field(description="Type of plot (e.g., line, bar, scatter, heatmap)")
    additional_insights: List[str] = Field(description="Additional insights from the visualization")

class AgentState(Dict):
    messages: Annotated[Sequence[Any], operator.add]

class AnalystAgent:
    def __init__(self):
        self._sessions = {}  # Store session data in memory
        self._llm = ChatOpenAI(
            model="gpt-4-turbo-preview",
            temperature=0.7,
            max_tokens=2000
        )
    
    def _get_session_data(self, session_id: str) -> pd.DataFrame:
        """Get data for a session or raise error if not found"""
        if session_id not in self._sessions:
            raise ValueError(f"No data found for session {session_id}")
        return self._sessions[session_id]

    def _detect_column_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """Detect and categorize column types in the dataset"""
        column_types = {}
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                # Check if it might be a currency column
                if df[col].astype(str).str.contains(r'[$₹€£¥]').any():
                    column_types[col] = 'currency'
                else:
                    column_types[col] = 'numeric'
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                column_types[col] = 'datetime'
            elif pd.api.types.is_categorical_dtype(df[col]) or df[col].nunique() / len(df) < 0.5:
                column_types[col] = 'categorical'
            else:
                column_types[col] = 'text'
        return column_types

    def _execute_insight_code(self, code: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Execute the insight validation code and return results"""
        try:
            # Create namespace with necessary imports
            local_vars = {
                "df": df,
                "np": np,
                "pd": pd,
                "result": {}
            }
            
            # Execute the code
            exec(code, globals(), local_vars)
            
            # Return the results
            return local_vars.get("result", {})
        except Exception as e:
            print(f"Error executing insight code: {str(e)}")
            return {"error": str(e)}

    def _execute_visualization_code(self, code: str, df: pd.DataFrame) -> str:
        """Execute the generated visualization code and return base64 encoded image"""
        try:
            # Create a new figure
            plt.figure(figsize=(10, 6))
            
            # Add df to local namespace
            local_vars = {
                "df": df,
                "plt": plt,
                "sns": sns,
                "np": np,
                "px": px
            }
            
            # Execute the code
            exec(code, globals(), local_vars)
            
            # Save plot to bytes buffer
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight')
            plt.close()
            
            # Encode to base64
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            return f"data:image/png;base64,{image_base64}"
        except Exception as e:
            print(f"Error executing visualization code: {str(e)}")
            plt.close()
            raise

    @tool
    async def process_file(self, file_bytes: bytes, filename: str, session_id: str) -> Dict[str, Any]:
        """Process an uploaded file and store it in the session"""
        try:
            # Convert bytes to DataFrame based on file type
            if filename.endswith('.csv'):
                df = pd.read_csv(io.BytesIO(file_bytes))
            elif filename.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(io.BytesIO(file_bytes))
            else:
                raise ValueError("Unsupported file format. Please upload CSV or Excel files.")
            
            # Store the DataFrame in the session
            self._sessions[session_id] = df
            
            # Detect column types
            column_types = self._detect_column_types(df)
            
            # Basic analysis
            analysis = {
                "row_count": len(df),
                "column_count": len(df.columns),
                "columns": list(df.columns),
                "column_types": column_types,
                "sample_data": df.head(5).to_dict('records')
            }
            
            return analysis
            
        except Exception as e:
            raise Exception(f"Error processing file: {str(e)}")

    @tool
    async def analyze_data(self, session_id: str, analysis_type: str, columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """Perform specific analysis on the data"""
        try:
            df = self._get_session_data(session_id)
            
            if analysis_type == "summary":
                return df.describe().to_dict()
            elif analysis_type == "correlation":
                if not columns:
                    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
                    return df[numeric_cols].corr().to_dict()
                return df[columns].corr().to_dict()
            elif analysis_type == "missing":
                return df.isnull().sum().to_dict()
            elif analysis_type == "unique_counts":
                if not columns:
                    return {col: df[col].nunique() for col in df.columns}
                return {col: df[col].nunique() for col in columns}
            else:
                raise ValueError(f"Unsupported analysis type: {analysis_type}")
                
        except Exception as e:
            raise Exception(f"Error analyzing data: {str(e)}")

    @tool
    async def create_visualization(self, session_id: str, viz_type: str, columns: List[str]) -> str:
        """Create a visualization using LLM-generated code"""
        try:
            df = self._get_session_data(session_id)
            column_types = self._detect_column_types(df)
            
            # Prepare data context for LLM
            data_context = {
                "shape": df.shape,
                "column_types": column_types,
                "focus_columns": columns,
                "preferred_viz_type": viz_type,
                "numeric_summary": df.describe().to_dict() if not df.empty else {},
                "sample_data": df.head(5).to_dict()
            }
            
            # Create prompt for visualization generation
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a data visualization expert. Generate Python code to create insightful visualizations.
                Use matplotlib, seaborn, or plotly. The code should:
                1. Be complete and executable
                2. Include proper styling and labels
                3. Handle data cleaning if needed
                4. Create a clear and informative visualization
                5. Include meaningful title and description
                
                The code should create a single figure that best represents the data and relationships."""),
                ("user", """Data Context: {data_context}
                
                Generate visualization code that best represents this data.
                Focus on the specified columns and visualization type.
                Make sure to handle any necessary data preprocessing.""")
            ])
            
            # Parse output with Pydantic
            parser = PydanticOutputParser(pydantic_object=VisualizationOutput)
            chain = prompt | self._llm | parser
            
            # Generate and execute visualization code
            result = chain.invoke({"data_context": json.dumps(data_context, default=str)})
            
            # Execute the generated code
            image_base64 = self._execute_visualization_code(result.python_code, df)
            
            # Return both the image and additional insights
            return {
                "image": image_base64,
                "title": result.title,
                "description": result.description,
                "insights": result.additional_insights
            }
            
        except Exception as e:
            raise Exception(f"Error creating visualization: {str(e)}")

    @tool
    async def get_insights(self, session_id: str) -> List[Dict[str, Any]]:
        """Generate insights using LLM by analyzing the data"""
        try:
            df = self._get_session_data(session_id)
            column_types = self._detect_column_types(df)
            
            # Prepare data summary for LLM
            data_summary = {
                "shape": df.shape,
                "column_types": column_types,
                "sample_data": df.head(5).to_dict(),
                "numeric_summary": df.describe().to_dict() if not df.empty else {},
                "missing_values": df.isnull().sum().to_dict()
            }
            
            # Create prompt for insight generation
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a data analysis expert. Generate meaningful insights from the data provided.
                Use Python code to validate your insights. Focus on:
                1. Statistical patterns and anomalies
                2. Trends and correlations
                3. Data quality issues
                4. Unusual patterns or outliers
                5. Business-relevant insights
                
                Format each insight using the provided Pydantic model structure.
                Make sure the Python code is executable and validates the insight."""),
                ("user", "Here's the data summary: {data_summary}\n\nGenerate an insight with Python code to validate it.")
            ])
            
            # Parse output with Pydantic
            parser = PydanticOutputParser(pydantic_object=InsightOutput)
            chain = prompt | self._llm | parser
            
            # Generate multiple insights
            insights = []
            for _ in range(3):  # Generate 3 different insights
                result = chain.invoke({"data_summary": json.dumps(data_summary, default=str)})
                
                # Execute the validation code
                validation_result = self._execute_insight_code(result.python_code, df)
                
                insights.append({
                    "type": result.insight_type,
                    "description": result.description,
                    "importance": result.importance,
                    "validation_result": validation_result,
                    "visualization_type": result.visualization_type
                })
            
            return insights
            
        except Exception as e:
            raise Exception(f"Error generating insights: {str(e)}")

    @tool
    async def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """Get the current status of an analysis session"""
        try:
            if session_id not in self._sessions:
                return {
                    "status": "not_found",
                    "message": "No data found for this session"
                }
            
            df = self._sessions[session_id]
            column_types = self._detect_column_types(df)
            
            return {
                "status": "active",
                "data_info": {
                    "rows": len(df),
                    "columns": len(df.columns),
                    "column_types": column_types,
                    "memory_usage": df.memory_usage(deep=True).sum() / (1024 * 1024),  # MB
                    "column_types": df.dtypes.to_dict()
                }
            }
            
        except Exception as e:
            raise Exception(f"Error getting session status: {str(e)}") 