from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import json
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from datetime import datetime

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

def setup_llm():
    """Initialize the LLM with appropriate settings"""
    return ChatOpenAI(
        model="gpt-4-turbo-preview",
        temperature=0.7,
        max_tokens=2000
    )

def generate_insights(df: pd.DataFrame, column_types: Dict[str, str]) -> List[InsightOutput]:
    """Generate insights using LLM by analyzing the data"""
    
    llm = setup_llm()
    parser = PydanticOutputParser(pydantic_object=InsightOutput)
    
    # Prepare data summary for LLM
    data_summary = {
        "shape": df.shape,
        "column_types": column_types,
        "sample_data": df.head(5).to_dict(),
        "numeric_summary": df.describe().to_dict() if not df.empty else {},
        "missing_values": df.isnull().sum().to_dict()
    }
    
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
    
    chain = prompt | llm | parser
    
    # Generate multiple insights
    insights = []
    try:
        # Make multiple calls for different aspects of the data
        for _ in range(3):  # Generate 3 different insights
            result = chain.invoke({"data_summary": json.dumps(data_summary, default=str)})
            insights.append(result)
    except Exception as e:
        print(f"Error generating insights: {str(e)}")
    
    return insights

def generate_visualization(
    df: pd.DataFrame,
    column_types: Dict[str, str],
    focus_columns: Optional[List[str]] = None,
    viz_type: Optional[str] = None
) -> VisualizationOutput:
    """Generate visualization code using LLM based on data characteristics"""
    
    llm = setup_llm()
    parser = PydanticOutputParser(pydantic_object=VisualizationOutput)
    
    # Prepare data context
    data_context = {
        "shape": df.shape,
        "column_types": column_types,
        "focus_columns": focus_columns,
        "preferred_viz_type": viz_type,
        "numeric_summary": df.describe().to_dict() if not df.empty else {},
        "sample_data": df.head(5).to_dict()
    }
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a data visualization expert. Generate Python code to create insightful visualizations.
        Use matplotlib, seaborn, or plotly. The code should:
        1. Be complete and executable
        2. Include proper styling and labels
        3. Handle data cleaning if needed
        4. Return a base64 encoded image
        5. Include meaningful title and description
        
        Format the output using the provided Pydantic model structure."""),
        ("user", """Data Context: {data_context}
        
        Generate visualization code that best represents this data.
        If specific columns or visualization type are provided, focus on those.
        Otherwise, choose the most insightful visualization possible.""")
    ])
    
    chain = prompt | llm | parser
    
    try:
        result = chain.invoke({"data_context": json.dumps(data_context, default=str)})
        return result
    except Exception as e:
        print(f"Error generating visualization: {str(e)}")
        raise

def execute_visualization_code(code: str, df: pd.DataFrame) -> str:
    """Execute the generated visualization code and return base64 encoded image"""
    try:
        # Create a new figure
        plt.figure(figsize=(10, 6))
        
        # Add df to local namespace
        local_vars = {"df": df, "plt": plt, "sns": sns, "np": np}
        
        # Execute the code
        exec(code, globals(), local_vars)
        
        # Save plot to bytes buffer
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        plt.close()
        
        # Encode to base64
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return image_base64
    except Exception as e:
        print(f"Error executing visualization code: {str(e)}")
        plt.close()
        raise

def execute_insight_code(code: str, df: pd.DataFrame) -> Dict[str, Any]:
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