from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from datetime import datetime
import io
from typing import Dict, List, Optional
from uuid import uuid4
import json
import re

app = FastAPI(title="Analyst AI")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store sessions in memory (replace with proper database in production)
sessions: Dict[str, dict] = {}

def detect_column_types(df: pd.DataFrame) -> Dict[str, str]:
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

def extract_numeric_value(value: any) -> float:
    """Extract numerical value from any type of input"""
    if pd.isna(value):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    try:
        # Remove currency symbols and common separators
        cleaned = re.sub(r'[^\d.-]', '', str(value))
        return float(cleaned)
    except:
        return 0.0

def calculate_column_statistics(df: pd.DataFrame, column: str) -> Dict:
    """Calculate comprehensive statistics for a column"""
    stats = {}
    try:
        if pd.api.types.is_numeric_dtype(df[column]):
            stats.update({
                "mean": float(df[column].mean()),
                "median": float(df[column].median()),
                "std": float(df[column].std()),
                "min": float(df[column].min()),
                "max": float(df[column].max()),
                "q1": float(df[column].quantile(0.25)),
                "q3": float(df[column].quantile(0.75))
            })
        elif pd.api.types.is_categorical_dtype(df[column]) or isinstance(df[column].dtype, pd.StringDtype):
            value_counts = df[column].value_counts()
            stats.update({
                "unique_values": int(df[column].nunique()),
                "most_common": value_counts.index[0] if not value_counts.empty else None,
                "most_common_count": int(value_counts.iloc[0]) if not value_counts.empty else 0
            })
        elif pd.api.types.is_datetime64_any_dtype(df[column]):
            stats.update({
                "min_date": df[column].min().isoformat(),
                "max_date": df[column].max().isoformat(),
                "date_range_days": (df[column].max() - df[column].min()).days
            })
    except Exception as e:
        stats["error"] = str(e)
    return stats

def clean_numeric_values(value):
    """Clean numeric values for JSON serialization"""
    if pd.isna(value) or pd.isnull(value):
        return None
    if isinstance(value, (np.int64, np.int32)):
        return int(value)
    if isinstance(value, (np.float64, np.float32)):
        if np.isinf(value) or np.isnan(value):
            return None
        return float(value)
    return value

def clean_dataframe_for_json(df: pd.DataFrame) -> pd.DataFrame:
    """Clean DataFrame to ensure JSON serialization"""
    df_clean = df.copy()
    for col in df_clean.columns:
        if df_clean[col].dtype.kind in 'fc':  # float or complex
            df_clean[col] = df_clean[col].apply(clean_numeric_values)
    return df_clean

@app.post("/api/v1/init")
async def initialize_session():
    """Initialize a new analysis session"""
    session_id = str(uuid4())
    sessions[session_id] = {
        "status": "initialized",
        "data": None,
        "analysis": None,
        "created_at": datetime.now().isoformat()
    }
    return {"session_id": session_id}

@app.post("/api/v1/upload/{session_id}")
async def upload_file(session_id: str, file: UploadFile = File(...)):
    """Upload and process a data file"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        content = await file.read()
        
        if file.filename.endswith('.csv'):
            # Try different encodings
            encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
            df = None
            last_error = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(io.BytesIO(content), encoding=encoding)
                    break
                except UnicodeDecodeError as e:
                    last_error = e
                    continue
            
            if df is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"Could not read CSV file with any supported encoding. Last error: {str(last_error)}"
                )
        elif file.filename.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(io.BytesIO(content))
        else:
            raise HTTPException(
                status_code=400, 
                detail="Unsupported file format. Please upload CSV or Excel files."
            )
        
        # Detect column types
        column_types = detect_column_types(df)
        
        # Store processed data in session
        sessions[session_id].update({
            "data": df.to_dict(),
            "column_types": column_types,
            "status": "data_loaded",
            "filename": file.filename,
            "upload_time": datetime.now().isoformat()
        })
        
        return {
            "message": "File processed successfully",
            "rows": len(df),
            "columns": list(df.columns),
            "column_types": column_types
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/v1/analyze/{session_id}")
async def analyze_data(session_id: str):
    """Analyze the uploaded dataset"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if sessions[session_id]["data"] is None:
        raise HTTPException(status_code=400, detail="No data available for analysis")
    
    try:
        # Create DataFrame and clean it for analysis
        df = pd.DataFrame(sessions[session_id]["data"])
        df = clean_dataframe_for_json(df)
        column_types = sessions[session_id]["column_types"]
        
        analysis = {
            "summary": {
                "total_columns": len(df.columns),
                "total_rows": len(df),
                "column_types": column_types,
                "missing_values": df.isnull().sum().to_dict()
            },
            "column_analysis": {},
            "correlations": {},
            "distributions": {},
            "insights": [],
            "data_quality": {}
        }
        
        # Analyze each column based on its type
        for col in df.columns:
            col_stats = calculate_column_statistics(df, col)
            # Clean numeric values in statistics
            analysis["column_analysis"][col] = {
                k: clean_numeric_values(v) if isinstance(v, (np.number, float)) else v
                for k, v in col_stats.items()
            }
        
        # Calculate correlations between numeric columns
        numeric_cols = [col for col, type_ in column_types.items() if type_ in ['numeric', 'currency']]
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            # Convert to dictionary and filter significant correlations
            correlations = {}
            for col1 in numeric_cols:
                for col2 in numeric_cols:
                    if col1 != col2:
                        corr_value = corr_matrix.loc[col1, col2]
                        if abs(corr_value) > 0.5:  # Only include significant correlations
                            correlations[f"{col1}__{col2}"] = clean_numeric_values(corr_value)
            analysis["correlations"] = correlations
        
        # Generate insights based on the data
        for col, stats in analysis["column_analysis"].items():
            col_type = column_types[col]
            
            if col_type in ['numeric', 'currency']:
                # Check for outliers using IQR method
                if all(k in stats for k in ['q1', 'q3']):
                    iqr = stats['q3'] - stats['q1']
                    lower_bound = stats['q1'] - (1.5 * iqr)
                    upper_bound = stats['q3'] + (1.5 * iqr)
                    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
                    if len(outliers) > 0:
                        analysis["insights"].append({
                            "type": "outliers",
                            "column": col,
                            "message": f"Found {len(outliers)} outliers in {col}",
                            "details": {
                                "outlier_count": len(outliers),
                                "percentage": round(len(outliers) / len(df) * 100, 2)
                            }
                        })
            
            elif col_type == 'categorical':
                # Check for imbalanced categories
                if 'unique_values' in stats:
                    value_counts = df[col].value_counts()
                    if len(value_counts) > 1:
                        imbalance_ratio = value_counts.iloc[0] / value_counts.iloc[-1]
                        if imbalance_ratio > 10:
                            analysis["insights"].append({
                                "type": "imbalanced_categories",
                                "column": col,
                                "message": f"Highly imbalanced categories in {col}",
                                "details": {
                                    "ratio": round(imbalance_ratio, 2),
                                    "dominant_category": value_counts.index[0],
                                    "dominant_category_percentage": round(value_counts.iloc[0] / len(df) * 100, 2)
                                }
                            })
        
        # Data quality checks
        analysis["data_quality"] = {
            "completeness": {
                col: round((1 - df[col].isnull().sum() / len(df)) * 100, 2)
                for col in df.columns
            },
            "unique_ratio": {
                col: round(df[col].nunique() / len(df) * 100, 2)
                for col in df.columns
            }
        }
        
        sessions[session_id]["analysis"] = analysis
        sessions[session_id]["status"] = "analysis_complete"
        
        return analysis
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/api/v1/chat/{session_id}")
async def chat_analysis(
    session_id: str,
    query: str = Body(..., embed=True),
    visualization_type: Optional[str] = Body(None, embed=True)
):
    """Interactive data analysis through chat"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if sessions[session_id]["data"] is None:
        raise HTTPException(status_code=400, detail="No data available for analysis")
    
    try:
        df = pd.DataFrame(sessions[session_id]["data"])
        column_types = sessions[session_id]["column_types"]
        
        response = {
            "answer": "",
            "visualization_data": None,
            "error": None,
            "statistics": None
        }
        
        # Process different types of queries
        query_lower = query.lower()
        
        # Find mentioned columns in the query
        requested_columns = []
        for col in df.columns:
            col_lower = col.lower()
            # Check for exact matches or partial matches
            if col_lower in query_lower or any(word in col_lower for word in query_lower.split()):
                requested_columns.append(col)
        
        # If no specific column is found but query mentions common column types
        if not requested_columns:
            if any(word in query_lower for word in ['status', 'state', 'category', 'type']):
                categorical_cols = [col for col, type_ in column_types.items() if type_ == 'categorical']
                if categorical_cols:
                    requested_columns = categorical_cols[:3]  # Limit to top 3
            elif any(word in query_lower for word in ['amount', 'price', 'cost', 'value', 'number']):
                numeric_cols = [col for col, type_ in column_types.items() if type_ in ['numeric', 'currency']]
                if numeric_cols:
                    requested_columns = numeric_cols[:3]
        
        # Handle distribution queries
        if any(word in query_lower for word in ["distribution", "breakdown", "split", "show", "chart"]):
            if not requested_columns:
                # If no specific column is mentioned, look for categorical columns
                categorical_cols = [col for col, type_ in column_types.items() 
                                 if type_ == 'categorical'][:3]
                requested_columns = categorical_cols
            
            if requested_columns:
                dist_data = {}
                for col in requested_columns:
                    if column_types[col] == 'categorical':
                        value_counts = df[col].value_counts()
                        dist_data[col] = {
                            "labels": value_counts.index.tolist(),
                            "values": value_counts.values.tolist()
                        }
                    else:
                        # For numeric columns, create bins
                        values = df[col].dropna()
                        hist, bins = np.histogram(values, bins='auto')
                        dist_data[col] = {
                            "labels": [f"{bins[i]:.2f}-{bins[i+1]:.2f}" for i in range(len(bins)-1)],
                            "values": hist.tolist()
                        }
                
                # Determine visualization type based on query and data
                viz_type = "bar"  # default
                if visualization_type:
                    viz_type = visualization_type
                elif "pie" in query_lower:
                    viz_type = "pie"
                elif len(requested_columns) > 1:
                    viz_type = "bar"
                
                response["visualization_data"] = {
                    "type": viz_type,
                    "data": dist_data
                }
                response["answer"] = f"Generated {viz_type} chart showing distribution for: {', '.join(requested_columns)}"
            else:
                response["error"] = "No suitable columns found for distribution analysis"
        
        # Statistical analysis queries
        elif any(word in query_lower for word in ["average", "mean", "statistics", "summary", "analyze"]):
            stats = {}
            target_cols = requested_columns if requested_columns else df.columns
            for col in target_cols:
                stats[col] = calculate_column_statistics(df, col)
            response["statistics"] = stats
            response["answer"] = f"Here are the detailed statistics for {', '.join(target_cols)}"
        
        # Trend analysis
        elif any(word in query_lower for word in ["trend", "pattern", "over time", "changes"]):
            trend_data = {}
            target_cols = requested_columns if requested_columns else [
                col for col, type_ in column_types.items() 
                if type_ in ['numeric', 'currency']
            ][:5]
            
            for col in target_cols:
                if column_types[col] in ['numeric', 'currency']:
                    values = df[col].dropna()
                    trend_data[col] = {
                        "values": values.tolist(),
                        "mean": float(values.mean()),
                        "trend": "increasing" if values.iloc[-1] > values.iloc[0] else "decreasing"
                    }
            
            if trend_data:
                response["visualization_data"] = {
                    "type": "line",
                    "data": trend_data
                }
                response["answer"] = f"Here's the trend analysis for {', '.join(target_cols)}"
            else:
                response["error"] = "No suitable numeric columns found for trend analysis"
        
        # Comparison queries
        elif any(word in query_lower for word in ["compare", "comparison", "versus", "vs"]):
            if len(requested_columns) >= 2:
                comparison_data = {}
                for col in requested_columns[:2]:  # Compare first two mentioned columns
                    if column_types[col] == 'categorical':
                        value_counts = df[col].value_counts()
                        comparison_data[col] = {
                            "labels": value_counts.index.tolist(),
                            "values": value_counts.values.tolist()
                        }
                    else:
                        values = df[col].dropna()
                        comparison_data[col] = {
                            "values": values.tolist(),
                            "mean": float(values.mean()),
                            "median": float(values.median())
                        }
                
                response["visualization_data"] = {
                    "type": "comparison",
                    "data": comparison_data
                }
                response["answer"] = f"Here's a comparison between {' and '.join(requested_columns[:2])}"
            else:
                response["error"] = "Please specify two columns to compare"
        
        else:
            # If no specific analysis type is detected, try to provide a meaningful visualization
            if requested_columns:
                col = requested_columns[0]
                if column_types[col] == 'categorical':
                    value_counts = df[col].value_counts()
                    response["visualization_data"] = {
                        "type": "bar",
                        "data": {
                            col: {
                                "labels": value_counts.index.tolist(),
                                "values": value_counts.values.tolist()
                            }
                        }
                    }
                    response["answer"] = f"Here's a visualization of {col} distribution"
                else:
                    response["statistics"] = {col: calculate_column_statistics(df, col)}
                    response["answer"] = f"Here are the statistics for {col}"
            else:
                response["error"] = "Please try asking about specific columns or analysis types (distribution, statistics, trends, etc.)"
        
        return response
        
    except Exception as e:
        print(f"Error in chat analysis: {str(e)}")  # Add logging
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/api/v1/status/{session_id}")
async def get_session_status(session_id: str):
    """Get the current status of an analysis session"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "status": sessions[session_id]["status"],
        "session_info": {
            "created_at": sessions[session_id].get("created_at"),
            "filename": sessions[session_id].get("filename"),
            "upload_time": sessions[session_id].get("upload_time"),
            "column_types": sessions[session_id].get("column_types")
        }
    } 