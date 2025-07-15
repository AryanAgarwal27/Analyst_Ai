from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from typing import Dict, List, Optional
import sys
from pathlib import Path
import uuid

# Add the parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from agents.analyst_agent import AnalystAgent

router = APIRouter()

@router.post("/init")
async def initialize_session() -> Dict[str, str]:
    """
    Initialize a new analysis session
    """
    try:
        session_id = str(uuid.uuid4())
        return {"session_id": session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error initializing session: {str(e)}")

@router.post("/upload/{session_id}")
async def upload_file(session_id: str, file: UploadFile = File(...)) -> Dict:
    """
    Upload and analyze a data file
    """
    try:
        # Read file content
        file_content = await file.read()
        
        # Initialize analyst agent
        agent = AnalystAgent()
        
        # Process file using the agent's process_file tool
        result = await agent.process_file(
            file_bytes=file_content,
            filename=file.filename
        )
        
        return {"detail": "File processed successfully", "result": result}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")

@router.post("/analyze/{session_id}")
async def analyze_data(
    session_id: str, 
    analysis_type: str,
    columns: Optional[List[str]] = None
) -> Dict:
    """
    Perform specific analysis on the uploaded data
    """
    try:
        agent = AnalystAgent()
        result = await agent.analyze_data(
            session_id=session_id,
            analysis_type=analysis_type,
            columns=columns
        )
        return {"detail": "Analysis completed", "result": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error during analysis: {str(e)}")

@router.post("/visualize/{session_id}")
async def create_visualization(
    session_id: str,
    viz_type: str,
    columns: List[str]
) -> Dict:
    """
    Create visualization for the specified columns
    """
    try:
        agent = AnalystAgent()
        result = await agent.create_visualization(
            session_id=session_id,
            viz_type=viz_type,
            columns=columns
        )
        return {"detail": "Visualization created", "result": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error creating visualization: {str(e)}")

@router.get("/insights/{session_id}")
async def get_insights(session_id: str) -> Dict:
    """
    Get AI-generated insights about the data
    """
    try:
        agent = AnalystAgent()
        insights = await agent.get_insights(session_id=session_id)
        return {"insights": insights}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error getting insights: {str(e)}")

@router.get("/status/{session_id}")
async def get_session_status(session_id: str) -> Dict:
    """
    Get the current status of an analysis session
    """
    try:
        agent = AnalystAgent()
        status = await agent.get_session_status(session_id=session_id)
        return {"status": status}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error getting session status: {str(e)}") 