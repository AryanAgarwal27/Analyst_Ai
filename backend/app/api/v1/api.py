from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from typing import Optional
from app.api.v1.models import (
    AnalysisResponse,
    ChatRequest,
    ChatResponse,
    InitResponse
)
from agents.analyst_agent import AnalystAgent
from utils.session import get_session_agent, create_session, get_or_create_session

api_router = APIRouter()

@api_router.post("/init", response_model=InitResponse)
async def initialize_session(api_key: str = Form(...)):
    """Initialize a new analysis session with OpenAI API key"""
    try:
        agent = AnalystAgent(api_key)
        session_id = await agent.initialize()
        await create_session(session_id, agent)
        return {"session_id": session_id, "message": "Session initialized successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@api_router.post("/upload/{session_id}", response_model=AnalysisResponse)
async def upload_file(
    session_id: str,
    file: UploadFile = File(...),
    agent: AnalystAgent = Depends(get_session_agent)
):
    """Upload and analyze a data file"""
    try:
        file_bytes = await file.read()
        file_type = "csv" if file.filename.endswith(".csv") else "xlsx"
        
        result = await agent.process_file(file_bytes=file_bytes, file_type=file_type)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@api_router.post("/chat/{session_id}", response_model=ChatResponse)
async def chat(
    session_id: str,
    request: ChatRequest,
    agent: AnalystAgent = Depends(get_session_agent)
):
    """Process a chat message"""
    try:
        result = await agent.process_message(request.message)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@api_router.delete("/session/{session_id}")
async def end_session(
    session_id: str,
    agent: AnalystAgent = Depends(get_session_agent)
):
    """End an analysis session"""
    try:
        await agent.cleanup()
        return {"message": "Session ended successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) 