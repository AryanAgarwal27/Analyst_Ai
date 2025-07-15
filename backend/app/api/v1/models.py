from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class InitResponse(BaseModel):
    session_id: str
    message: str

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    thought_process: List[str]
    visualization: Optional[str] = None

class AnalysisResponse(BaseModel):
    message: str
    analysis: Dict[str, Any]
    summary: Optional[Dict[str, Any]] = None
    visualization: Optional[str] = None

class ErrorResponse(BaseModel):
    detail: str 