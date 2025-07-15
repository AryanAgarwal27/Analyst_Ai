from fastapi import Depends, HTTPException
from typing import Dict
from agents.analyst_agent import AnalystAgent

# In-memory session storage
# In production, you might want to use Redis or another persistent store
_sessions: Dict[str, AnalystAgent] = {}

async def get_session_agent(session_id: str) -> AnalystAgent:
    """Get the analyst agent for a given session"""
    if session_id not in _sessions:
        raise HTTPException(
            status_code=404,
            detail="Session not found. Please initialize a new session."
        )
    return _sessions[session_id]

async def create_session(session_id: str, agent: AnalystAgent):
    """Create a new session with an analyst agent"""
    _sessions[session_id] = agent

async def end_session(session_id: str):
    """End a session and cleanup resources"""
    if session_id in _sessions:
        await _sessions[session_id].cleanup()
        del _sessions[session_id]

async def get_or_create_session(session_id: str, api_key: str) -> AnalystAgent:
    """Get an existing session or create a new one"""
    if session_id not in _sessions:
        agent = AnalystAgent(api_key)
        await create_session(session_id, agent)
    return _sessions[session_id] 