"""
Chat API endpoints.

Provides RESTful endpoints for chat interactions and streaming responses.
"""
import logging
import json

from typing import Optional

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import StreamingResponse

from src.models.schemas import ChatMessage
from src.interfaces import IChatService
from src.adapters.memory.session_memory import SessionMemoryStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["chat"])


def get_memory(request: Request) -> Optional[SessionMemoryStore]:
    return getattr(request.app.state, "memory", None)


def get_chat_service(request: Request) -> IChatService:
    """
    Dependency provider for ChatService.
    
    Args:
        request: FastAPI request object
        
    Returns:
        ChatService instance from app state
    """
    return request.app.state.chat_service


@router.post("/chat")
async def send_message(
    chat_message: ChatMessage,
    service: IChatService = Depends(get_chat_service),
):
    """
    Send a chat message.
    
    Args:
        chat_message: The user's message
        service: Injected ChatService
        
    Returns:
        JSON response with chat result
    """
    results = []
    try:
        async for chunk in service.stream_chat(chat_message.message, session_id=chat_message.session_id):
            results.append(chunk)
    except TypeError:
        # Backward-compatible with older IChatService implementations
        async for chunk in service.stream_chat(chat_message.message):
            results.append(chunk)
    return {"results": results}


@router.post("/chat/stream")
async def stream_message(
    chat_message: ChatMessage,
    service: IChatService = Depends(get_chat_service),
):
    """
    Stream chat response as Server-Sent Events.
    
    Args:
        chat_message: The user's message
        service: Injected ChatService
        
    Returns:
        StreamingResponse with SSE events
    """
    
    async def event_generator():
        try:
            try:
                async for chunk in service.stream_chat(chat_message.message, session_id=chat_message.session_id):
                    yield f"data: {chunk}\n\n"
            except TypeError:
                async for chunk in service.stream_chat(chat_message.message):
                    yield f"data: {chunk}\n\n"
        except Exception as e:
            logger.error(f"Error in stream_message: {e}")
            error_response = {"type": "error", "message": str(e)}
            yield f"data: {json.dumps(error_response)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/chat/history")
async def chat_history(
    session_id: str = Query(..., description="Conversation session id"),
    limit: int = Query(50, ge=1, le=200),
    memory: Optional[SessionMemoryStore] = Depends(get_memory),
):
    """
    Return stored chat turns for a session (Redis or in-process fallback).
    """
    if not session_id:
        return {"session_id": session_id, "messages": []}
    if memory is None:
        return {"session_id": session_id, "messages": [], "memory": "disabled"}

    turns = memory.get_history(session_id, limit=limit)
    return {
        "session_id": session_id,
        "messages": [{"role": t.role, "content": t.content} for t in turns],
        "memory": memory.backend,
    }


@router.get("/health")
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        Status response
    """
    return {"status": "healthy"}
