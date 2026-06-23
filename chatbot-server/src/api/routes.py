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
from src.models.schemas import RegenerateRequest, EditRequest
from fastapi import HTTPException
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


@router.post("/chat/regenerate")
async def regenerate_message(
    req: RegenerateRequest,
    service: IChatService = Depends(get_chat_service),
    memory: Optional[SessionMemoryStore] = Depends(get_memory),
):
    """
    Regenerate response for a previous user message identified by index.
    If index is negative or omitted, regenerate the last user message.
    """
    if not req.session_id:
        return {"error": "session_id is required to regenerate"}

    # Fetch history and locate user turns
    if memory is None:
        return {"error": "memory disabled or unavailable"}

    turns = memory.get_history(req.session_id, limit=200)
    # collect indices of user turns
    user_indices = [i for i, t in enumerate(turns) if t.role == "user"]
    if not user_indices:
        return {"error": "no user turns found in session"}

    # default to last user turn
    idx = req.index if req.index is not None else -1
    if idx < 0:
        # map negative to actual user turn index
        target_user_pos = user_indices[idx]
    else:
        if idx >= len(user_indices):
            return {"error": "user turn index out of range"}
        target_user_pos = user_indices[idx]

    target_turn = turns[target_user_pos]

    results = []
    try:
        async for chunk in service.stream_chat(target_turn.content, session_id=req.session_id):
            results.append(chunk)
    except TypeError:
        async for chunk in service.stream_chat(target_turn.content):
            results.append(chunk)

    return {"results": results}


@router.post("/chat/edit")
async def edit_message(
    req: EditRequest,
    service: IChatService = Depends(get_chat_service),
    memory: Optional[SessionMemoryStore] = Depends(get_memory),
):
    """
    Edit a previous user message stored in memory and regenerate the assistant response.
    """
    if not req.session_id:
        return {"error": "session_id is required to edit"}

    if memory is None:
        return {"error": "memory disabled or unavailable"}

    # Replace the target turn
    from src.adapters.memory.redis_memory import ChatTurn

    success = memory.edit_turn(req.session_id, req.index, ChatTurn(role="user", content=req.new_message))
    if not success:
        return {"error": "failed to edit the specified turn"}

    # Regenerate response for the edited message
    results = []
    try:
        async for chunk in service.stream_chat(req.new_message, session_id=req.session_id):
            results.append(chunk)
    except TypeError:
        async for chunk in service.stream_chat(req.new_message):
            results.append(chunk)

    return {"results": results}


@router.get("/model")
async def get_model(service: IChatService = Depends(get_chat_service)):
    """Return the active LLM model name for the chat service."""
    try:
        return {"model": getattr(service, "llm_model", None)}
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to read model")


@router.post("/model")
async def set_model(payload: dict, service: IChatService = Depends(get_chat_service)):
    """Set the active LLM model for the chat service at runtime.

    Payload: {"model": "model-name"}
    """
    model = payload.get("model")
    if not model or not isinstance(model, str):
        raise HTTPException(status_code=400, detail="model is required")
    try:
        setattr(service, "llm_model", model)
        return {"model": model}
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to set model")


@router.post("/chat/delete")
async def delete_turn(payload: dict, memory: Optional[SessionMemoryStore] = Depends(get_memory)):
    """
    Delete a turn by index from session memory.

    Payload: {"session_id": "...", "index": -1}
    """
    session_id = payload.get("session_id")
    index = payload.get("index", -1)
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id required")
    if memory is None:
        raise HTTPException(status_code=400, detail="memory disabled")
    try:
        ok = memory.delete_turn(session_id, int(index))
        if not ok:
            raise HTTPException(status_code=404, detail="turn not found")
        return {"deleted": True}
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="delete failed")


@router.post("/chat/reaction")
async def set_reaction(payload: dict, memory: Optional[SessionMemoryStore] = Depends(get_memory)):
    """Set or clear a reaction for a specific turn index in a session.

    Payload: {"session_id": "...", "index": 3, "reaction": "up"}
    To clear reaction send `"reaction": null`.
    """
    session_id = payload.get("session_id")
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id required")
    if memory is None:
        raise HTTPException(status_code=400, detail="memory disabled")
    try:
        index = int(payload.get("index", -1))
        reaction = payload.get("reaction")
        # Accept None to clear
        memory.set_reaction(session_id, index, reaction)
        return {"ok": True}
    except Exception:
        raise HTTPException(status_code=500, detail="failed to set reaction")


@router.get("/chat/reactions")
async def get_reactions(session_id: str = Query(...), memory: Optional[SessionMemoryStore] = Depends(get_memory)):
    if memory is None:
        return {}
    return memory.get_reactions(session_id)


@router.get("/health")
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        Status response
    """
    return {"status": "healthy"}
