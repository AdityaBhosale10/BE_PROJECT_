"""
API request/response schemas using Pydantic.

Defines the contract for API endpoints with validation and documentation.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """
    Request model for chat messages.
    
    Attributes:
        user: The user identifier sending the message
        message: The content of the user's message
    """
    user: str = "user"
    message: str
    # Backward-compatible: frontend doesn't send it yet.
    session_id: Optional[str] = None


class ChatbotResponse(BaseModel):
    """
    Response model for chatbot replies.
    
    Attributes:
        value: The chatbot's response text
        uid: Unique identifier for the response
    """
    value: str
    uid: str


class QueryFilters(BaseModel):
    """
    Structured constraints extracted from a user query.

    These filters are applied BEFORE vector retrieval.
    """

    category: Optional[str] = None
    brand: Optional[str] = None
    year: Optional[int] = Field(default=None, ge=1900, le=2100)

    price_min: Optional[float] = Field(default=None, ge=0)
    price_max: Optional[float] = Field(default=None, ge=0)

    # Free-form extracted specs/constraints (e.g. {"ram_gb": 16, "storage": "512gb ssd"})
    other_specs: Dict[str, Any] = Field(default_factory=dict)


class ProductRecord(BaseModel):
    """
    Canonical product record stored/retrieved by RAG.
    """

    id: str
    name: str
    price: Optional[float] = None
    category: Optional[str] = None
    year: Optional[int] = Field(default=None, ge=1900, le=2100)
    specs: Optional[str] = None

    image_url: Optional[str] = None
    source: Optional[str] = None  # e.g. "kaggle", "apify"
    source_url: Optional[str] = None
    updated_at: Optional[datetime] = None


class ProductRecommendation(BaseModel):
    """
    UI-facing recommendation object.
    """

    name: str
    price: Optional[float] = None
    image: Optional[str] = None
    url: Optional[str] = None
    source: Optional[str] = None
    reason: Optional[str] = None


class ChatRequest(BaseModel):
    """
    New chat request shape (compatible with existing ChatMessage).
    """

    message: str
    session_id: Optional[str] = None


class RegenerateRequest(BaseModel):
    """Request to regenerate a response for a previous user message."""
    session_id: Optional[str] = None
    # Index of the target user turn (0-based from oldest). If omitted, -1 = last user turn.
    index: Optional[int] = Field(default=-1)


class EditRequest(BaseModel):
    """Request to edit a previous user message and regenerate response."""
    session_id: str
    index: int = Field(..., description="0-based index of the user turn to edit (from oldest)")
    new_message: str = Field(..., description="Replacement user message content")


class ChatResponse(BaseModel):
    """
    New chat response shape (single JSON, non-streaming).
    """

    answer: str
    products: List[ProductRecommendation] = Field(default_factory=list)
    filters: Optional[QueryFilters] = None
    data_source: Optional[Literal["static", "dynamic", "hybrid"]] = None
    debug: Optional[Dict[str, Any]] = None
