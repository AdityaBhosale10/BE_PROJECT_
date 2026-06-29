"""
Vector search API endpoints.

Provides endpoints for semantic search and vector database operations.
"""
import logging
from typing import Optional

from fastapi import APIRouter, Depends, Request, Query
from pydantic import BaseModel
from fastapi import UploadFile, File, Form, HTTPException

from src.interfaces import IVectorStoreService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/vector", tags=["vector-search"])


class SearchQuery(BaseModel):
    """Vector search query request model."""
    query: str
    top_k: int = 5
    department: Optional[str] = None
    region: Optional[str] = None


class SearchResult(BaseModel):
    """Search result model."""
    similarityScore: float
    document: dict


def get_vector_store(request: Request) -> IVectorStoreService:
    """
    Dependency provider for IVectorStoreService.
    
    Args:
        request: FastAPI request object
        
    Returns:
        IVectorStoreService instance from app state
    """
    return request.app.state.vector_store


def get_multimodal_service(request: Request):
    return getattr(request.app.state, "multimodal_service", None)


@router.post("/search")
async def semantic_search(
    search_query: SearchQuery,
    vector_store: IVectorStoreService = Depends(get_vector_store),
):
    """
    Perform semantic search on vector database.
    
    Args:
        search_query: Search parameters
        vector_store: Injected IVectorStoreService
        
    Returns:
        List of matching documents with similarity scores
    """
    try:
        results = vector_store.search_similar(
            query=search_query.query,
            top_k=search_query.top_k,
            department=search_query.department,
            region=search_query.region,
        )
        return {"results": results, "count": len(results)}
    except Exception as e:
        logger.error(f"Error during search: {e}")
        return {"error": str(e), "results": []}


@router.get("/search/simple")
async def simple_search(
    query: str = Query(..., description="Search query"),
    top_k: int = Query(5, ge=1, le=100),
    vector_store: IVectorStoreService = Depends(get_vector_store),
):
    """
    Simple GET endpoint for semantic search.
    
    Args:
        query: Search query string
        top_k: Number of results
        vector_store: Injected IVectorStoreService
        
    Returns:
        Search results
    """
    results = vector_store.search_similar(query=query, top_k=top_k)
    return {"results": results, "count": len(results)}


@router.get("/status")
async def vector_db_status(
    vector_store: IVectorStoreService = Depends(get_vector_store),
):
    """
    Get vector database status.
    
    Returns:
        Status information
    """
    return {
        "status": "ready",
        "database": "MongoDB Atlas",
        "has_indexes": True,
    }


@router.post("/multimodal/search")
async def multimodal_search(
    query: str = Form(""),
    image: UploadFile | None = File(None),
    image_url: str | None = Form(None),
    top_k: int = Form(5),
    multimodal_service = Depends(get_multimodal_service),
):
    """Search using text + optional image. Accepts multipart file upload or image_url."""
    if multimodal_service is None:
        raise HTTPException(status_code=501, detail="Multimodal search not configured")

    image_bytes = None
    if image is not None:
        image_bytes = await image.read()

    try:
        results = multimodal_service.search(
            query or "",
            top_k=top_k,
            image_query_url=image_url,
            image_query_bytes=image_bytes,
        )
        return {"results": results, "count": len(results)}
    except Exception as e:
        logger.error("Error during multimodal search: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/images/upload")
async def upload_image(
    title: str = Form(""),
    image: UploadFile = File(...),
    image_url: str | None = Form(None),
    multimodal_service = Depends(get_multimodal_service),
):
    """Upload and index a single image + metadata into multimodal index."""
    if multimodal_service is None:
        raise HTTPException(status_code=501, detail="Multimodal service not configured")

    content = await image.read()
    product = {"title": title or image.filename, "name": title or image.filename, "image_url": image_url or ""}
    try:
        idx = multimodal_service.add_image(product=product, image_bytes=content, image_url=image_url)
        return {"ok": True, "id": idx}
    except Exception as e:
        logger.error("Failed to upload image: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
