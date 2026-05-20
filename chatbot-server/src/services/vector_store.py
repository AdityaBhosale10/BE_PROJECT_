"""
Vector store service for semantic search operations.

Provides high-level interface for searching and storing vectors.
"""
import logging
from typing import List, Dict, Optional

from src.interfaces import IVectorStoreService
from src.repositories.vector_db_repository import VectorDBRepository
from src.repositories.faiss_repository import FaissRepository
from src.services.embeddings import EmbeddingsService
from src.services.faiss_vector_store import FaissVectorStoreService
from bson import ObjectId

logger = logging.getLogger(__name__)


class VectorStoreService(IVectorStoreService):
    """
    Service for vector search and storage operations.
    
    Combines embeddings generation with vector search to enable
    semantic similarity search on product data.
    
    Implements IVectorStoreService contract for dependency injection.
    """
    
    def __init__(
        self,
        vector_db_repo: Optional[VectorDBRepository],
        embeddings_service: EmbeddingsService,
        faiss_repo: Optional[FaissRepository] = None,
    ):
        """
        Initialize vector store service.
        
        Args:
            vector_db_repo: Vector database repository
            embeddings_service: Embeddings generation service
        """
        self.repo = vector_db_repo
        self.embeddings = embeddings_service
        self._faiss = FaissVectorStoreService(faiss_repo, embeddings_service) if faiss_repo else None
    
    def search_similar(self, 
                      query: str, 
                      top_k: int = 5,
                      department: Optional[str] = None,
                      region: Optional[str] = None) -> List[Dict]:
        """
        Search for semantically similar products.
        
        Args:
            query: Search query
            top_k: Number of results to return
            department: Filter by department
            region: Filter by region
            
        Returns:
            List of similar documents with scores
        """
        # Prefer FAISS if configured
        if self._faiss is not None:
            return self._faiss.search(query=query, top_k=top_k)

        if self.repo is None:
            logger.error("No vector store configured (FAISS repo missing and Mongo repo is None).")
            return []

        # Generate embedding for query
        query_embedding = self.embeddings.embed_text(query)
        
        # Build search query pipeline
        collection = self.repo.get_collection()
        
        pipeline = [
            {
                "$search": {
                    "cosmosSearch": {
                        "vector": query_embedding,
                        "k": top_k,
                    },
                    "returnBase64EncodedVectors": False,
                }
            },
            {
                "$project": {
                    "similarityScore": {"$meta": "searchScore"},
                    "document": "$$ROOT",
                }
            },
        ]
        
        # Add filters if provided
        if department or region:
            match_stage = {}
            if department:
                match_stage["department"] = department
            if region:
                match_stage["region"] = region
            pipeline.append({"$match": match_stage})
        
        try:
            results = list(collection.aggregate(pipeline))
            logger.info(f"Found {len(results)} similar items for query: {query}")
            return results
        except Exception as e:
            logger.error(f"Error searching vectors: {e}")
            return []
    
    def insert_vector(self, vector_data: Dict) -> str:
        """
        Insert a vector document into the store.
        
        Args:
            vector_data: Document with vector embedding
            
        Returns:
            Document ID
        """
        if self.repo is None:
            raise RuntimeError("insert_vector not supported without MongoDB repository")
        collection = self.repo.get_collection()
        result = collection.insert_one(vector_data)
        logger.info(f"Inserted vector document: {result.inserted_id}")
        return str(result.inserted_id)

    def delete_document(self, doc_id: str) -> bool:
        """
        Delete document from vector store.
        
        Args:
            doc_id: Document identifier
        
        Returns:
            True if deleted, False if not found
        """
        if self.repo is None:
            return False
        collection = self.repo.get_collection()
        result = collection.delete_one({"_id": ObjectId(doc_id)})
        return result.deleted_count > 0
