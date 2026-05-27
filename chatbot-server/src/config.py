"""
Application configuration and dependency management.

Consolidates configuration, environment setup, and dependency injection
for the FastAPI application.
"""
import os
import logging
from typing import Optional

from dotenv import find_dotenv, load_dotenv
from langchain_core.prompts import ChatPromptTemplate

from src.interfaces import (
    LLMClientInterface,
    HybridSearchInterface,
    ProductSourceSearchInterface,
    ModelProviderInterface,
    IVectorStoreService,
    IChatService,
)
from src.adapters.llm import GroqProvider
from src.adapters.search import TavilyHybridSearchProvider, TavilySourceSearchProvider
from src.adapters.vector import MongoDBVectorProvider
from src.adapters.model_provider import CustomModelProvider, default_model_path
from src.repositories import VectorDBRepository
from src.repositories.faiss_repository import FaissRepository
from src.services import ChatService, PromptMessage
from src.services.vector_store import VectorStoreService

logger = logging.getLogger(__name__)


class Config:
    """
    Application configuration management.
    
    Loads environment variables and provides configuration access
    for the entire application.
    """
    
    def __init__(self):
        """Load configuration from environment."""
        load_dotenv(find_dotenv())
        
    @property
    def groq_api_key(self) -> str:
        """Get Groq API key from environment."""
        return os.getenv("GROQ_API_KEY", "")
    
    @property
    def tavily_api_key(self) -> str:
        """Get Tavily API key from environment."""
        return os.getenv("TAVILY_API_KEY", "")
    
    @property
    def cohere_api_key(self) -> str:
        """Get Cohere API key from environment."""
        # Backward compatible: support both COHERE_API_KEY and CO_API_KEY
        return os.getenv("COHERE_API_KEY", "") or os.getenv("CO_API_KEY", "")
    
    @property
    def mongo_username(self) -> str:
        """Get MongoDB username from environment."""
        return os.getenv("MONGO_DB_USERNAME", "")
    
    @property
    def mongo_password(self) -> str:
        """Get MongoDB password from environment."""
        return os.getenv("MONGO_DB_PASSWORD", "")
    
    @property
    def mongo_cluster(self) -> str:
        """Get MongoDB cluster from environment."""
        return os.getenv("MONGO_DB_CLUSTER", "")
    
    @property
    def mongo_database(self) -> str:
        """Get MongoDB database name from environment."""
        return os.getenv("MONGO_DB_NAME", "productgpt")

    @property
    def faiss_dir(self) -> str:
        """
        Directory for FAISS artifacts (index + metadata).
        """
        return os.getenv("FAISS_DIR", os.path.join(os.getcwd(), "data", "faiss"))

    @property
    def embeddings_provider(self) -> str:
        return os.getenv("EMBEDDINGS_PROVIDER", "sentence_transformers")

    @property
    def embeddings_model(self) -> str:
        return os.getenv("EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    @property
    def redis_url(self) -> str:
        return os.getenv("REDIS_URL", "redis://localhost:6379/0")

    @property
    def use_mongo_vector(self) -> bool:
        """
        Keep MongoDB optional; default to enabled for backward compatibility.

        Set `USE_MONGO_VECTOR=false` to fully disable MongoDB/Tavily wiring.
        """
        raw = os.getenv("USE_MONGO_VECTOR")
        if raw is not None:
            return raw.lower() in {"1", "true", "yes"}
        return True


class DependencyContainer:
    """
    Dependency injection container.
    
    Manages creation and caching of service dependencies following
    the Factory pattern with singleton instances.
    """
    
    def __init__(self):
        """Initialize the dependency container."""
        self.config = Config()
        self._services = {}
        self._initialize_adapters()
    
    def _initialize_adapters(self) -> None:
        """Initialize external service adapters."""
        logger.info("Initializing adapters...")
        
        # Model provider
        self._model_provider = CustomModelProvider(default_model_path())
        
        # LLM adapter
        self._llm_client = GroqProvider(api_key=self.config.groq_api_key)
        
        # Embeddings service (local by default)
        from src.services.embeddings import EmbeddingsService
        emb_kwargs = {"provider_type": self.config.embeddings_provider}
        if self.config.embeddings_provider == "sentence_transformers":
            emb_kwargs["model_name"] = self.config.embeddings_model
        if self.config.cohere_api_key:
            emb_kwargs["api_key"] = self.config.cohere_api_key

        # Keep unit tests (FakeEmbeddings) compatible by falling back
        # when the patched EmbeddingsService doesn't accept extra kwargs.
        try:
            self._embeddings_service = EmbeddingsService(**emb_kwargs)
        except TypeError:
            self._embeddings_service = EmbeddingsService(provider_type=self.config.embeddings_provider)

        # FAISS vector store (primary)
        self._faiss_repo = FaissRepository(self.config.faiss_dir)

        # Session memory (Redis with in-process fallback)
        from src.adapters.memory.session_memory import SessionMemoryStore

        self._memory = SessionMemoryStore(redis_url=self.config.redis_url)

        # Optional MongoDB + Tavily (kept as fallback / legacy path)
        self._mongo_db = None
        self._vector_db_repo = None
        self._hybrid_search = None
        self._source_search = None

        if self.config.use_mongo_vector:
            try:
                mongo_provider = MongoDBVectorProvider(
                    username=self.config.mongo_username,
                    password=self.config.mongo_password,
                    cluster=self.config.mongo_cluster,
                    database=self.config.mongo_database,
                )
                self._mongo_db = mongo_provider.get_database()

                self._vector_db_repo = VectorDBRepository(self._mongo_db)
                logger.info("Initializing vector database indexes...")
                self._vector_db_repo.initialize()

                self._hybrid_search = TavilyHybridSearchProvider(
                    api_key=self.config.tavily_api_key,
                    mongo_db=self._mongo_db,
                    cohere_api_key=self.config.cohere_api_key,
                )
                self._source_search = TavilySourceSearchProvider(api_key=self.config.tavily_api_key)
            except Exception as e:
                logger.warning("Mongo/Tavily path disabled (init failed): %s", e)
                self._mongo_db = None
                self._vector_db_repo = None
                self._hybrid_search = None
                self._source_search = None

        # Vector store service (FAISS-first)
        try:
            self._vector_store_service = VectorStoreService(
                vector_db_repo=self._vector_db_repo,
                embeddings_service=self._embeddings_service,
                faiss_repo=self._faiss_repo,
            )
        except TypeError:
            # Unit tests may patch VectorStoreService with a fake that doesn't accept faiss_repo.
            self._vector_store_service = VectorStoreService(
                vector_db_repo=self._vector_db_repo,
                embeddings_service=self._embeddings_service,
            )
    
    @property
    def llm_client(self) -> LLMClientInterface:
        """Get LLM client adapter."""
        return self._llm_client
    
    @property
    def hybrid_search(self) -> HybridSearchInterface:
        """Get hybrid search adapter."""
        if self._hybrid_search is None:
            raise RuntimeError("Hybrid search not configured. Set USE_MONGO_VECTOR=true and provide Tavily+Cohere+Mongo env vars.")
        return self._hybrid_search
    
    @property
    def source_search(self) -> ProductSourceSearchInterface:
        """Get source search adapter."""
        if self._source_search is None:
            raise RuntimeError("Source search not configured. Set USE_MONGO_VECTOR=true and provide Tavily env vars.")
        return self._source_search
    
    @property
    def model_provider(self) -> ModelProviderInterface:
        """Get model provider."""
        return self._model_provider
    
    @property
    def memory(self):
        """Redis-backed session memory (optional)."""
        return self._memory

    @property
    def vector_store(self) -> IVectorStoreService:
        """Get vector store service."""
        return self._vector_store_service
    
    @property
    def vector_db_repo(self) -> VectorDBRepository:
        """Get vector database repository."""
        if self._vector_db_repo is None:
            raise RuntimeError("Mongo vector repository not configured.")
        return self._vector_db_repo
    
    def get_chat_service(self) -> IChatService:
        """
        Create ChatService with all dependencies.
        
        Returns:
            Configured ChatService instance implementing IChatService
        """
        template = ChatPromptTemplate.from_messages([
            ("system", PromptMessage.System_Message),
            ("human", PromptMessage.Human_Message),
            ("ai", PromptMessage.AI_Message),
        ])
        
        return ChatService(
            template=template,
            llm_client=self.llm_client,
            llm_model=self.model_provider.get_model_name(),
            hybrid_search=self._hybrid_search,
            source_search=self._source_search,
            memory=self._memory,
        )


def get_dependency_container() -> DependencyContainer:
    """
    Get or create the dependency injection container.
    
    Returns:
        Singleton DependencyContainer instance
    """
    if not hasattr(get_dependency_container, "_instance"):
        get_dependency_container._instance = DependencyContainer()
    return get_dependency_container._instance
