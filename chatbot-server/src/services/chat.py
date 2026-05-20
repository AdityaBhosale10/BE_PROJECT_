"""
Chat service for handling conversation logic.

Manages chat interactions, query relevance, and orchestrates the search pipeline.
"""
import json
import logging
import uuid
from typing import Optional

from src.interfaces import LLMClientInterface, HybridSearchInterface, ProductSourceSearchInterface, IChatService
from src.services.search_agent import SearchAgent
from src.services.prompt_messages import PromptMessage
from src.adapters.memory.redis_memory import RedisMemory, ChatTurn
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


class ChatService(IChatService):
    """
    Main chat service for conversation management.
    
    Handles user messages, determines relevance, and orchestrates
    the product search pipeline through SearchAgent.
    
    Implements IChatService contract for dependency injection.
    """

    def __init__(self,
                 template: Optional[ChatPromptTemplate] = None,
                 llm_client: LLMClientInterface = None,
                 llm_model: str = "",
                 hybrid_search: Optional[HybridSearchInterface] = None,
                 source_search: Optional[ProductSourceSearchInterface] = None,
                 memory: Optional[RedisMemory] = None):
        """
        Initialize chat service.

        Args:
            template: Prompt template for conversations
            llm_client: LLM adapter for text generation
            llm_model: Language model identifier
            hybrid_search: Hybrid search adapter
            source_search: Product source search adapter
        """
        self.template = template
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.hybrid_search = hybrid_search
        self.source_search = source_search
        self.memory = memory
        self.thread_id = str(uuid.uuid4())

    def query_llm(self, prompt: str, model: str) -> str:
        """
        Query the LLM with a prompt.

        Args:
            prompt: Prompt text to send
            model: Model identifier

        Returns:
            LLM response text

        Raises:
            ValueError: If prompt is not a string
        """
        if not isinstance(prompt, str):
            raise ValueError(f"Prompt must be a string, but got {type(prompt)}")
        return self.llm_client.generate(prompt=prompt, model=model)

    def is_query_relevant(self, query: str) -> bool:
        """
        Determine if query is relevant to product search.

        Args:
            query: User query to evaluate

        Returns:
            True if relevant, False otherwise
        """
        relevance_prompt = (
            f"This is prompt template: \"{self.template}\". Evaluate whether the following query is relevant to the prompt template: \"{query}\". Respond only one word 'relevant' or 'irrelevant'."
        )

        response = self.query_llm(prompt=relevance_prompt, model=self.llm_model)
        return response.lower().strip() == "relevant"

    async def stream_chat(self, query: str, session_id: Optional[str] = None):
        """
        Stream chat response as Server-Sent Events.

        Yields progress updates and final results as the pipeline executes.

        Args:
            query: User query to process
            
        Yields:
            JSON-encoded SSE events
        """
        # Load recent history from Redis (if configured)
        history_text = ""
        if session_id and self.memory:
            history = self.memory.get_history(session_id, limit=12)
            if history:
                history_text = "\n".join([f"{t.role}: {t.content}" for t in history])

        effective_query = query if not history_text else f"{history_text}\nuser: {query}"

        if not self.is_query_relevant(effective_query):
            yield json.dumps({
                "type": "result",
                "data": {"default": PromptMessage.Default_Message}
            })
            return

        NODE_MESSAGES = {
            "analyze_query": "Understanding your request...",
            "search_online_shop": "Searching for products...",
            "analyze_and_rank": "Ranking and analyzing results...",
            "search_product_source": "Finding product sources...",
        }

        async with AsyncSqliteSaver.from_conn_string(":memory:") as memory:
            agent = SearchAgent(
                llm_model=self.llm_model,
                llm_client=self.llm_client,
                hybrid_search=self.hybrid_search,
                source_search=self.source_search,
                checkpointer=memory
            )

            logger.info(f"Thread ID: {self.thread_id}")
            thread = {"configurable": {"thread_id": self.thread_id}}

            # Persist user turn early
            if session_id and self.memory:
                self.memory.append_turn(session_id, ChatTurn(role="user", content=query))

            async for chunk in agent.graph.astream({"user_query": effective_query}, thread, stream_mode="updates"):
                node_name = next(iter(chunk))
                state_update = chunk[node_name]

                if node_name in NODE_MESSAGES:
                    yield json.dumps({
                        "type": "progress",
                        "message": NODE_MESSAGES[node_name]
                    })

                if "result" in state_update:
                    if session_id and self.memory:
                        self.memory.append_turn(
                            session_id,
                            ChatTurn(role="assistant", content=json.dumps(state_update["result"])),
                        )
                    yield json.dumps({
                        "type": "result",
                        "data": state_update["result"]
                    })
