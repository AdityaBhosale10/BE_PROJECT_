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
from src.services.conversation_context import build_conversation_context
from src.adapters.memory.redis_memory import ChatTurn
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
                 memory=None):
        self.template = template
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.hybrid_search = hybrid_search
        self.source_search = source_search
        self.memory = memory
        self.thread_id = str(uuid.uuid4())

    def query_llm(self, prompt: str, model: str) -> str:
        if not isinstance(prompt, str):
            raise ValueError(f"Prompt must be a string, but got {type(prompt)}")
        return self.llm_client.generate(prompt=prompt, model=model)

    def is_query_relevant(self, query: str) -> bool:
        relevance_prompt = (
            "You are ProductGPT, a shopping and product-research assistant.\n"
            "Decide if the message is about products, shopping, recommendations, comparisons, "
            "or a follow-up about prior product advice in the same chat.\n"
            f"Message:\n{query[:1500]}\n"
            "Reply with exactly one word: relevant or irrelevant."
        )
        try:
            response = self.query_llm(prompt=relevance_prompt, model=self.llm_model)
            normalized = response.lower().strip()
            if "irrelevant" in normalized:
                return False
            return "relevant" in normalized
        except Exception as exc:
            logger.warning("Relevance check failed, allowing query: %s", exc)
            return True

    # Maximum recent turns to include in conversation context sent to LLM.
    # Keeping this low avoids 413 token-limit errors on Groq's free tier.
    MAX_HISTORY_TURNS = 4
    # Hard character cap for the full context string embedded in prompts.
    MAX_CONTEXT_CHARS = 1500

    async def stream_chat(self, query: str, session_id: Optional[str] = None):
        try:
            conversation_context = ""
            if session_id and self.memory:
                history = self.memory.get_history(session_id, limit=self.MAX_HISTORY_TURNS)
                conversation_context = build_conversation_context(history, exclude_last_user=True)
                # Hard-truncate to avoid exceeding token limits
                if len(conversation_context) > self.MAX_CONTEXT_CHARS:
                    conversation_context = conversation_context[-self.MAX_CONTEXT_CHARS:]

            if not self.is_query_relevant(query):
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

            thread_id = session_id or self.thread_id
            thread = {"configurable": {"thread_id": thread_id}}

            if session_id and self.memory:
                self.memory.append_turn(session_id, ChatTurn(role="user", content=query))

            graph_input = {
                "user_query": query,
                "conversation_context": conversation_context,
            }

            async with AsyncSqliteSaver.from_conn_string(":memory:") as checkpointer:
                agent = SearchAgent(
                    llm_model=self.llm_model,
                    llm_client=self.llm_client,
                    hybrid_search=self.hybrid_search,
                    source_search=self.source_search,
                    checkpointer=checkpointer,
                )

                logger.info("Chat thread_id=%s session_id=%s", thread_id, session_id)

                async for chunk in agent.graph.astream(graph_input, thread, stream_mode="updates"):
                    node_name = next(iter(chunk))
                    state_update = chunk[node_name]

                    if node_name in NODE_MESSAGES:
                        yield json.dumps({
                            "type": "progress",
                            "message": NODE_MESSAGES[node_name],
                        })

                    if "result" in state_update:
                        if session_id and self.memory:
                            self.memory.append_turn(
                                session_id,
                                ChatTurn(role="assistant", content=json.dumps(state_update["result"])),
                            )
                        # After storing assistant turn, optionally compress older history
                        try:
                            if session_id and self.memory and hasattr(self.memory, "max_turns"):
                                history = self.memory.get_history(session_id, limit=200)
                                # Only attempt summarization when history grows beyond configured max_turns
                                if len(history) > getattr(self.memory, "max_turns", 50):
                                    from src.services.conversation_context import summarize_history

                                    new_history, compressed = summarize_history(
                                        history, llm_client=self.llm_client, llm_model=self.llm_model, keep_recent=6
                                    )
                                    if compressed:
                                        # Replace stored history with compressed version
                                        try:
                                            self.memory.set_history(session_id, new_history)
                                        except Exception:
                                            logger.exception("Failed to set compressed history")
                        except Exception:
                            logger.exception("History summarization failed")
                        yield json.dumps({
                            "type": "result",
                            "data": state_update["result"],
                        })

        except Exception as exc:
            logger.exception("stream_chat failed: %s", exc)
            yield json.dumps({
                "type": "error",
                "message": "Something went wrong while processing your message. Please try again.",
            })
