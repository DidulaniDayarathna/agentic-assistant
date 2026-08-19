"""
Agent construction.

Wraps LangGraph's `create_agent` with:
  - a persistent-per-thread memory checkpointer (multi-turn conversations)
  - a max-iteration guard against runaway tool-call loops
  - a single place (`build_agent`) that owns wiring, so app.py / tests /
    notebooks all construct the agent the same way.
"""
import logging

from langgraph.checkpoint.memory import MemorySaver

from langchain.agents import create_agent

from src.config import settings
from src.llm import get_llm
from src.tools import ALL_TOOLS

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a helpful, precise assistant with access to tools.

Guidelines:
- Use a tool whenever the answer depends on current, external, or
  computed information (weather, search, document lookup, math, word
  counts). Don't guess when a tool can give a grounded answer.
- Never call more tools than necessary. If you already have enough
  information to answer, answer directly.
- When you use search_documents, mention the source of the information.
- If a tool returns an error, explain the problem to the user plainly
  instead of retrying it silently more than once.
- Keep answers concise and directly address what was asked.
"""


def build_agent(checkpointer: MemorySaver | None = None):
    """Construct the agent graph.

    Args:
        checkpointer: optional memory backend. Defaults to an in-memory
            MemorySaver; pass a Postgres/Redis-backed checkpointer in
            production so conversation history survives restarts.
    """
    llm = get_llm()
    checkpointer = checkpointer or MemorySaver()

    agent = create_agent(
        model=llm,
        tools=ALL_TOOLS,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )
    logger.info(
        "Agent built with %d tools, max_iterations=%d",
        len(ALL_TOOLS),
        settings.max_agent_iterations,
    )
    return agent


def run_turn(agent, message: str, thread_id: str) -> str:
    """Run a single user turn against an existing agent and return the
    assistant's text reply. `thread_id` scopes conversation memory —
    use one per user/session.
    """
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": settings.max_agent_iterations * 2,
    }
    response = agent.invoke(
        {"messages": [{"role": "user", "content": message}]},
        config=config,
    )
    return response["messages"][-1].content
