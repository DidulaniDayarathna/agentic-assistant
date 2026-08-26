import logging

from langchain.tools import tool

from src.config import settings

logger = logging.getLogger(__name__)


def _search_tavily(query: str, max_results: int = 5) -> str:
    from langchain_tavily import TavilySearch

    client = TavilySearch(api_key=settings.tavily_api_key, max_results=max_results)
    results = client.invoke(query)
    return str(results)


def _search_duckduckgo(query: str, max_results: int = 5) -> str:
    from ddgs import DDGS

    with DDGS() as ddgs:
        hits = list(ddgs.text(query, max_results=max_results))

    if not hits:
        return f"No search results found for {query!r}."

    lines = [f"- {h.get('title', '')}: {h.get('body', '')} ({h.get('href', '')})" for h in hits]
    return "\n".join(lines)


@tool
def web_search(query: str) -> str:
    """Search the web for current information, news, or facts not in
    the model's training data.

    Args:
        query: The search query.
    """
    try:
        if settings.tavily_api_key:
            return _search_tavily(query)
        return _search_duckduckgo(query)
    except Exception as exc:  # noqa: BLE001 - tool boundary, must not crash the agent
        logger.warning("Search tool failed for query %r: %s", query, exc)
        return f"Search failed: {exc}"
