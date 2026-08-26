
from src.tools.calculator import calculator
from src.tools.rag import search_documents
from src.tools.search import web_search
from src.tools.weather import get_weather
from src.tools.word_count import word_count

ALL_TOOLS = [
    web_search,
    get_weather,
    word_count,
    calculator,
    search_documents,
]

__all__ = ["ALL_TOOLS"]
