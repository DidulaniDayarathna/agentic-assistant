from langchain.tools import tool


@tool
def word_count(text: str) -> int:
    """Count the number of words in a given piece of text.

    Args:
        text: The text to count words in.
    """
    return len(text.split())
