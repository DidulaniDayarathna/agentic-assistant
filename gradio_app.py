import uuid

import gradio as gr

from src.agent import build_agent, run_turn

_agent = build_agent()


def chat_fn(message: str, history: list, thread_id: str):
    if not thread_id:
        thread_id = str(uuid.uuid4())
    reply = run_turn(_agent, message, thread_id=thread_id)
    return reply


with gr.Blocks(title="Agentic Assistant") as demo:
    gr.Markdown(
        "# Agentic Assistant\n"
        "A tool-using LLM agent with weather, web search, document RAG, "
        "and calculator tools. Try: *\"What's the weather in Kandy?\"* or "
        "*\"What is the R1 vacuum's battery life?\"*"
    )
    thread_state = gr.State(str(uuid.uuid4()))
    chat = gr.ChatInterface(
        fn=lambda message, history: chat_fn(message, history, thread_state.value),
        examples=[
            "What's the weather in Kandy?",
            "What is the battery life of the Acme R1 vacuum?",
            "Calculate sqrt(144) + 7 * 3",
            "How many words are in this sentence: the quick brown fox jumps",
        ],
    )

if __name__ == "__main__":
    demo.launch()