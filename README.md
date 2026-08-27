---
title: Agentic Assistant
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: gradio
app_file: gradio_app.py
pinned: false
---

# Agentic Assistant

🔗 **[Try the live demo](https://huggingface.co/spaces/Didulani/agentic-assistant)**

A tool using LLM agent (LangGraph `create_agent`) with weather, web
search, document RAG, calculator, and word-count tools, served over a
FastAPI HTTP API and a Gradio chat UI, with per-session memory.

Started as a Colab notebook exploring LangChain agents; rebuilt as a
small production shaped project tests, config management, structured
tool outputs, error handling, Docker, and CI.

## Architecture

```
User → FastAPI /chat → LangGraph agent (create_agent)
                              │
              ┌───────────────┼───────────────┬────────────┬──────────────┐
              ▼               ▼                ▼            ▼              ▼
         web_search      get_weather     search_documents  calculator   word_count
        (Tavily/DDG)   (OpenWeatherMap)   (FAISS + local     (safe AST    (utility)
                                             embeddings)      eval)

Conversation state persisted per thread_id via a LangGraph checkpointer.
```


- **`src/config.py`** — all env vars / secrets loaded once via
  `pydantic-settings`. Nothing else touches `os.environ` directly.
- **`src/llm.py`** — provider factory (OpenAI or a HF-router-hosted
  model today; swap in Anthropic/local vLLM by editing one function).
- **`src/tools/`** — one file per tool, registered in
  `src/tools/__init__.py`. Adding a tool means: write it, import it,
  add it to `ALL_TOOLS`.
- **`src/agent.py`** — builds the LangGraph agent with a system prompt,
  a memory checkpointer, and an iteration cap.
- **`app.py`** — FastAPI wrapper (`/chat`, `/health`).
- **`gradio_app.py`** — Gradio chat UI, used for the live Hugging Face
  Spaces demo above.

## Why these design choices

- **`create_agent` (LangGraph) over a hand-rolled ReAct loop** — gets
  battle-tested tool-calling, streaming, and interrupt/checkpoint
  support for free, and is the direction LangChain itself is
  standardizing on.
- **Qwen2.5-72B-Instruct over DeepSeek-R1 for the agent's LLM.**
  Reasoning models like R1 narrate their chain-of-thought as plain
  text instead of reliably emitting structured tool calls, which
  broke tool execution in testing (the model would describe calling
  `get_weather` without actually invoking it). Non-reasoning instruct
  models with dedicated tool-calling support are the more reliable
  choice for an agent that must actually execute tools, not just
  reason about them.
- **Structured tool outputs where it matters** (`WeatherReport` is a
  Pydantic model) rather than returning a free-text API dump — makes
  the tool unit-testable independent of the LLM, and gives the model a
  consistent shape to summarize.
- **No `eval()` in the calculator tool.** Expressions are parsed to an
  AST and walked against a whitelist of operators/functions. A raw
  `eval(user_input)` tool is a code-execution vulnerability the moment
  an agent is exposed to untrusted input — worth calling out
  explicitly since it's an easy trap in agent demos.
- **Search has a zero-key fallback** (DuckDuckGo) with an upgrade path
  to Tavily if a key is present, so the project runs out of the box.
- **Errors are caught at the tool boundary and returned as strings**,
  not raised — an uncaught exception inside a tool call crashes the
  entire agent turn; a returned error string lets the LLM explain the
  failure to the user and decide whether to retry.

## Setup

```bash
git clone <this-repo>
cd agentic-assistant
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in at least one LLM key + OPENWEATHERMAP_API_KEY
```

Run the API:

```bash
uvicorn app:app --reload
# → http://localhost:8000/docs
```

Or with Docker (local testing only — the live demo runs on Gradio, see above):

```bash
docker compose up --build
```

## Example usage

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the weather in Kandy, and how many words are in your answer?"}'
```

```json
{
  "reply": "The weather in Kandy, LK is broken clouds, 27.7°C... (18 words)",
  "thread_id": "b3f2e1a4-..."
}
```

Send a follow-up in the same conversation by reusing `thread_id`:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What about tomorrow?", "thread_id": "b3f2e1a4-..."}'
```

Document search (RAG) example:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the battery life of the R1 vacuum?"}'
```

## Tools

| Tool | Purpose | Notes |
|---|---|---|
| `web_search` | Current events / general web info | Tavily if configured, else DuckDuckGo |
| `get_weather` | Current weather for a city | OpenWeatherMap, retried with backoff |
| `search_documents` | RAG over `data/sample_docs/` | Local FAISS index, sentence-transformers embeddings |
| `calculator` | Arithmetic / math functions | AST-based, no `eval()` |
| `word_count` | Word count of given text | Simple utility, demonstrates the `@tool` decorator pattern |

## Testing

```bash
pytest --cov=src --cov-report=term-missing
```

- `tests/test_tools.py` — unit tests per tool, external calls mocked.
- `tests/test_agent.py` — agent-level tool-routing test using a
  scripted fake chat model, so it runs with no API key and no network.

CI (`.github/workflows/ci.yml`) runs lint + tests on every push/PR.

## Known limitations / next steps

- Memory checkpointer is in-process (`MemorySaver`); swap for a
  Postgres/Redis-backed checkpointer before running more than one
  server replica.
- No streaming responses yet — `/chat` returns the full completion.
- No auth on the API — add an API key or OAuth layer before exposing
  publicly.
- LangSmith tracing is wired into config but not yet enabled by
  default; flip it on for tool-call observability.
- HF Spaces' free tier only offers ZeroGPU hardware, which requires a
  dummy `@spaces.GPU`-decorated function to be present at startup
  (see `gradio_app.py`) even though this project's inference runs
  remotely via the HF router, not on Space hardware. Expect a slow
  cold start after periods of inactivity on the free tier.

## License

MIT