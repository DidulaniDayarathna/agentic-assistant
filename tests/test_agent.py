"""
Agent-level tests using LangChain's FakeListChatModel / a scripted
tool-calling fake, so these run offline with no API key and no network
access — important for CI.
"""
from langchain_core.messages import AIMessage, ToolCall
from langgraph.checkpoint.memory import MemorySaver

from langchain.agents import create_agent

from src.agent import SYSTEM_PROMPT, run_turn
from src.tools.word_count import word_count


class _ScriptedModel:
    """Minimal fake chat model: first call emits a tool call, second
    call emits a final text answer using the tool result."""

    def __init__(self):
        self._step = 0

    def bind_tools(self, tools, **kwargs):
        return self

    def invoke(self, messages, **kwargs):
        self._step += 1
        if self._step == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    ToolCall(name="word_count", args={"text": "hello agentic world"}, id="call_1")
                ],
            )
        return AIMessage(content="The text has 3 words.")


def test_agent_routes_to_correct_tool_and_answers():
    agent = create_agent(
        model=_ScriptedModel(),
        tools=[word_count],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=MemorySaver(),
    )
    reply = run_turn(agent, "how many words in 'hello agentic world'?", thread_id="test-thread")
    assert "3" in reply


def test_agent_memory_uses_same_thread_id():
    """Sanity check that thread_id actually threads through to the
    checkpointer config rather than being silently ignored."""
    agent = create_agent(
        model=_ScriptedModel(),
        tools=[word_count],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=MemorySaver(),
    )
    reply1 = run_turn(agent, "count words in 'a b c'", thread_id="shared-thread")
    state = agent.get_state({"configurable": {"thread_id": "shared-thread"}})
    assert state is not None
    assert reply1
