"""
Memory integration (Part 1, Step 6): lets InsightForge retain and reuse
context from earlier turns in the conversation so follow-up questions like
"and how does that compare to last quarter?" resolve correctly.

`ConversationMemory` is a minimal, dependency-free implementation (a bounded
deque of (question, answer) pairs). When LangChain is installed and an LLM
is configured, `to_langchain_memory()` can bridge this into a
`ConversationBufferWindowMemory` for use inside an LCEL/LangGraph chain —
but the core app never *requires* LangChain just to remember prior turns.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class ConversationMemory:
    """Bounded conversational memory: keeps the last `window` Q&A turns."""

    window: int = 6
    _turns: deque[tuple[str, str]] = field(default_factory=deque)

    def __post_init__(self) -> None:
        self._turns = deque(maxlen=self.window)

    def add_turn(self, question: str, answer: str) -> None:
        self._turns.append((question, answer))

    def get_turns(self) -> list[tuple[str, str]]:
        return list(self._turns)

    def clear(self) -> None:
        self._turns.clear()

    def to_langchain_memory(self):
        """Bridge to a LangChain memory object, if the package is installed.

        Only used by `src.rag.chain` when building a fully LLM-backed chain;
        the rest of the app relies purely on `get_turns()`.
        """
        from langchain.memory import ConversationBufferWindowMemory  # noqa: WPS433

        lc_memory = ConversationBufferWindowMemory(k=self.window, return_messages=True)
        for question, answer in self._turns:
            lc_memory.save_context({"input": question}, {"output": answer})
        return lc_memory
