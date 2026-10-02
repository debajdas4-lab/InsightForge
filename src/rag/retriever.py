"""
Custom retriever for InsightForge (Part 1, Step 3 -> "Integration with RAG
System" -> "Develop a custom retriever to extract relevant statistics").

Two retrieval backends are provided behind one common interface:

1. `KeywordFactRetriever` — a dependency-free BM25-style lexical retriever.
   Runs anywhere (no API key, no vector DB, no network access required),
   always available as a deterministic fallback, and is what the
   evaluation suite uses so scores are reproducible in CI.

2. `EmbeddingFactRetriever` — wraps LangChain + Chroma + an embeddings model
   for semantic retrieval. Instantiated lazily (imports inside the
   constructor) so the rest of the application works even when
   `langchain`/`chromadb` aren't installed.

`get_retriever()` picks the best available backend based on `Settings`, and
**gracefully degrades to `KeywordFactRetriever` if the embedding backend
fails to initialize for any reason** — missing packages, a bad/missing API
key, or (very commonly on corporate networks) a firewall/proxy blocking the
one-time download `tiktoken` needs to fetch its encoding file. Without this
broad fallback, a purely environmental network issue would crash the whole
app instead of just disabling the semantic-search upgrade.
"""
from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass

from src.config import Settings, get_settings
from src.rag.fact_documents import FactDocument

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9%$.]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class RetrievedFact:
    doc_id: str
    text: str
    category: str
    score: float


class KeywordFactRetriever:
    """Lightweight, dependency-free BM25 retriever over fact documents.

    BM25 is a strong, well-understood ranking function for this use case:
    fact documents are short, mostly-unique, and keyword-rich (product
    names, regions, period labels) — exactly the regime where BM25 performs
    on par with dense embeddings but with zero external dependencies, zero
    network calls, and fully reproducible scores (ideal for the evaluation
    harness in `src/evaluation`).
    """

    k1 = 1.5
    b = 0.75

    def __init__(self, documents: list[FactDocument]):
        self.documents = documents
        self._tokenized = [_tokenize(d.text) for d in documents]
        self._doc_len = [len(t) for t in self._tokenized]
        self._avg_len = sum(self._doc_len) / max(len(self._doc_len), 1)
        self._df: Counter = Counter()
        for tokens in self._tokenized:
            for term in set(tokens):
                self._df[term] += 1
        self._n_docs = len(documents)

    def _idf(self, term: str) -> float:
        df = self._df.get(term, 0)
        return math.log(1 + (self._n_docs - df + 0.5) / (df + 0.5))

    def _score(self, query_tokens: list[str], doc_idx: int) -> float:
        tokens = self._tokenized[doc_idx]
        freqs = Counter(tokens)
        doc_len = self._doc_len[doc_idx] or 1
        score = 0.0
        for term in query_tokens:
            if term not in freqs:
                continue
            f = freqs[term]
            idf = self._idf(term)
            denom = f + self.k1 * (1 - self.b + self.b * doc_len / self._avg_len)
            score += idf * (f * (self.k1 + 1)) / denom
        return score

    def retrieve(self, query: str, top_k: int = 6, category: str | None = None) -> list[RetrievedFact]:
        query_tokens = _tokenize(query)
        candidates = range(len(self.documents))
        if category:
            candidates = [i for i in candidates if self.documents[i].category == category]

        scored = [(i, self._score(query_tokens, i)) for i in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)

        # Fallback: if lexical overlap is weak (e.g. vague query), surface the
        # highest-signal statistics/trend facts instead of returning nothing.
        if not scored or scored[0][1] <= 0:
            priority = ["statistics", "trend"]
            fallback = [i for i in range(len(self.documents)) if self.documents[i].category in priority]
            scored = [(i, 0.0) for i in fallback]

        results = []
        for idx, score in scored[:top_k]:
            d = self.documents[idx]
            results.append(RetrievedFact(doc_id=d.doc_id, text=d.text, category=d.category, score=round(score, 4)))
        return results


class EmbeddingFactRetriever:
    """Semantic retriever backed by LangChain + Chroma (optional upgrade path).

    Only instantiated when `settings.llm_available` is True and the
    `langchain-community` / `chromadb` packages are installed — imports are
    deliberately deferred to avoid a hard dependency for users running in
    offline/stats-only mode.

    Note: the underlying `tiktoken` tokenizer used by `langchain-openai`
    downloads its encoding file from `openaipublic.blob.core.windows.net`
    the first time it runs, on ANY machine, even if you already have an
    API key configured. On corporate networks with firewalls, proxies, or
    TLS-inspection appliances, this one-time download can be blocked,
    which raises a `ConnectionError`/`ConnectionAbortedError` here. See
    `get_retriever()` below for how this is handled.
    """

    def __init__(self, documents: list[FactDocument], settings: Settings):
        from langchain_community.vectorstores import Chroma  # noqa: WPS433
        from langchain_core.documents import Document  # noqa: WPS433

        if settings.insightforge_llm_provider == "openai":
            from langchain_openai import OpenAIEmbeddings  # noqa: WPS433
            embeddings = OpenAIEmbeddings(model=settings.openai_embedding_model, api_key=settings.openai_api_key)
        else:
            raise ValueError(
                "EmbeddingFactRetriever currently supports the 'openai' embedding "
                "provider only; use KeywordFactRetriever otherwise."
            )

        self.documents = documents
        lc_docs = [
            Document(page_content=d.text, metadata={"doc_id": d.doc_id, "category": d.category})
            for d in documents
        ]
        self._store = Chroma.from_documents(
            lc_docs, embeddings, persist_directory=str(settings.vectorstore_dir)
        )

    def retrieve(self, query: str, top_k: int = 6, category: str | None = None) -> list[RetrievedFact]:
        filter_ = {"category": category} if category else None
        results = self._store.similarity_search_with_relevance_scores(query, k=top_k, filter=filter_)
        out = []
        for doc, score in results:
            out.append(RetrievedFact(
                doc_id=doc.metadata.get("doc_id", "unknown"),
                text=doc.page_content,
                category=doc.metadata.get("category", "unknown"),
                score=round(float(score), 4),
            ))
        return out


def get_retriever(documents: list[FactDocument], settings: Settings | None = None):
    """Factory: returns the best retriever available for the current environment.

    Falls back gracefully to the dependency-free keyword retriever if the
    embedding backend can't be built for **any** reason — missing packages,
    a missing/invalid API key, or a network/proxy/firewall issue (e.g. the
    `tiktoken` encoding-file download being blocked, which is common on
    locked-down corporate networks). This keeps InsightForge fully runnable
    even when the "upgrade path" to semantic search is unavailable, rather
    than crashing the whole app on startup.
    """
    settings = settings or get_settings()
    if settings.llm_available and settings.insightforge_llm_provider == "openai":
        try:
            return EmbeddingFactRetriever(documents, settings)
        except Exception as exc:  # noqa: BLE001 - intentionally broad; see docstring
            logger.warning(
                "Falling back to the offline KeywordFactRetriever because the "
                "embedding-based retriever failed to initialize (%s: %s). This "
                "is commonly caused by a network/proxy/firewall blocking "
                "tiktoken's one-time encoding-file download, or an invalid "
                "OPENAI_API_KEY. The app will continue to work fully, just "
                "without semantic search.",
                type(exc).__name__, exc,
            )
    return KeywordFactRetriever(documents)
