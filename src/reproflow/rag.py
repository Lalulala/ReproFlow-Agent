from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter
from pathlib import Path

from .models import RetrievedItem


TOKEN_RE = re.compile(r"[A-Za-z0-9_\-]+|[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def split_markdown(path: Path, text: str) -> list[RetrievedItem]:
    chunks: list[RetrievedItem] = []
    section = "Document"
    buffer: list[str] = []

    def flush() -> None:
        if not buffer:
            return
        content = "\n".join(buffer).strip()
        if not content:
            return
        digest = hashlib.sha256(content.encode()).hexdigest()
        chunks.append(
            RetrievedItem(
                source_id=digest[:16],
                title=path.stem,
                content=content,
                path=str(path),
                section=section,
                tags=[part.lower() for part in path.stem.replace("_", "-").split("-")],
                content_hash=digest,
            )
        )
        buffer.clear()

    for line in text.splitlines():
        if line.startswith("#"):
            flush()
            section = line.lstrip("# ").strip() or section
        else:
            buffer.append(line)
            if sum(len(item) for item in buffer) > 1400:
                flush()
    flush()
    return chunks


class LexicalKnowledgeBase:
    """Offline RAG fallback with explicit sources and deterministic ranking."""

    def __init__(self, project_root: str | Path):
        self.project_root = Path(project_root).resolve()
        self.index_path = self.project_root / ".reproflow" / "knowledge_index.json"
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

    def index(self, knowledge_dir: str | Path | None = None) -> int:
        source = Path(knowledge_dir or self.project_root / "knowledge").resolve()
        items: list[RetrievedItem] = []
        if source.exists():
            for path in sorted(source.rglob("*")):
                if path.suffix.lower() in {".md", ".txt"}:
                    items.extend(split_markdown(path, path.read_text(encoding="utf-8")))
                elif path.suffix.lower() == ".pdf":
                    try:
                        from pypdf import PdfReader

                        reader = PdfReader(path)
                        for page_number, page in enumerate(reader.pages, 1):
                            text = page.extract_text() or ""
                            digest = hashlib.sha256(text.encode()).hexdigest()
                            items.append(
                                RetrievedItem(
                                    source_id=digest[:16],
                                    title=path.stem,
                                    content=text[:3000],
                                    path=str(path),
                                    page=page_number,
                                    content_hash=digest,
                                )
                            )
                    except ImportError:
                        continue
        self.index_path.write_text(
            json.dumps([item.model_dump(mode="json") for item in items], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return len(items)

    def _items(self) -> list[RetrievedItem]:
        if not self.index_path.exists():
            self.index()
        return [
            RetrievedItem.model_validate(item)
            for item in json.loads(self.index_path.read_text(encoding="utf-8"))
        ]

    def search(self, query: str, limit: int = 5) -> list[RetrievedItem]:
        items = self._items()
        query_counts = Counter(tokenize(query))
        if not query_counts:
            return []
        scored: list[tuple[float, RetrievedItem]] = []
        for item in items:
            document_counts = Counter(tokenize(item.title + " " + (item.section or "") + " " + item.content))
            overlap = set(query_counts) & set(document_counts)
            numerator = sum(query_counts[token] * document_counts[token] for token in overlap)
            denominator = math.sqrt(sum(v * v for v in query_counts.values())) * math.sqrt(
                sum(v * v for v in document_counts.values())
            )
            score = numerator / denominator if denominator else 0.0
            item.score = score
            scored.append((score, item))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [item for score, item in scored[:limit] if score > 0]


class ChromaKnowledgeBase(LexicalKnowledgeBase):
    """Chroma + local all-MiniLM-L6-v2 index; falls back to lexical RAG when unavailable."""

    collection_name = "reproflow_knowledge"

    def index(self, knowledge_dir: str | Path | None = None) -> int:
        count = super().index(knowledge_dir)
        import chromadb
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

        client = chromadb.PersistentClient(path=str(self.project_root / "knowledge" / ".chroma"))
        try:
            client.delete_collection(self.collection_name)
        except Exception:
            pass
        collection = client.create_collection(
            self.collection_name, embedding_function=DefaultEmbeddingFunction()
        )
        items = self._items()
        if items:
            collection.add(
                ids=[item.source_id for item in items],
                documents=[item.content for item in items],
                metadatas=[
                    {
                        "title": item.title,
                        "path": item.path,
                        "section": item.section or "",
                        "page": item.page or 0,
                        "content_hash": item.content_hash,
                        "source_type": item.source_type,
                    }
                    for item in items
                ],
            )
        return count

    def search(self, query: str, limit: int = 5) -> list[RetrievedItem]:
        import chromadb
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

        client = chromadb.PersistentClient(path=str(self.project_root / "knowledge" / ".chroma"))
        try:
            collection = client.get_collection(
                self.collection_name, embedding_function=DefaultEmbeddingFunction()
            )
        except Exception:
            self.index()
            collection = client.get_collection(
                self.collection_name, embedding_function=DefaultEmbeddingFunction()
            )
        result = collection.query(query_texts=[query], n_results=limit)
        items: list[RetrievedItem] = []
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        ids = result.get("ids", [[]])[0]
        for source_id, content, metadata, distance in zip(ids, documents, metadatas, distances):
            items.append(
                RetrievedItem(
                    source_id=source_id,
                    title=metadata["title"],
                    content=content,
                    path=metadata["path"],
                    section=metadata.get("section") or None,
                    page=metadata.get("page") or None,
                    source_type=metadata.get("source_type", "document"),
                    content_hash=metadata["content_hash"],
                    score=1.0 / (1.0 + float(distance)),
                )
            )
        return items


def get_knowledge_base(project_root: str | Path):
    backend = os.getenv("REPROFLOW_RAG_BACKEND", "auto").lower()
    if backend == "lexical":
        return LexicalKnowledgeBase(project_root)
    if backend in {"auto", "chroma"}:
        try:
            import chromadb  # noqa: F401

            return ChromaKnowledgeBase(project_root)
        except ImportError:
            if backend == "chroma":
                raise
    return LexicalKnowledgeBase(project_root)

