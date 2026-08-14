from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class KnowledgeDocument:
    id: str
    category: str
    title: str
    text: str
    tags: list[str]
    source: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> "KnowledgeDocument":
        return cls(
            id=data["id"],
            category=data["category"],
            title=data.get("title", ""),
            text=data["text"],
            tags=data.get("tags", []),
            source=data.get("source", ""),
        )

    def embedding_text(self) -> str:
        tags = ", ".join(self.tags)
        return f"{self.title}\nCategory: {self.category}\nTags: {tags}\n{self.text}"

    def prompt_text(self) -> str:
        return f"[{self.title}] {self.text}"


def load_documents(knowledge_dir: str | Path) -> list[KnowledgeDocument]:
    root = Path(knowledge_dir)
    docs: list[KnowledgeDocument] = []

    for path in sorted(root.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                docs.append(KnowledgeDocument.from_dict(json.loads(line)))

    if not docs:
        raise RuntimeError(f"No knowledge documents found in {root}")

    return docs


class KnowledgeRetriever:
    def __init__(
        self,
        knowledge_dir: str | Path,
        index_dir: str | Path,
        embedding_model: str,
    ) -> None:
        self.knowledge_dir = Path(knowledge_dir)
        self.index_dir = Path(index_dir)
        self.embedding_model_name = embedding_model
        self.model = SentenceTransformer(embedding_model)

        self.docs: list[KnowledgeDocument] = []
        self.index: faiss.Index | None = None

    @property
    def index_path(self) -> Path:
        return self.index_dir / "knowledge.faiss"

    @property
    def metadata_path(self) -> Path:
        return self.index_dir / "metadata.json"

    def build(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.docs = load_documents(self.knowledge_dir)

        texts = [d.embedding_text() for d in self.docs]
        embeddings = self.model.encode(
            texts,
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).astype(np.float32)

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        faiss.write_index(index, str(self.index_path))

        with self.metadata_path.open("w", encoding="utf-8") as f:
            json.dump(
                [d.__dict__ for d in self.docs],
                f,
                ensure_ascii=False,
                indent=2,
            )

        self.index = index

    def load(self) -> None:
        if not self.index_path.exists() or not self.metadata_path.exists():
            self.build()
            return

        self.index = faiss.read_index(str(self.index_path))

        with self.metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)

        self.docs = [KnowledgeDocument.from_dict(x) for x in metadata]

    def query(
        self,
        query: str,
        category: str | None = None,
        top_k: int = 5,
    ) -> list[KnowledgeDocument]:
        if self.index is None:
            self.load()

        assert self.index is not None

        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
        ).astype(np.float32)

        search_k = min(len(self.docs), max(top_k * 8, 20))
        _, indices = self.index.search(query_embedding, search_k)

        results: list[KnowledgeDocument] = []
        for index in indices[0]:
            if index < 0:
                continue
            doc = self.docs[int(index)]
            if category is not None and doc.category != category:
                continue
            results.append(doc)
            if len(results) >= top_k:
                break

        return results


def format_documents(documents: Iterable[KnowledgeDocument]) -> str:
    docs = list(documents)
    if not docs:
        return "No relevant knowledge was retrieved."
    return "\n".join(f"{i + 1}. {d.prompt_text()}" for i, d in enumerate(docs))