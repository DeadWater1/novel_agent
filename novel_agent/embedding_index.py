from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import AgentConfig
from .session_meta import utc_now_iso


@dataclass(slots=True)
class EmbeddingIndexItem:
    source_id: str
    source_kind: str
    source_path: str
    target: str
    text: str
    text_hash: str

    @classmethod
    def create(
        cls,
        *,
        source_id: str,
        source_kind: str,
        source_path: str,
        target: str,
        text: str,
    ) -> "EmbeddingIndexItem":
        clean_text = text.strip()
        return cls(
            source_id=source_id,
            source_kind=source_kind,
            source_path=source_path,
            target=target,
            text=clean_text,
            text_hash=_content_hash(clean_text),
        )

    def metadata(self) -> dict[str, str]:
        return {
            "source_id": self.source_id,
            "source_kind": self.source_kind,
            "source_path": self.source_path,
            "target": self.target,
            "text_hash": self.text_hash,
        }

    def key(self) -> tuple[str, str, str, str, str]:
        return (
            self.source_id,
            self.source_kind,
            self.source_path,
            self.target,
            self.text_hash,
        )


class EmbeddingIndexManager:
    def __init__(self, config: AgentConfig, embedding_backend: Any) -> None:
        self.config = config
        self.embedding_backend = embedding_backend
        self.root = Path(config.embedding_index_root)
        self.workspace_root = self.root / "workspace"
        self.compaction_root = self.root / "compactions"
        self.session_root = self.root / "sessions"

    def bootstrap(self) -> None:
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self.compaction_root.mkdir(parents=True, exist_ok=True)
        self.session_root.mkdir(parents=True, exist_ok=True)

    def workspace_shard_path(self) -> Path:
        return self.workspace_root / "workspace.pt"

    def compaction_shard_path(self, session_id: str) -> Path:
        return self.compaction_root / f"{session_id}.pt"

    def session_shard_path(self, session_id: str) -> Path:
        return self.session_root / f"{session_id}.pt"

    def build_query_embedding(self, query: str) -> Any:
        return self.embedding_backend.embed_query(query)

    def score_items(
        self,
        query: str,
        items: list[EmbeddingIndexItem],
        *,
        shard_path: Path | None = None,
        allow_write: bool = True,
    ) -> list[float]:
        query_embedding = self.build_query_embedding(query)
        return self.score_items_with_query_embedding(
            query_embedding,
            items,
            shard_path=shard_path,
            allow_write=allow_write,
        )

    def score_items_with_query_embedding(
        self,
        query_embedding: Any,
        items: list[EmbeddingIndexItem],
        *,
        shard_path: Path | None = None,
        allow_write: bool = True,
    ) -> list[float]:
        if not items:
            return []
        if not self.config.embedding_index_enabled or shard_path is None:
            texts = [item.text for item in items]
            embeddings = self.embedding_backend.embed_texts(texts, prompt_type="document")
        else:
            embeddings = self.ensure_embeddings(shard_path, items, allow_write=allow_write)

        if getattr(embeddings, "shape", (0,))[0] == 0 or getattr(query_embedding, "shape", (0,))[0] == 0:
            return [0.0 for _ in items]
        scores = embeddings @ query_embedding.T
        return [max(float(item), 0.0) for item in scores.squeeze(1).tolist()]

    def ensure_embeddings(
        self,
        shard_path: Path,
        items: list[EmbeddingIndexItem],
        *,
        allow_write: bool = True,
    ) -> Any:
        self.bootstrap()
        if not items:
            torch = self._torch_module()
            empty = torch.empty((0, 0), dtype=torch.float32)
            if allow_write:
                self._save_shard(shard_path, [], empty)
            return empty
        existing = self._load_shard(shard_path)
        return self._resolve_embeddings(shard_path, items, existing=existing, allow_write=allow_write)

    def refresh_embeddings(self, shard_path: Path, items: list[EmbeddingIndexItem]) -> Path:
        self.ensure_embeddings(shard_path, items, allow_write=True)
        return shard_path

    def _resolve_embeddings(
        self,
        shard_path: Path,
        items: list[EmbeddingIndexItem],
        *,
        existing: dict[str, Any] | None,
        allow_write: bool,
    ) -> Any:
        torch = self._torch_module()
        current_metadata = [item.metadata() for item in items]
        embedding_rows: list[Any] = [None] * len(items)
        changed = existing is None
        existing_map: dict[tuple[str, str, str, str, str], Any] = {}

        if existing is not None:
            stored_items = list(existing.get("items") or [])
            stored_embeddings = existing.get("embeddings")
            if not torch.is_tensor(stored_embeddings) or stored_embeddings.dim() != 2 or stored_embeddings.shape[0] != len(stored_items):
                existing = None
                changed = True
            else:
                existing_map = {
                    (
                        str(item.get("source_id", "")),
                        str(item.get("source_kind", "")),
                        str(item.get("source_path", "")),
                        str(item.get("target", "")),
                        str(item.get("text_hash", "")),
                    ): stored_embeddings[index]
                    for index, item in enumerate(stored_items)
                }
                if stored_items != current_metadata:
                    changed = True

        missing_indexes: list[int] = []
        missing_texts: list[str] = []
        for index, item in enumerate(items):
            cached = existing_map.get(item.key())
            if cached is None:
                changed = True
                missing_indexes.append(index)
                missing_texts.append(item.text)
                continue
            embedding_rows[index] = cached.float()

        if missing_texts:
            new_embeddings = self.embedding_backend.embed_texts(missing_texts, prompt_type="document")
            for index, row in zip(missing_indexes, new_embeddings):
                embedding_rows[index] = row.float()

        if any(row is None for row in embedding_rows):
            raise RuntimeError("embedding_index_incomplete")

        embeddings = torch.stack([row for row in embedding_rows], dim=0).float().cpu()
        if allow_write and changed:
            self._save_shard(shard_path, items, embeddings)
        return embeddings

    def _load_shard(self, shard_path: Path) -> dict[str, Any] | None:
        if not shard_path.exists():
            return None
        torch = self._torch_module()
        try:
            payload = torch.load(shard_path, map_location="cpu")
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        if str(payload.get("model_path", "")) != self._model_path_text():
            return None
        if int(payload.get("embedding_max_length", 0) or 0) != int(self.config.embedding_max_length):
            return None
        return payload

    def _save_shard(self, shard_path: Path, items: list[EmbeddingIndexItem], embeddings: Any) -> None:
        torch = self._torch_module()
        shard_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_path": self._model_path_text(),
            "embedding_max_length": int(self.config.embedding_max_length),
            "built_at": utc_now_iso(),
            "items": [item.metadata() for item in items],
            "embeddings": embeddings.cpu(),
        }
        torch.save(payload, shard_path)

    def _torch_module(self):
        cached = getattr(self.embedding_backend, "_torch", None)
        if cached is not None:
            return cached
        import torch

        return torch

    def _model_path_text(self) -> str:
        return str(getattr(self.embedding_backend, "model_path", self.config.embedding_model_path))


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
