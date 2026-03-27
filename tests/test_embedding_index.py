from __future__ import annotations

from pathlib import Path

import torch

from novel_agent.backends.embedding import LocalEmbeddingBackend
from novel_agent.config import AgentConfig
from novel_agent.embedding_index import EmbeddingIndexItem, EmbeddingIndexManager


class CountingEmbeddingBackend:
    def __init__(self) -> None:
        self.model_path = Path("/tmp/stub-embedding")
        self.document_texts_encoded = 0
        self.query_calls = 0
        self._torch = torch

    def embed_query(self, query: str):
        self.query_calls += 1
        return self.embed_texts([query], prompt_type="query")

    def embed_texts(self, texts: list[str], *, prompt_type: str = "document"):
        if prompt_type == "document":
            self.document_texts_encoded += len(texts)
        vectors = []
        for text in texts:
            clean = text.strip()
            vectors.append([float(len(clean)), float(len(set(clean))) or 1.0, 1.0 if prompt_type == "query" else 0.5])
        tensor = torch.tensor(vectors, dtype=torch.float32)
        return torch.nn.functional.normalize(tensor, p=2, dim=1)


def _item(text: str, *, target: str) -> EmbeddingIndexItem:
    return EmbeddingIndexItem.create(
        source_id=target,
        source_kind="memory",
        source_path="memory.md",
        target=target,
        text=text,
    )


def test_embedding_config_defaults_to_32k(tmp_path: Path):
    config = AgentConfig(workspace_root=tmp_path / "workspace", session_root=tmp_path / "sessions")
    assert config.embedding_max_length == 32768


def test_embedding_backend_effective_max_length_defaults_to_32k(tmp_path: Path):
    config = AgentConfig(workspace_root=tmp_path / "workspace", session_root=tmp_path / "sessions")
    backend = LocalEmbeddingBackend(config)
    backend._tokenizer = type("Tokenizer", (), {"model_max_length": 32768})()
    assert backend._effective_max_length() == 32768


def test_embedding_index_reuses_cached_shard_and_backfills_missing_items(tmp_path: Path):
    config = AgentConfig(
        workspace_root=tmp_path / "workspace",
        session_root=tmp_path / "sessions",
        embedding_index_root=tmp_path / "runtime" / "embeddings",
    )
    backend = CountingEmbeddingBackend()
    manager = EmbeddingIndexManager(config, backend)
    manager.bootstrap()

    shard_path = manager.workspace_shard_path()
    items = [_item("人物关系很复杂", target="a"), _item("今天讨论剧情推进", target="b")]

    scores_first = manager.score_items("人物关系", items, shard_path=shard_path)
    assert len(scores_first) == 2
    assert shard_path.exists()
    assert backend.document_texts_encoded == 2

    scores_second = manager.score_items("人物关系", items, shard_path=shard_path)
    assert len(scores_second) == 2
    assert backend.document_texts_encoded == 2

    updated_items = items + [_item("新的压缩结果", target="c")]
    manager.score_items("压缩结果", updated_items, shard_path=shard_path)
    assert backend.document_texts_encoded == 3


def test_embedding_index_rebuilds_when_text_hash_changes(tmp_path: Path):
    config = AgentConfig(
        workspace_root=tmp_path / "workspace",
        session_root=tmp_path / "sessions",
        embedding_index_root=tmp_path / "runtime" / "embeddings",
    )
    backend = CountingEmbeddingBackend()
    manager = EmbeddingIndexManager(config, backend)
    manager.bootstrap()

    shard_path = manager.workspace_shard_path()
    manager.refresh_embeddings(shard_path, [_item("旧内容", target="a")])
    first_payload = torch.load(shard_path, map_location="cpu")
    first_hash = first_payload["items"][0]["text_hash"]

    manager.refresh_embeddings(shard_path, [_item("新内容", target="a")])
    second_payload = torch.load(shard_path, map_location="cpu")
    second_hash = second_payload["items"][0]["text_hash"]

    assert first_hash != second_hash
