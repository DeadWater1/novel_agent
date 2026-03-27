from __future__ import annotations

from pathlib import Path

from novel_agent.config import AgentConfig
from novel_agent.embedding_index import EmbeddingIndexManager
from novel_agent.schemas import BackendHealth
from novel_agent.workspace import WorkspaceManager


class StubEmbeddingBackend:
    name = "embedding_backend"

    def healthcheck(self) -> BackendHealth:
        return BackendHealth(ok=True, name=self.name, detail="stub")

    def similarity(self, query: str, text: str) -> float:
        return self.similarity_batch(query, [text])[0]

    def similarity_batch(self, query: str, texts: list[str]) -> list[float]:
        clean_query = query.strip()
        query_chars = set(clean_query)
        results: list[float] = []
        for text in texts:
            clean_text = text.strip()
            if not clean_query or not clean_text:
                results.append(0.0)
                continue
            overlap = len(query_chars & set(clean_text)) / max(len(query_chars), 1)
            exact_bonus = 1.0 if clean_query in clean_text else 0.0
            results.append(overlap + exact_bonus)
        return results

    def embed_query(self, query: str):
        return self.embed_texts([query], prompt_type="query")

    def embed_texts(self, texts: list[str], *, prompt_type: str = "document"):
        import torch

        vectors = []
        for text in texts:
            clean_text = text.strip()
            if not clean_text:
                vectors.append([0.0] * 128)
                continue
            vector = [0.0] * 128
            for char in clean_text:
                vector[ord(char) % 128] += 1.0
            if prompt_type == "query":
                vector[0] += 0.5
            vectors.append(vector)
        tensor = torch.tensor(vectors, dtype=torch.float32)
        return torch.nn.functional.normalize(tensor, p=2, dim=1)


def test_workspace_doc_order_contains_agent_first(tmp_path: Path):
    config = AgentConfig(
        workspace_root=tmp_path / "workspace",
        session_root=tmp_path / "sessions",
        embedding_index_root=tmp_path / "runtime" / "embeddings",
    )
    embedding_backend = StubEmbeddingBackend()
    embedding_index_manager = EmbeddingIndexManager(config, embedding_backend)
    embedding_index_manager.bootstrap()
    workspace = WorkspaceManager(
        config,
        embedding_backend=embedding_backend,
        embedding_index_manager=embedding_index_manager,
    )
    workspace.bootstrap()
    docs = workspace.load_workspace_docs("摘要")
    agent_index = docs.find("# Agent")
    tools_index = docs.find("# Tools")
    memory_access_index = docs.find("# Memory Access")
    session_index = docs.find("# Session Summary")
    assert agent_index != -1
    assert tools_index != -1
    assert memory_access_index != -1
    assert session_index != -1
    assert agent_index < tools_index < memory_access_index < session_index


def test_memory_search_and_memory_get_are_available(tmp_path: Path):
    config = AgentConfig(
        workspace_root=tmp_path / "workspace",
        session_root=tmp_path / "sessions",
        embedding_index_root=tmp_path / "runtime" / "embeddings",
    )
    embedding_backend = StubEmbeddingBackend()
    embedding_index_manager = EmbeddingIndexManager(config, embedding_backend)
    embedding_index_manager.bootstrap()
    workspace = WorkspaceManager(
        config,
        embedding_backend=embedding_backend,
        embedding_index_manager=embedding_index_manager,
    )
    workspace.bootstrap()
    workspace.append_long_term_entries(["用户偏好压缩时保留人物关系"])
    workspace.append_daily_entries(["今天讨论了剧情推进"])

    results = workspace.memory_search("人物关系", max_results=3)
    assert results
    assert results[0]["source_id"] == "long_term"
    assert str(results[0]["target"]).startswith("long_term#L")

    latest_daily = workspace.memory_get("daily_latest")
    assert latest_daily["target"] == "daily_latest"
    assert "今天讨论了剧情推进" in latest_daily["content"]

    ranged = workspace.memory_get(str(results[0]["target"]))
    assert ranged["resolved_target"].startswith("long_term#L")
    assert "人物关系" in ranged["content"]
