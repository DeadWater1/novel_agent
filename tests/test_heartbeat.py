from __future__ import annotations

from pathlib import Path

from novel_agent.compaction import ContextCompactionManager
from novel_agent.config import AgentConfig
from novel_agent.embedding_index import EmbeddingIndexManager
from novel_agent.heartbeat import HeartbeatManager
from novel_agent.memory import SessionStore
from novel_agent.schemas import BackendHealth
from novel_agent.session_meta import SessionMetaStore
from novel_agent.workspace import WorkspaceManager


class StubEmbeddingBackend:
    name = "embedding_backend"

    def healthcheck(self) -> BackendHealth:
        return BackendHealth(ok=True, name=self.name, detail="stub")

    def embed_query(self, query: str):
        return self.embed_texts([query], prompt_type="query")

    def embed_texts(self, texts: list[str], *, prompt_type: str = "document"):
        import torch

        values = []
        for text in texts:
            clean = text.strip()
            score = float(len(clean)) or 1.0
            values.append([score, 1.0])
        return torch.tensor(values, dtype=torch.float32)


def build_heartbeat(tmp_path: Path, *, long_term_repeat_threshold: int = 2):
    config = AgentConfig(
        workspace_root=tmp_path / "workspace",
        session_root=tmp_path / "sessions",
        embedding_index_root=tmp_path / "runtime" / "embeddings",
        idle_heartbeat_interval_seconds=0,
        long_term_repeat_threshold=long_term_repeat_threshold,
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
    session_store = SessionStore(config.session_root, summary_max_chars=config.session_summary_max_chars)
    session_store.bootstrap()
    meta_store = SessionMetaStore(config.session_root)
    meta_store.bootstrap()
    compaction_manager = ContextCompactionManager(config, session_store, meta_store)
    compaction_manager.bootstrap()
    heartbeat = HeartbeatManager(
        config=config,
        session_store=session_store,
        meta_store=meta_store,
        workspace=workspace,
        compaction_manager=compaction_manager,
        embedding_index_manager=embedding_index_manager,
    )
    return config, workspace, session_store, meta_store, compaction_manager, embedding_index_manager, heartbeat


def test_turn_heartbeat_updates_meta_and_cached_summary(tmp_path: Path):
    _, _, session_store, meta_store, _, _, heartbeat = build_heartbeat(tmp_path)
    session = session_store.create_session()
    session.add_user_message("请分析这章的人物关系")
    session.add_assistant_message("这里重点是林秋和沈砚的关系变化。")

    heartbeat.run_turn_heartbeat(session)

    meta = meta_store.load(session.session_id)
    assert meta is not None
    assert meta.dirty_summary is True
    assert meta.dirty_daily_memory is True
    assert meta.dirty_long_term is True
    assert meta.dirty_compaction is True
    assert meta.dirty_embedding is True
    assert "人物关系" in meta.cached_summary


def test_idle_heartbeat_writes_memory_and_clears_dirty_flags(tmp_path: Path):
    _, workspace, session_store, meta_store, compaction_manager, embedding_index_manager, heartbeat = build_heartbeat(
        tmp_path,
        long_term_repeat_threshold=1,
    )
    session = session_store.create_session()
    session.add_user_message("请压缩这一章，保留人物关系，保留剧情顺序。")
    session.add_assistant_message("这是压缩结果。")
    session_store.append_events(
        session,
        [
            {
                "turn_index": 1,
                "event_type": "user_message",
                "role": "user",
                "content": "请压缩这一章，保留人物关系，保留剧情顺序。",
            },
            {
                "turn_index": 1,
                "event_type": "agent_decision",
                "payload": {"action": "call_tool", "tool_name": "compress_chapter"},
            },
            {
                "turn_index": 1,
                "event_type": "tool_call",
                "tool_name": "compress_chapter",
                "tool_args": {"raw_text": "章节正文"},
            },
            {
                "turn_index": 1,
                "event_type": "tool_result",
                "tool_name": "compress_chapter",
                "content": "这是压缩结果。",
                "payload": {"tool_trace": {"requested_tool": "compress_chapter", "status": "ok"}},
            },
            {
                "turn_index": 1,
                "event_type": "assistant_message",
                "role": "assistant",
                "content": "这是压缩结果。",
            },
        ],
    )

    heartbeat.run_turn_heartbeat(session)
    processed = heartbeat.run_idle_heartbeat_once()
    assert processed == 1

    meta = meta_store.load(session.session_id)
    assert meta is not None
    assert meta.dirty_summary is False
    assert meta.dirty_daily_memory is False
    assert meta.dirty_long_term is False
    assert meta.dirty_compaction is False
    assert meta.dirty_embedding is False
    assert meta.last_memory_turn_index == 1
    assert meta.latest_compaction_path.endswith(f"{session.session_id}.json")
    assert Path(meta.latest_compaction_path).exists()
    assert meta.latest_session_embedding_path.endswith(f"{session.session_id}.pt")
    assert Path(meta.latest_session_embedding_path).exists()
    assert meta.latest_compaction_embedding_path.endswith(f"{session.session_id}.pt")
    assert Path(meta.latest_compaction_embedding_path).exists()
    assert embedding_index_manager.workspace_shard_path().exists()

    daily_content = workspace.ensure_daily_file().read_text(encoding="utf-8")
    long_term_content = (workspace.root / "memory.md").read_text(encoding="utf-8")
    assert "今天执行了 1 次章节压缩任务" in daily_content
    assert "用户偏好压缩时保留人物关系" in long_term_content
    assert "用户偏好压缩时保留剧情顺序" in long_term_content
    artifact = compaction_manager.load_compaction(session.session_id)
    assert artifact is not None
    assert artifact.compression_history
