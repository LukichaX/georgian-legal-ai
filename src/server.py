"""
src/server.py
=============================================================================
Georgian Legal AI — Unified Production FastAPI Server & Auto Hot-Swap Engine
- Implements the full REST API contract for the Vite + React Frontend
- Pins heavy GPU models (Embedder, Reranker, Router, Chroma, Memory) in VRAM
- Automatically hot-reloads `hybrid_retriever.py`, `legal_agent.py`, and
  `telemetry_store.py` whenever you save a file (Ctrl+S), with 0s downtime!
- Serves the built React SPA (`frontend/dist`) for Cloudflare Tunnel hosting.
=============================================================================
"""

from __future__ import annotations

import importlib
import logging
import os
import sys
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

_THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _THIS_DIR.parent if _THIS_DIR.name in ("src", "backend") else _THIS_DIR
SRC_DIR = PROJECT_ROOT / "src"

for _p in (PROJECT_ROOT, SRC_DIR):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

load_dotenv(PROJECT_ROOT / ".env")

import hybrid_retriever
import legal_agent
import telemetry_store

logger = logging.getLogger("georgian_legal_server")
ADMIN_PASSWORD = os.getenv("ADMIN_PASS", "admin123")
ENABLE_DEV_RELOAD = os.getenv("ENABLE_DEV_RELOAD", "1") == "1"


# =============================================================================
# 1. ZERO-DOWNTIME GPU HOT-SWAP ENGINE
# =============================================================================
class HotSwapEngine:
    """
    ინახავს მძიმე GPU/VRAM რესურსებს უძრავად და ავტომატურად აახლებს
    Python მოდულებს (hybrid_retriever, legal_agent, telemetry_store) კოდის შეცვლისას.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.gpu_resources: Dict[str, Any] = {}
        self.mtimes: Dict[str, float] = {}
        self.agent: Optional[Any] = None
        self.store: Optional[Any] = None

    def initialize(self) -> None:
        with self._lock:
            logger.info("🚀 მიმდინარეობს GPU მოდელების ერთჯერადი ჩატვირთვა VRAM-ში...")
            initial_retriever = hybrid_retriever.GeorgianLegalHybridRetriever()
            initial_agent = legal_agent.LegalAgentPipeline(retriever=initial_retriever)
            initial_store = telemetry_store.TelemetryStore()

            self.gpu_resources = {
                "embedder_adapter": initial_retriever.embedder,
                "reranker_adapter": initial_retriever.reranker,
                "chroma_collection": initial_retriever.chroma_collection,
                "router_llm": initial_agent.router_llm,
                "router_lock": initial_agent._router_lock,
                "memory": initial_agent.memory,
            }
            self.agent = initial_agent
            self.store = initial_store
            self._record_mtimes()
            logger.info("✅ GPU მოდელები ჩატვირთულია VRAM-ში. Auto Hot-Swap აქტიურია!")

    def _record_mtimes(self) -> None:
        for mod in (hybrid_retriever, legal_agent, telemetry_store):
            fpath = getattr(mod, "__file__", None)
            if fpath and os.path.exists(fpath):
                self.mtimes[fpath] = os.path.getmtime(fpath)

    def check_and_reload(self, force: bool = False) -> Dict[str, Any]:
        if not ENABLE_DEV_RELOAD and not force:
            return {"reloaded": False, "modules": []}

        changed_modules: List[str] = []
        for mod in (hybrid_retriever, legal_agent, telemetry_store):
            fpath = getattr(mod, "__file__", None)
            if fpath and os.path.exists(fpath):
                curr_mtime = os.path.getmtime(fpath)
                if force or curr_mtime > self.mtimes.get(fpath, 0.0):
                    changed_modules.append(mod.__name__)

        if not changed_modules:
            return {"reloaded": False, "modules": []}

        with self._lock:
            try:
                logger.info(f"♻️ კოდის ცვლილება დაფიქსირდა ({', '.join(changed_modules)}). მიმდინარეობს Hot-Swap...")
                importlib.reload(hybrid_retriever)
                importlib.reload(legal_agent)
                importlib.reload(telemetry_store)

                new_retriever = hybrid_retriever.GeorgianLegalHybridRetriever(
                    embedder_adapter=self.gpu_resources["embedder_adapter"],
                    reranker_adapter=self.gpu_resources["reranker_adapter"],
                    chroma_collection=self.gpu_resources["chroma_collection"],
                )
                self.agent = legal_agent.LegalAgentPipeline(
                    router_llm=self.gpu_resources["router_llm"],
                    retriever=new_retriever,
                    memory=self.gpu_resources["memory"],
                    router_lock=self.gpu_resources["router_lock"],
                )
                self.store = telemetry_store.TelemetryStore()
                self._record_mtimes()
                logger.info("✅ Hot-Swap წარმატებით დასრულდა 0 წამში (VRAM ხელუხლებელია)!")
                return {"reloaded": True, "modules": changed_modules}
            except Exception as exc:
                logger.error(f"❌ Hot-Swap შეცდომა (შენარჩუნებულია წინა მუშა ვერსია): {exc}")
                return {"reloaded": False, "error": str(exc)}

    def shutdown(self) -> None:
        if self.agent:
            self.agent.close()


ENGINE = HotSwapEngine()


@asynccontextmanager
async def lifespan(app: FastAPI):
    ENGINE.initialize()
    yield
    ENGINE.shutdown()


app = FastAPI(
    title="Georgian Legal AI API",
    version="7.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# 2. PYDANTIC v2 SCHEMAS (Matches FRONTEND_ARCHITECTURE_AND_DECISIONS.md)
# =============================================================================
class ChatRequest(BaseModel):
    thread_id: str
    tester_id: str
    tester_name: Optional[str] = "მომხმარებელი"
    message: str


class CreateSessionRequest(BaseModel):
    tester_id: str
    tester_name: str
    title: Optional[str] = "ახალი დიალოგი"


class FeedbackRequest(BaseModel):
    turn_id: str
    rating: int = Field(..., ge=1, le=10)
    issue_tags: List[str] = Field(default_factory=list)
    expected_article_hint: str = ""
    tester_note: str = ""


# =============================================================================
# 3. API ENDPOINTS
# =============================================================================
@app.post("/api/chat")
def api_chat(req: ChatRequest) -> Dict[str, Any]:
    ENGINE.check_and_reload()
    agent = ENGINE.agent
    store = ENGINE.store

    msg = req.message.strip()
    if not msg:
        raise HTTPException(status_code=400, detail="შეკითხვა არ შეიძლება იყოს ცარიელი.")

    # დავრწმუნდეთ, რომ სესია არსებობს
    store.create_session(req.thread_id, req.tester_id, req.tester_name or "მომხმარებელი")
    existing_turns = store.get_session_turns(req.thread_id, req.tester_id)

    result = agent.run(msg, thread_id=req.thread_id)
    answer_text = result.get("answer", "")

    meta_data = {
        "router_decision": result.get("router_decision") or {},
        "retrieved_articles": result.get("retrieved_articles") or [],
        "citation_status": result.get("citation_status") or {},
        "timings": {k: round(float(v), 3) for k, v in (result.get("timings") or {}).items()},
    }

    turn_id = f"turn_{uuid.uuid4().hex[:10]}"
    articles_compact = store.log_turn(
        turn_id=turn_id,
        thread_id=req.thread_id,
        tester_id=req.tester_id,
        tester_name=req.tester_name or "მომხმარებელი",
        turn_index=len(existing_turns) + 1,
        user_query=msg,
        assistant_answer=answer_text,
        meta=meta_data,
    )

    return {
        "turn_id": turn_id,
        "answer": answer_text,
        "router_decision": meta_data["router_decision"],
        "retrieved_articles": articles_compact,
        "citation_status": meta_data["citation_status"],
        "timings": meta_data["timings"],
    }


@app.get("/api/articles/{node_id:path}")
def api_get_article(node_id: str) -> Dict[str, Any]:
    ENGINE.check_and_reload()
    art = ENGINE.agent.retriever.get_article_by_node_id(node_id)
    if not art:
        raise HTTPException(status_code=404, detail=f"მუხლი ვერ მოიძებნა: {node_id}")
    return art


@app.get("/api/sessions")
def api_get_sessions(
    tester_id: str = Query(...),
    tester_name: str = Query("მომხმარებელი"),
) -> List[Dict[str, Any]]:
    ENGINE.check_and_reload()
    store = ENGINE.store
    sessions = store.get_user_sessions(tester_id)
    if not sessions:
        first_tid = f"{tester_id}_s_{uuid.uuid4().hex[:6]}"
        store.create_session(first_tid, tester_id, tester_name)
        sessions = store.get_user_sessions(tester_id)
    return sessions


@app.post("/api/sessions")
def api_create_session(req: CreateSessionRequest) -> Dict[str, Any]:
    ENGINE.check_and_reload()
    new_tid = f"{req.tester_id}_s_{uuid.uuid4().hex[:6]}"
    return ENGINE.store.create_session(new_tid, req.tester_id, req.tester_name, req.title or "ახალი დიალოგი")


@app.get("/api/sessions/{thread_id}/turns")
def api_get_session_turns(
    thread_id: str,
    tester_id: str = Query(...),
) -> List[Dict[str, Any]]:
    ENGINE.check_and_reload()
    return ENGINE.store.get_session_turns(thread_id, tester_id)


@app.post("/api/feedback")
def api_save_feedback(req: FeedbackRequest) -> Dict[str, Any]:
    ENGINE.check_and_reload()
    ENGINE.store.save_feedback(
        turn_id=req.turn_id,
        rating=req.rating,
        issue_tags=req.issue_tags,
        expected_article_hint=req.expected_article_hint,
        tester_note=req.tester_note,
    )
    return {"status": "ok", "turn_id": req.turn_id}


@app.get("/api/admin/export")
def api_admin_export(x_admin_password: str = Header(default="")) -> Dict[str, Any]:
    if x_admin_password != ADMIN_PASSWORD:
        raise HTTPException(status_code=403, detail="არასწორი Admin პაროლი.")
    ENGINE.check_and_reload()
    stats = ENGINE.store.get_summary_stats()
    jsonl_content = ENGINE.store._rebuild_jsonl_export()
    return {
        "stats": stats,
        "jsonl": jsonl_content,
    }


@app.post("/api/admin/reload")
def api_admin_reload(x_admin_password: str = Header(default="")) -> Dict[str, Any]:
    if x_admin_password != ADMIN_PASSWORD:
        raise HTTPException(status_code=403, detail="არასწორი Admin პაროლი.")
    res = ENGINE.check_and_reload(force=True)
    return {"status": "ok", **res}


# =============================================================================
# 4. STATIC FRONTEND SERVING (Vite React Build in `frontend/dist`)
# =============================================================================
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"

if FRONTEND_DIST.exists():
    assets_dir = FRONTEND_DIST / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    @app.get("/{full_path:path}")
    def serve_spa(full_path: str):
        target_file = FRONTEND_DIST / full_path
        if full_path and target_file.is_file():
            return FileResponse(str(target_file))
        index_html = FRONTEND_DIST / "index.html"
        if index_html.is_file():
            return FileResponse(str(index_html))
        return JSONResponse({"status": "API Running (Frontend index.html not found)"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)