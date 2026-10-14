"""
hybrid_retriever.py
=============================================================================
Georgian Legal AI (v4 Production Pipeline) — ნაბიჯი 4
Multi-Tier Hybrid Retriever for 21 Georgian Codes (5,624 Articles)

არქიტექტურა:
  - Tier 0 (`direct_lookup`): 0ms SQLite B-Tree SELECT კოდექსითა და მუხლის ნომრით
  - Tier 1 (`search_dense` + `search_sparse_fts5` + RRF):
      Qwen3-Embedding-8B (ChromaDB) + SQLite FTS5 Trigram -> Reciprocal Rank Fusion
  - Tier 2 (`hybrid_search`):
      Qwen/Qwen3-Reranker-0.6B Cross-Encoder GPU-ზე (FP16, max_length=512, batch=4)
      + კოდექსის კონტექსტური ბუსტინგი (Code-Aware Soft Prioritization)
=============================================================================
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Windows CUDA DLL-ების მიბმა llama_cpp-მდე
if os.name == "nt":
    try:
        import torch
        torch_lib = Path(torch.__file__).resolve().parent / "lib"
        if torch_lib.is_dir() and hasattr(os, "add_dll_directory"):
            os.add_dll_directory(str(torch_lib))
    except Exception:
        pass

import chromadb
import torch
from llama_cpp import Llama
from sentence_transformers import CrossEncoder

for _noisy in ("httpx", "httpcore", "huggingface_hub", "sentence_transformers", "chromadb"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

_THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _THIS_DIR.parent if _THIS_DIR.name == "src" else _THIS_DIR

DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "legal_database.db"
DEFAULT_CHROMA_DIR = PROJECT_ROOT / "data" / "chroma_db"
DEFAULT_COLLECTION = "georgian_legal_dense"
DEFAULT_EMBEDDER = os.getenv("EMBED_MODEL_PATH", "Qwen3-Embedding-8B-Q8_0.gguf")
DEFAULT_RERANKER = os.getenv("RERANKER_MODEL_NAME", "Qwen/Qwen3-Reranker-0.6B")

SUPERSCRIPT_MAP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")


def resolve_file_path(filename: str | Path) -> Path:
    p = Path(filename)
    if p.is_file():
        return p.resolve()
    candidates = [
        PROJECT_ROOT / p.name,
        PROJECT_ROOT / "data" / p.name,
        PROJECT_ROOT / "models" / p.name,
        PROJECT_ROOT.parent / p.name,
        PROJECT_ROOT.parent / "SuperProject" / p.name,
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    raise FileNotFoundError(f"ფაილი ვერ მოიძებნა: {filename}")


def l2_normalize(vec: List[float]) -> List[float]:
    arr = np.asarray(vec, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr[-1]
    norm = float(np.linalg.norm(arr))
    return (arr / norm).tolist() if norm > 0.0 else arr.tolist()


def normalize_article_number(raw_str: str) -> str:
    s = str(raw_str or "").strip()
    s = re.sub(r"\(\s*(\d+)\s*\)", r"^\1", s)
    if "^" not in s:
        m = re.match(r"^(\d+)([⁰¹²³⁴⁵⁶⁷⁸⁹]+)$", s)
        if m:
            s = f"{m.group(1)}^{m.group(2).translate(SUPERSCRIPT_MAP)}"
    else:
        s = s.translate(SUPERSCRIPT_MAP)
    return re.sub(r"\s+", "", s)


class GeorgianLegalHybridRetriever:
    def __init__(
        self,
        db_path: str | Path = DEFAULT_DB_PATH,
        chroma_dir: str | Path = DEFAULT_CHROMA_DIR,
        collection_name: str = DEFAULT_COLLECTION,
        embedder_path: str | Path = DEFAULT_EMBEDDER,
        reranker_name: str = DEFAULT_RERANKER,
        device: Optional[str] = None,
    ):
        self.db_path = str(resolve_file_path(db_path))
        self.chroma_dir = str(Path(chroma_dir).resolve())
        self.collection_name = collection_name
        self.embedder_path = str(resolve_file_path(embedder_path))
        self.reranker_name = reranker_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[*] საძიებო ძრავის ინიციალიზაცია (მოწყობილობა: {self.device.upper()})...")

        if not os.path.exists(self.chroma_dir):
            raise FileNotFoundError(f"ChromaDB საქაღალდე ვერ მოიძებნა: {self.chroma_dir}")

        self.chroma_client = chromadb.PersistentClient(path=self.chroma_dir)
        self.chroma_collection = self.chroma_client.get_collection(name=self.collection_name)
        print(f"[✓] SQLite & ChromaDB დაკავშირებულია ({self.chroma_collection.count()} ვექტორი, 21 კოდექსი).")

        self._init_embedder()
        self._init_reranker()

    def _init_embedder(self) -> None:
        print(f"[*] იტვირთება GGUF Embedder: {Path(self.embedder_path).name} ...")
        base_params = dict(
            model_path=self.embedder_path,
            embedding=True,
            n_gpu_layers=-1,
            n_ctx=2048,
            n_batch=512,
            verbose=False,
        )
        try:
            self.embedder = Llama(**base_params, flash_attn=True)
            print("[✓] Qwen3-8B Embedder მზადაა GPU-ზე (Flash Attention).")
        except Exception:
            self.embedder = Llama(**base_params, flash_attn=False)
            print("[✓] Qwen3-8B Embedder მზადაა GPU-ზე.")

    def _init_reranker(self) -> None:
        print(f"[*] იტვირთება Cross-Encoder Reranker: {self.reranker_name} ({self.device})...")
        if self.device == "cuda":
            try:
                self.reranker = CrossEncoder(
                    self.reranker_name,
                    device="cuda",
                    max_length=512,
                    model_kwargs={
                        "torch_dtype": torch.float16,
                        "attn_implementation": "sdpa",
                    },
                    trust_remote_code=True,
                )
                # 1-წამიანი ტესტი RTX 5060 Ti (sm_120) კერნელის თავსებადობაზე
                self.reranker.predict([("ტესტი", "ტესტი")], show_progress_bar=False)
                print("[✓] Cross-Encoder Reranker მზადაა (CUDA FP16).")
                return
            except Exception as e:
                print(f"[!] PyTorch CUDA შეუთავსებლობა ({str(e)[:60]}...). Reranker გადადის CPU რეჟიმზე.")
                self.device = "cpu"

        self.reranker = CrossEncoder(
            self.reranker_name,
            device="cpu",
            max_length=512,
            trust_remote_code=True,
        )
        print("[✓] Cross-Encoder Reranker მზადაა (CPU Fallback).")

    # =========================================================================
    # TIER 0: დეტერმინისტული პირდაპირი SQL ძიება (0 ms)
    # =========================================================================
    def direct_lookup(
        self, code_slug: Optional[str], article_no: str
    ) -> List[Dict[str, Any]]:
        norm_art = normalize_article_number(article_no)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        if code_slug:
            cur.execute(
                """
                SELECT node_id, code_slug, code_title, book, part, chapter,
                       article_no, title, body_text, breadcrumb
                FROM nodes
                WHERE code_slug = ? AND article_no = ?
                """,
                (code_slug, norm_art),
            )
            rows = cur.fetchall()
            # თუ მითითებულ კოდექსში ეს ნომერი არ აღმოჩნდა, მოვძებნოთ ძირითად კოდექსებში
            if not rows:
                cur.execute(
                    """
                    SELECT node_id, code_slug, code_title, book, part, chapter,
                           article_no, title, body_text, breadcrumb
                    FROM nodes
                    WHERE article_no = ?
                    LIMIT 4
                    """,
                    (norm_art,),
                )
                rows = cur.fetchall()
        else:
            cur.execute(
                """
                SELECT node_id, code_slug, code_title, book, part, chapter,
                       article_no, title, body_text, breadcrumb
                FROM nodes
                WHERE article_no = ?
                LIMIT 6
                """,
                (norm_art,),
            )
            rows = cur.fetchall()

        conn.close()

        results: List[Dict[str, Any]] = []
        for r in rows:
            results.append({
                "node_id": r["node_id"],
                "code_slug": r["code_slug"],
                "code_title": r["code_title"],
                "book": r["book"] or "",
                "part": r["part"] or "",
                "chapter": r["chapter"] or "",
                "article_no": r["article_no"],
                "title": r["title"],
                "body_text": r["body_text"],
                "breadcrumb": r["breadcrumb"],
                "retrieval_tier": "Tier 0 (Deterministic SQL)",
                "score": 1.0,
                "rerank_score": 10.0,
            })
        return results

    # =========================================================================
    # TIER 1A: Sparse FTS5 Trigram ძიება
    # =========================================================================
    def search_sparse_fts5(
        self,
        query: str,
        top_k: int = 25,
        code_slug: Optional[str] = None,
    ) -> List[str]:
        stop_words = {
            "რა", "როგორ", "როდის", "არის", "აქვს", "უნდა", "მიხედვით",
            "შესახებ", "დროს", "თუ", "ან", "და", "რომელი", "წლიდან", "ასაკიდან",
            "საქართველოში", "კოდექსის", "მუხლი", "მუხლში", "წერია", "ისჯება",
        }
        words = re.findall(r"[\u10D0-\u10FA0-9]+", query)
        valid_words = [w for w in words if len(w) >= 3 and w not in stop_words]
        if not valid_words:
            return []

        fts_query = " OR ".join(f'"{w}"' for w in valid_words[:12])
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        try:
            if code_slug:
                cur.execute(
                    """
                    SELECT f.node_id
                    FROM nodes_fts f
                    JOIN nodes n ON f.node_id = n.node_id
                    WHERE nodes_fts MATCH ? AND n.code_slug = ?
                    ORDER BY f.rank
                    LIMIT ?
                    """,
                    (fts_query, code_slug, top_k),
                )
            else:
                cur.execute(
                    """
                    SELECT node_id
                    FROM nodes_fts
                    WHERE nodes_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                    """,
                    (fts_query, top_k),
                )
            return [r[0] for r in cur.fetchall()]
        except sqlite3.OperationalError:
            return []
        finally:
            conn.close()

    # =========================================================================
    # TIER 1B: Dense ChromaDB ძიება (Qwen3-Embedding-8B)
    # =========================================================================
    def search_dense(
        self,
        query: str,
        top_k: int = 25,
        code_slug: Optional[str] = None,
    ) -> List[str]:
        raw_emb = self.embedder.create_embedding(query[:1500])
        query_vec = l2_normalize(raw_emb["data"][0]["embedding"])

        query_kwargs: Dict[str, Any] = {
            "query_embeddings": [query_vec],
            "n_results": top_k,
            "include": [],
        }
        if code_slug:
            query_kwargs["where"] = {"code_slug": code_slug}

        try:
            results = self.chroma_collection.query(**query_kwargs)
            if results and results.get("ids") and results["ids"][0]:
                return results["ids"][0]
        except Exception:
            pass
        return []

    # =========================================================================
    # RRF Fusion + Tier 2 Cross-Encoder Reranking
    # =========================================================================
    @staticmethod
    def _reciprocal_rank_fusion(
        ranked_lists: List[tuple[List[str], float]],
        k: int = 60,
        top_n: int = 25,
    ) -> List[str]:
        rrf_scores: Dict[str, float] = {}
        for id_list, weight in ranked_lists:
            for rank, node_id in enumerate(id_list):
                rrf_scores[node_id] = rrf_scores.get(node_id, 0.0) + (weight / (k + rank + 1))

        sorted_candidates = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [nid for nid, _ in sorted_candidates[:top_n]]

    def _fetch_nodes_by_ids(self, node_ids: List[str]) -> List[Dict[str, Any]]:
        if not node_ids:
            return []
        placeholders = ",".join("?" for _ in node_ids)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT node_id, code_slug, code_title, book, part, chapter,
                   article_no, title, body_text, breadcrumb
            FROM nodes
            WHERE node_id IN ({placeholders})
            """,
            node_ids,
        )
        rows = cur.fetchall()
        conn.close()

        nodes_map = {
            r["node_id"]: {
                "node_id": r["node_id"],
                "code_slug": r["code_slug"],
                "code_title": r["code_title"],
                "book": r["book"] or "",
                "part": r["part"] or "",
                "chapter": r["chapter"] or "",
                "article_no": r["article_no"],
                "title": r["title"],
                "body_text": r["body_text"],
                "breadcrumb": r["breadcrumb"],
                "retrieval_tier": "Tier 1 (Hybrid RRF)",
                "score": 0.0,
            }
            for r in rows
        }
        return [nodes_map[nid] for nid in node_ids if nid in nodes_map]

    def hybrid_search(
        self,
        query: str,
        top_k: int = 4,
        code_slug: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        ჰიბრიდული ძიება 21 კოდექსზე:
        თუ code_slug გადმოცემულია, აერთიანებს როგორც კონკრეტული კოდექსის (Filtered),
        ისე გლობალურ (Unfiltered) კანდიდატებს RRF-ში, რათა არც სიზუსტე დაიკარგოს და არც მომიჯნავე კოდექსები!
        """
        ranked_lists: List[tuple[List[str], float]] = []

        # 1. გლობალური ძიება 21-ვე კოდექსში
        global_dense = self.search_dense(query, top_k=20, code_slug=None)
        global_sparse = self.search_sparse_fts5(query, top_k=20, code_slug=None)
        ranked_lists.append((global_dense, 1.2))
        ranked_lists.append((global_sparse, 1.0))

        # 2. თუ სამიზნე კოდექსი ცნობილია, დავამატოთ კოდექს-სპეციფიკური კანდიდატები მაღალი წონით
        if code_slug:
            code_dense = self.search_dense(query, top_k=15, code_slug=code_slug)
            code_sparse = self.search_sparse_fts5(query, top_k=15, code_slug=code_slug)
            ranked_lists.append((code_dense, 1.5))
            ranked_lists.append((code_sparse, 1.3))

        fused_ids = self._reciprocal_rank_fusion(ranked_lists, k=60, top_n=24)
        candidates = self._fetch_nodes_by_ids(fused_ids)
        if not candidates:
            return []

        # 3. Tier 2 Cross-Encoder Reranking
        pairs = [
            (query, f"{c['breadcrumb']}\n{c['body_text'][:1100]}".strip())
            for c in candidates
        ]
        scores = self.reranker.predict(
            pairs,
            batch_size=4,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        for cand, raw_score in zip(candidates, scores):
            s_val = float(raw_score)
            # მსუბუქი ბონუსი სამიზნე კოდექსს თანაბარი ქულების დროს
            if code_slug and cand["code_slug"] == code_slug:
                s_val += 0.15
            cand["rerank_score"] = s_val
            cand["score"] = s_val
            cand["retrieval_tier"] = "Tier 2 (Cross-Encoder)"

        return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:top_k]

    @staticmethod
    def build_evidence_pack(articles: List[Dict[str, Any]]) -> str:
        if not articles:
            return "არცერთი მტკიცებულება (Evidence) არ მოიძებნა."

        pack_lines = [
            "### LEGAL EVIDENCE PACK (დოკუმენტური მტკიცებულებები) ###",
            "გამოიყენეთ მხოლოდ ქვემოთ მოცემული მუხლები პასუხის დასასაბუთებლად.",
            "=" * 70,
        ]
        for idx, art in enumerate(articles, 1):
            score_val = art.get("rerank_score", art.get("score", 0.0))
            pack_lines.extend([
                f"\n[E{idx}] {art['code_title']} | {art['title']}",
                f"იდენტიფიკატორი: {art['node_id']} | ტიპი: {art['retrieval_tier']} | Score: {score_val:.4f}",
                f"იერარქია: {art['breadcrumb']}",
                f"ნორმატიული ტექსტი:\n{art['body_text'].strip()}",
                "-" * 70,
            ])
        return "\n".join(pack_lines)

    def close(self) -> None:
        close_fn = getattr(getattr(self, "embedder", None), "close", None)
        if callable(close_fn):
            close_fn()