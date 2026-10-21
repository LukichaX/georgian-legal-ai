"""
src/hybrid_retriever.py
=============================================================================
Georgian Legal AI (v7.1 — Modular, Hardware-Agnostic Multi-Tier Retriever)
=============================================================================
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

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

for _noisy in ("httpx", "httpcore", "huggingface_hub", "sentence_transformers", "chromadb"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

logger = logging.getLogger("georgian_legal_retriever")

_THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _THIS_DIR.parent if _THIS_DIR.name in ("src", "backend", "utils") else _THIS_DIR

DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "legal_database.db"
DEFAULT_CHROMA_DIR = PROJECT_ROOT / "data" / "chroma_db"
DEFAULT_COLLECTION = os.getenv("CHROMA_COLLECTION", "georgian_legal_dense")

SUPERSCRIPT_MAP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")

_GEORGIAN_SUFFIXES = (
    "ებისთვის", "ებისგან", "ებისკენ", "ებამდე", "ებიდან", "თაგან",
    "ებისა", "ებით", "ებმა", "ებსა", "ების", "ებზე", "ებში", "ებად",
    "ისთვის", "ისგან", "ისკენ", "ამდე", "იდან", "ეთა",
    "ები", "ებს", "ითა", "ისა", "ზედ", "შია",
    "მა", "ის", "ით", "ზე", "ში", "ად", "სა", "ო", "ი",
)

_STOP_WORDS = {
    "რა", "როგორ", "როდის", "არის", "აქვს", "უნდა", "მიხედვით",
    "შესახებ", "დროს", "თუ", "ან", "და", "რომელი", "წლიდან", "ასაკიდან",
    "საქართველოში", "საქართველოს", "კოდექსის", "კოდექსი", "მუხლი", "მუხლში",
    "წერია", "ისჯება", "შემთხვევაში", "შეიძლება", "მითხარი", "ამიხსენი", "კანონით",
    "ზოგადი", "წესი", "წესები", "პირი", "პირის",
}


def resolve_file_path(filename: str | Path) -> Path:
    p = Path(filename)
    if p.is_file():
        return p.resolve()
    candidates = [
        PROJECT_ROOT / p,
        PROJECT_ROOT / p.name,
        PROJECT_ROOT / "models" / p.name,
        PROJECT_ROOT / "data" / p.name,
        PROJECT_ROOT.parent / p.name,
        PROJECT_ROOT.parent / "SuperProject" / p.name,
    ]
    for c in candidates:
        if c.is_file():
            return c.resolve()
    raise FileNotFoundError(f"ფაილი ვერ მოიძებნა: {filename} (მოიძებნა {PROJECT_ROOT}-ში და models/-ში)")


def l2_normalize(vec: Sequence[float]) -> List[float]:
    arr = np.asarray(vec, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr[-1]
    norm = float(np.linalg.norm(arr))
    return (arr / norm).tolist() if norm > 0.0 else arr.tolist()


def normalize_article_number(raw_str: str) -> str:
    s = str(raw_str or "").strip()
    s = re.sub(r"\s*(?:პრიმა|პრიმ)\b", "^1", s, flags=re.IGNORECASE)
    s = re.sub(r"\(\s*(\d+)\s*\)", r"^\1", s)
    if "^" not in s:
        m = re.match(r"^(\d+)([⁰¹²³⁴⁵⁶⁷⁸⁹]+)$", s)
        if m:
            s = f"{m.group(1)}^{m.group(2).translate(SUPERSCRIPT_MAP)}"
    else:
        s = s.translate(SUPERSCRIPT_MAP)
    return re.sub(r"\s+", "", s)


def _georgian_stem(word: str) -> str:
    """
    ქართული იურიდიული ტერმინების სტემერი:
    ინარჩუნებს 3-ასოიან ფუნდამენტურ ფუძეებს (ვად, წილ, სესხ, ნივთ, ზიან, ბრალ, ქირ),
    რაც ზუსტად ემთხვევა SQLite FTS5 Trigram-ის მინიმალურ სიგრძეს (3 სიმბოლო).
    """
    if len(word) <= 3:
        return word
    for suf in _GEORGIAN_SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            return word[: -len(suf)]
    return word


def extract_stemmed_keywords(query: str) -> List[str]:
    words = re.findall(r"[\u10D0-\u10FA0-9]+", query.casefold())
    stems: List[str] = []
    for w in words:
        if len(w) < 3 or w in _STOP_WORDS:
            continue
        st = _georgian_stem(w)
        if len(st) >= 3 and st not in _STOP_WORDS and st not in stems:
            stems.append(st)
    return stems


# =============================================================================
# MODULAR EMBEDDING & RERANKER ADAPTERS
# =============================================================================
class ModularEmbedder:
    def __init__(
        self,
        provider: Optional[str] = None,
        model_name_or_path: Optional[str] = None,
        existing_model: Optional[Any] = None,
        lock: Optional[threading.Lock] = None,
    ):
        self.provider = (provider or os.getenv("EMBED_PROVIDER", "huggingface")).strip().lower()
        self.model_spec = model_name_or_path or os.getenv("EMBED_MODEL_PATH", "Qwen/Qwen3-Embedding-0.6B")
        self.chunk_chars = int(os.getenv("EMBED_CHUNK_CHARS", "2400"))
        self.lock = lock or threading.Lock()
        self.model = existing_model

        if self.model is None:
            self._load_model()

    def _load_model(self) -> None:
        if self.provider in ("huggingface", "hf", "sentence_transformers"):
            from sentence_transformers import SentenceTransformer
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model_kwargs: Dict[str, Any] = {}
            if device == "cuda":
                model_kwargs["torch_dtype"] = torch.float16
            logger.info(f"[*] იტვირთება HuggingFace Embedder: {self.model_spec} ({device.upper()})...")
            self.model = SentenceTransformer(
                self.model_spec,
                device=device,
                trust_remote_code=True,
                model_kwargs=model_kwargs,
            )
            self.model.max_seq_length = int(os.getenv("EMBED_MAX_SEQ_LENGTH", "1024"))
            logger.info(f"[✓] HuggingFace Embedder ({self.model_spec}) მზადაა.")

        elif self.provider == "gguf":
            from llama_cpp import Llama
            resolved = str(resolve_file_path(self.model_spec))
            n_gpu = int(os.getenv("EMBED_GPU_LAYERS", "-1"))
            n_ctx = int(os.getenv("EMBED_CTX", "4096"))
            logger.info(f"[*] იტვირთება GGUF Embedder: {Path(resolved).name} (gpu_layers={n_gpu}, ctx={n_ctx})...")
            base_params = dict(
                model_path=resolved,
                embedding=True,
                n_gpu_layers=n_gpu,
                n_ctx=n_ctx,
                n_batch=1024,
                verbose=False,
            )
            try:
                self.model = Llama(**base_params, flash_attn=True)
            except Exception:
                self.model = Llama(**base_params, flash_attn=False)
            logger.info("[✓] GGUF Embedder მზადაა.")

        elif self.provider == "google":
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            g_model = self.model_spec if "embedding" in self.model_spec else "models/gemini-embedding-001"
            logger.info(f"[*] იტვირთება Cloud Google Embedder (0GB VRAM): {g_model}...")
            self.model = GoogleGenerativeAIEmbeddings(model=g_model)
            logger.info("[✓] Google Cloud Embedder მზადაა.")
        else:
            raise ValueError(f"უცნობი EMBED_PROVIDER: {self.provider}")

    def _split_long_article(self, text: str) -> List[str]:
        clean = text.strip()
        if len(clean) <= self.chunk_chars:
            return [clean]

        lines = clean.split("\n", 1)
        header = lines[0].strip()
        body = lines[1].strip() if len(lines) > 1 else clean

        step = max(800, self.chunk_chars - len(header) - 150)
        overlap = 200
        chunks: List[str] = []
        pos = 0
        while pos < len(body):
            piece = body[pos : pos + step]
            chunks.append(f"{header}\n\n{piece}")
            if pos + step >= len(body):
                break
            pos += step - overlap
        return chunks

    def _encode_raw_chunks(self, chunks: List[str]) -> List[List[float]]:
        if self.provider in ("huggingface", "hf", "sentence_transformers"):
            vecs = self.model.encode(
                chunks,
                batch_size=8,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            return [v.tolist() for v in vecs]
        elif self.provider == "gguf":
            try:
                raw = self.model.create_embedding(chunks)
                return [l2_normalize(item["embedding"]) for item in raw["data"]]
            except Exception:
                out_vecs = []
                for c in chunks:
                    single = self.model.create_embedding(c[:2000])
                    out_vecs.append(l2_normalize(single["data"][0]["embedding"]))
                return out_vecs
        elif self.provider == "google":
            vecs = self.model.embed_documents(chunks)
            return [l2_normalize(v) for v in vecs]
        raise RuntimeError("Embedder provider not initialized")

    def embed_query(self, text: str) -> List[float]:
        clean_text = text[:1800].strip()
        with self.lock:
            return self._encode_raw_chunks([clean_text])[0]

    def embed_documents(self, texts: List[str], max_chars: int = 0) -> List[List[float]]:
        all_chunks: List[str] = []
        doc_slices: List[Tuple[int, int]] = []

        for doc in texts:
            parts = self._split_long_article(doc)
            start = len(all_chunks)
            all_chunks.extend(parts)
            doc_slices.append((start, len(all_chunks)))

        with self.lock:
            chunk_vectors = self._encode_raw_chunks(all_chunks)

        final_doc_vectors: List[List[float]] = []
        for start, end in doc_slices:
            if end - start == 1:
                final_doc_vectors.append(chunk_vectors[start])
            else:
                sub_vecs = np.asarray(chunk_vectors[start:end], dtype=np.float32)
                weights = np.ones(len(sub_vecs), dtype=np.float32)
                weights[0] = 1.5
                pooled = np.average(sub_vecs, axis=0, weights=weights)
                final_doc_vectors.append(l2_normalize(pooled.tolist()))

        return final_doc_vectors

    def close(self) -> None:
        close_fn = getattr(self.model, "close", None)
        if callable(close_fn):
            close_fn()


class ModularReranker:
    def __init__(
        self,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        existing_model: Optional[Any] = None,
        lock: Optional[threading.Lock] = None,
    ):
        self.provider = (provider or os.getenv("RERANKER_PROVIDER", "cross_encoder")).strip().lower()
        self.model_name = model_name or os.getenv("RERANKER_MODEL_NAME", "Qwen/Qwen3-Reranker-0.6B")
        self.max_length = int(os.getenv("RERANKER_MAX_LENGTH", "1024"))
        self.lock = lock or threading.Lock()
        self.model = existing_model

        if self.model is None and self.provider not in ("none", "off", "false", "0"):
            self._load_model()

    def _load_model(self) -> None:
        from sentence_transformers import CrossEncoder
        req_device = os.getenv("RERANKER_DEVICE", "auto").strip().lower()
        device = ("cuda" if torch.cuda.is_available() else "cpu") if req_device == "auto" else req_device

        logger.info(f"[*] იტვირთება Cross-Encoder Reranker: {self.model_name} ({device})...")
        if device == "cuda":
            try:
                self.model = CrossEncoder(
                    self.model_name,
                    device="cuda",
                    max_length=self.max_length,
                    model_kwargs={"torch_dtype": torch.float16, "attn_implementation": "sdpa"},
                    trust_remote_code=True,
                )
                if getattr(self.model.tokenizer, "pad_token", None) is None:
                    self.model.tokenizer.pad_token = self.model.tokenizer.eos_token
                self.model.predict([("ტესტი", "ტესტი")], show_progress_bar=False)
                logger.info(f"[✓] Cross-Encoder Reranker მზადაა (CUDA FP16, max_length={self.max_length}).")
                return
            except Exception as exc:
                logger.warning(f"[!] CUDA Reranker შეუთავსებლობა ({str(exc)[:60]}). გადადის CPU-ზე.")

        self.model = CrossEncoder(
            self.model_name,
            device="cpu",
            max_length=self.max_length,
            trust_remote_code=True,
        )
        if getattr(self.model.tokenizer, "pad_token", None) is None:
            self.model.tokenizer.pad_token = self.model.tokenizer.eos_token
        logger.info("[✓] Cross-Encoder Reranker მზადაა (CPU Mode).")

    @property
    def is_enabled(self) -> bool:
        return self.model is not None and self.provider not in ("none", "off", "false", "0")

    def score_pairs(self, pairs: List[tuple[str, str]], batch_size: int = 4) -> List[float]:
        if not self.is_enabled or not pairs:
            return [0.0] * len(pairs)
        with self.lock:
            scores = self.model.predict(
                pairs,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            return [float(s) for s in scores]


# =============================================================================
# MAIN HYBRID RETRIEVER ENGINE
# =============================================================================
class GeorgianLegalHybridRetriever:
    def __init__(
        self,
        db_path: str | Path = DEFAULT_DB_PATH,
        chroma_dir: str | Path = DEFAULT_CHROMA_DIR,
        collection_name: str = DEFAULT_COLLECTION,
        embedder_adapter: Optional[ModularEmbedder] = None,
        reranker_adapter: Optional[ModularReranker] = None,
        chroma_collection: Optional[Any] = None,
    ):
        self.db_path = str(resolve_file_path(db_path))
        self.chroma_dir = str(Path(chroma_dir).resolve())
        self.collection_name = collection_name

        if chroma_collection is not None:
            self.chroma_collection = chroma_collection
        else:
            if not os.path.exists(self.chroma_dir):
                raise FileNotFoundError(f"ChromaDB საქაღალდე ვერ მოიძებნა: {self.chroma_dir}")
            self.chroma_client = chromadb.PersistentClient(path=self.chroma_dir)
            self.chroma_collection = self.chroma_client.get_collection(name=self.collection_name)

        self.embedder = embedder_adapter or ModularEmbedder()
        self.reranker = reranker_adapter or ModularReranker()

    def _get_db_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=15.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def get_article_by_node_id(self, node_id: str) -> Optional[Dict[str, Any]]:
        clean_id = (node_id or "").strip()
        if not clean_id:
            return None
        with self._get_db_conn() as conn:
            row = conn.execute(
                """
                SELECT node_id, code_slug, code_title, book, part, chapter,
                       article_no, title, body_text, breadcrumb
                FROM nodes
                WHERE node_id = ?
                """,
                (clean_id,),
            ).fetchone()
        if not row:
            return None
        return {
            "node_id": row["node_id"],
            "code_slug": row["code_slug"],
            "code_title": row["code_title"],
            "book": row["book"] or "",
            "part": row["part"] or "",
            "chapter": row["chapter"] or "",
            "article_no": row["article_no"],
            "title": row["title"],
            "body_text": row["body_text"],
            "breadcrumb": row["breadcrumb"],
        }

    # =========================================================================
    # TIER 0: DETERMINISTIC SQL LOOKUP
    # =========================================================================
    def direct_lookup(
        self,
        code_slug: Optional[str],
        article_no: str,
        strict_code: bool = False,
    ) -> List[Dict[str, Any]]:
        norm_art = normalize_article_number(article_no)
        if not norm_art or not re.match(r"^\d", norm_art) or norm_art == "0":
            return []

        rows = []
        with self._get_db_conn() as conn:
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

                if not rows and "^" in norm_art:
                    base_art = norm_art.split("^")[0]
                    cur.execute(
                        """
                        SELECT node_id, code_slug, code_title, book, part, chapter,
                               article_no, title, body_text, breadcrumb
                        FROM nodes
                        WHERE code_slug = ? AND (article_no = ? OR article_no LIKE ?)
                        ORDER BY article_no ASC
                        LIMIT 3
                        """,
                        (code_slug, base_art, f"{base_art}^%"),
                    )
                    rows = cur.fetchall()

                if not rows and not strict_code:
                    cur.execute(
                        """
                        SELECT node_id, code_slug, code_title, book, part, chapter,
                               article_no, title, body_text, breadcrumb
                        FROM nodes
                        WHERE article_no = ?
                        ORDER BY CASE code_slug
                            WHEN 'criminal' THEN 1
                            WHEN 'civil' THEN 2
                            WHEN 'labor' THEN 3
                            WHEN 'constitution' THEN 4
                            WHEN 'admin_offenses' THEN 5
                            ELSE 10 END
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
                    ORDER BY CASE code_slug
                        WHEN 'criminal' THEN 1
                        WHEN 'civil' THEN 2
                        WHEN 'labor' THEN 3
                        WHEN 'constitution' THEN 4
                        WHEN 'admin_offenses' THEN 5
                        ELSE 10 END
                    LIMIT 6
                    """,
                    (norm_art,),
                )
                rows = cur.fetchall()

        return [
            {
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
                "retrieval_tier": "TIER_0_SQL",
                "score": 10.0,
                "rerank_score": 10.0,
            }
            for r in rows
        ]

    def direct_lookup_many(
        self,
        code_slugs: Sequence[Optional[str]],
        article_numbers: Sequence[str],
        strict_code: bool = False,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        seen_ids = set()

        target_codes = [c for c in code_slugs if c] or [None]
        use_strict = strict_code or (len(target_codes) > 1)
        for art_no in article_numbers:
            for c_slug in target_codes:
                found = self.direct_lookup(c_slug, art_no, strict_code=use_strict)
                for item in found:
                    if item["node_id"] not in seen_ids:
                        seen_ids.add(item["node_id"])
                        results.append(item)

        if not results and article_numbers and not strict_code:
            for art_no in article_numbers:
                for item in self.direct_lookup(target_codes[0], art_no, strict_code=False):
                    if item["node_id"] not in seen_ids:
                        seen_ids.add(item["node_id"])
                        results.append(item)

        return results

    # =========================================================================
    # TIER 1C: STATUTORY TITLE & BREADCRUMB HEADER SEARCH (<1ms)
    # =========================================================================
    def search_titles_and_breadcrumbs(
        self,
        query: str,
        top_k: int = 10,
        code_slug: Optional[str] = None,
    ) -> List[str]:
        stems = extract_stemmed_keywords(query)[:8]
        if not stems:
            return []

        try:
            with self._get_db_conn() as conn:
                cur = conn.cursor()
                score_expr_parts: List[str] = []
                where_parts: List[str] = []
                score_params: List[Any] = []
                where_params: List[Any] = []

                for st in stems:
                    like_pat = f"%{st}%"
                    score_expr_parts.append(
                        "(CASE WHEN title LIKE ? THEN 3.2 ELSE 0.0 END + "
                        "CASE WHEN breadcrumb LIKE ? THEN 1.5 ELSE 0.0 END)"
                    )
                    score_params.extend([like_pat, like_pat])
                    where_parts.append("(title LIKE ? OR breadcrumb LIKE ?)")
                    where_params.extend([like_pat, like_pat])

                # ბონუსი ფუნდამენტური/დეფინიციური მუხლებისთვის ("ცნება", "განაკვეთი")
                definitional_bonus = (
                    "(CASE WHEN title LIKE '%ცნება%' OR title LIKE '%განაკვეთ%' "
                    "OR title LIKE '%ზოგადი დებულებ%' THEN 0.65 ELSE 0.0 END)"
                )
                score_sql = " + ".join(score_expr_parts) + f" + {definitional_bonus}"
                where_sql = " OR ".join(where_parts)

                if code_slug:
                    sql = f"""
                        SELECT node_id, ({score_sql}) - (length(title) * 0.004) AS hdr_score
                        FROM nodes
                        WHERE code_slug = ? AND ({where_sql})
                        ORDER BY hdr_score DESC
                        LIMIT ?
                    """
                    final_params = score_params + [code_slug] + where_params + [top_k]
                else:
                    sql = f"""
                        SELECT node_id, ({score_sql}) - (length(title) * 0.004) AS hdr_score
                        FROM nodes
                        WHERE {where_sql}
                        ORDER BY hdr_score DESC
                        LIMIT ?
                    """
                    final_params = score_params + where_params + [top_k]

                cur.execute(sql, final_params)
                return [r[0] for r in cur.fetchall()]
        except Exception:
            return []

    # =========================================================================
    # TIER 1A: SPARSE FTS5 TRIGRAM SEARCH (Title-Weighted BM25)
    # =========================================================================
    def search_sparse_fts5(
        self,
        query: str,
        top_k: int = 25,
        code_slug: Optional[str] = None,
    ) -> List[str]:
        words = re.findall(r"[\u10D0-\u10FA0-9]+", query.casefold())
        terms: List[str] = []
        seen_terms = set()

        for w in words:
            if len(w) < 3 or w in _STOP_WORDS:
                continue
            stem = _georgian_stem(w)
            for candidate_term in (stem, w):
                if len(candidate_term) >= 3 and candidate_term not in seen_terms:
                    seen_terms.add(candidate_term)
                    terms.append(candidate_term)

        if not terms:
            return []

        fts_query = " OR ".join(f'"{t}"' for t in terms[:16])
        try:
            with self._get_db_conn() as conn:
                cur = conn.cursor()
                if code_slug:
                    cur.execute(
                        """
                        SELECT f.node_id
                        FROM nodes_fts f
                        JOIN nodes n ON f.node_id = n.node_id
                        WHERE nodes_fts MATCH ? AND n.code_slug = ?
                        ORDER BY bm25(nodes_fts, 0.0, 10.0, 1.0, 6.0)
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
                        ORDER BY bm25(nodes_fts, 0.0, 10.0, 1.0, 6.0)
                        LIMIT ?
                        """,
                        (fts_query, top_k),
                    )
                return [r[0] for r in cur.fetchall()]
        except sqlite3.OperationalError:
            return []

    # =========================================================================
    # TIER 1B: DENSE CHROMADB SEARCH (Single-Pass Vector Reuse!)
    # =========================================================================
    def search_dense_by_vector(
        self,
        query_vec: List[float],
        top_k: int = 25,
        code_slug: Optional[str] = None,
    ) -> List[str]:
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

    def search_dense(
        self,
        query: str,
        top_k: int = 25,
        code_slug: Optional[str] = None,
    ) -> List[str]:
        query_vec = self.embedder.embed_query(query)
        return self.search_dense_by_vector(query_vec, top_k=top_k, code_slug=code_slug)

    # =========================================================================
    # RRF FUSION & TIER 2 RERANKING
    # =========================================================================
    @staticmethod
    def _reciprocal_rank_fusion(
        ranked_lists: List[tuple[List[str], float]],
        k: int = 60,
        top_n: int = 24,
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
        with self._get_db_conn() as conn:
            rows = conn.execute(
                f"""
                SELECT node_id, code_slug, code_title, book, part, chapter,
                       article_no, title, body_text, breadcrumb
                FROM nodes
                WHERE node_id IN ({placeholders})
                """,
                node_ids,
            ).fetchall()

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
                "retrieval_tier": "TIER_1_HYBRID_RRF",
                "score": 0.0,
            }
            for r in rows
        }
        return [nodes_map[nid] for nid in node_ids if nid in nodes_map]

    @staticmethod
    def _compute_header_match_bonus(query: str, cand: Dict[str, Any]) -> float:
        stems = extract_stemmed_keywords(query)[:6]
        if not stems:
            return 0.0
        title_clean = re.sub(r"^მუხლი\s+[^\s.]+\.?\s*", "", cand.get("title", "")).casefold()
        bc_clean = cand.get("breadcrumb", "").casefold()

        title_hits = sum(1 for st in stems if st in title_clean)
        bc_hits = sum(1 for st in stems if st in bc_clean)

        bonus = 0.0
        if title_hits > 0:
            ratio = title_hits / len(stems)
            bonus += 0.90 * ratio + (0.35 if len(title_clean) < 42 else 0.0)
            if any(k in title_clean for k in ("ცნება", "განაკვეთ", "ზოგადი დებულებ")):
                bonus += 0.25
        if bc_hits > 0:
            bonus += 0.35 * (bc_hits / len(stems))
        return min(bonus, 1.55)

    def hybrid_search(
        self,
        query: str,
        top_k: int = 6,
        code_slug: Optional[str] = None,
        secondary_code_slugs: Optional[Sequence[str]] = None,
        rerank_query: Optional[str] = None,
        max_candidates_to_rerank: int = 22,
    ) -> List[Dict[str, Any]]:
        ranked_lists: List[tuple[List[str], float]] = []

        query_vec = self.embedder.embed_query(query)

        # 1. გლობალური ძიება
        global_dense = self.search_dense_by_vector(query_vec, top_k=20, code_slug=None)
        global_sparse = self.search_sparse_fts5(query, top_k=20, code_slug=None)
        global_headers = self.search_titles_and_breadcrumbs(query, top_k=8, code_slug=None)
        ranked_lists.append((global_dense, 1.15))
        ranked_lists.append((global_sparse, 1.10))
        ranked_lists.append((global_headers, 1.40))

        # 2. ძირითადი სამიზნე კოდექსი
        target_codes_set = set()
        if code_slug:
            target_codes_set.add(code_slug)
            code_dense = self.search_dense_by_vector(query_vec, top_k=16, code_slug=code_slug)
            code_sparse = self.search_sparse_fts5(query, top_k=16, code_slug=code_slug)
            code_headers = self.search_titles_and_breadcrumbs(query, top_k=8, code_slug=code_slug)
            ranked_lists.append((code_dense, 1.55))
            ranked_lists.append((code_sparse, 1.45))
            ranked_lists.append((code_headers, 1.85))

        # 3. მომიჯნავე კოდექსები
        if secondary_code_slugs:
            for sec_slug in secondary_code_slugs:
                if sec_slug and sec_slug != code_slug:
                    target_codes_set.add(sec_slug)
                    s_dense = self.search_dense_by_vector(query_vec, top_k=10, code_slug=sec_slug)
                    s_sparse = self.search_sparse_fts5(query, top_k=10, code_slug=sec_slug)
                    s_headers = self.search_titles_and_breadcrumbs(query, top_k=5, code_slug=sec_slug)
                    ranked_lists.append((s_dense, 1.30))
                    ranked_lists.append((s_sparse, 1.25))
                    ranked_lists.append((s_headers, 1.50))

        fused_ids = self._reciprocal_rank_fusion(ranked_lists, k=60, top_n=max_candidates_to_rerank)
        candidates = self._fetch_nodes_by_ids(fused_ids)
        if not candidates:
            return []

        effective_rerank_q = (rerank_query or query).strip()[:380]

        if self.reranker.is_enabled:
            pairs = [
                (effective_rerank_q, f"{c['title']}\n{c['breadcrumb']}\n{c['body_text'][:1300]}".strip())
                for c in candidates
            ]
            raw_scores = self.reranker.score_pairs(pairs, batch_size=4)
            tier_label = "TIER_2_CROSS_ENCODER"
        else:
            raw_scores = [0.0] * len(candidates)
            tier_label = "TIER_1_HEADER_RRF"

        for rank_idx, (cand, raw_score) in enumerate(zip(candidates, raw_scores)):
            s_val = float(raw_score)
            if code_slug and cand["code_slug"] == code_slug:
                s_val += 0.25
            elif cand["code_slug"] in target_codes_set:
                s_val += 0.15
            s_val += self._compute_header_match_bonus(effective_rerank_q, cand)
            s_val += max(0.0, (max_candidates_to_rerank - rank_idx) * 0.004)
            cand["rerank_score"] = s_val
            cand["score"] = s_val
            cand["retrieval_tier"] = tier_label

        return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:top_k]

    def multi_query_hybrid_search(
        self,
        sub_queries: Sequence[str],
        top_k: int = 6,
        code_slug: Optional[str] = None,
        secondary_code_slugs: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        clean_queries = [q.strip() for q in sub_queries if q and q.strip()]
        if not clean_queries:
            return []
        if len(clean_queries) == 1:
            return self.hybrid_search(
                query=clean_queries[0],
                top_k=top_k,
                code_slug=code_slug,
                secondary_code_slugs=secondary_code_slugs,
            )

        per_query_results: List[List[Dict[str, Any]]] = []
        cands_per_sq = max(10, 24 // len(clean_queries))

        for sq in clean_queries[:3]:
            sq_hits = self.hybrid_search(
                query=sq,
                top_k=top_k,
                code_slug=code_slug,
                secondary_code_slugs=secondary_code_slugs,
                rerank_query=sq,
                max_candidates_to_rerank=cands_per_sq,
            )
            if sq_hits:
                per_query_results.append(sq_hits)

        if not per_query_results:
            return []

        merged: List[Dict[str, Any]] = []
        seen_ids = set()
        max_len = max(len(lst) for lst in per_query_results)

        for depth in range(max_len):
            for lst in per_query_results:
                if depth < len(lst):
                    cand = lst[depth]
                    nid = cand["node_id"]
                    if nid not in seen_ids:
                        seen_ids.add(nid)
                        merged.append(cand)
                        if len(merged) >= top_k:
                            return merged

        return merged

    @staticmethod
    def build_evidence_pack(articles: List[Dict[str, Any]]) -> str:
        if not articles:
            return "კონკრეტული კოდექსის მუხლები ამ სვლაზე არ მოძებნილა (გამოიყენეთ ზოგადი სამართლებრივი დოქტრინა)."

        pack_lines = [
            "### INTERNAL RETRIEVER CANDIDATES (სისტემის მიერ ბაზიდან ავტომატურად ამოღებული მუხლები) ###",
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
        if hasattr(self, "embedder") and self.embedder:
            self.embedder.close()