"""
step3_indexer.py
=============================================================================
Georgian Legal AI (v4 Production Pipeline) — ნაბიჯი 3 (v3.1 Fixed)
Dual Storage Indexer for all 21 Georgian Codes (5,624 Active Articles):
  1. SQLite Relational Table (`nodes`) + B-Tree Index for 0ms Tier 0 Lookup
  2. SQLite FTS5 (`nodes_fts`) with Trigram Tokenizer for Georgian BM25 Search
  3. ChromaDB Dense Vector Store (`georgian_legal_dense`) using
     Qwen3-Embedding-8B-Q8_0.gguf on CUDA GPU (Resumable Batching + L2 Norm)
=============================================================================
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List

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
from llama_cpp import Llama

_THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _THIS_DIR.parent if _THIS_DIR.name == "src" else _THIS_DIR

AST_DIR = PROJECT_ROOT / "data" / "processed_ast"
DB_PATH = PROJECT_ROOT / "data" / "legal_database.db"
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma_db"
COLLECTION_NAME = "georgian_legal_dense"

EMBED_FILENAME = os.getenv("EMBED_MODEL_PATH", "Qwen3-Embedding-8B-Q8_0.gguf")
MAX_EMBED_CHARS = 2800  # უსაფრთხო ზღვარი 4096 კონტექსტისთვის ქართულ უნიკოდზე
BATCH_SIZE = 16


def resolve_model_path(filename: str) -> Path:
    """ეძებს GGUF მოდელს მიმდინარე პროექტში და მშობელ/SuperProject საქაღალდეებში."""
    candidates = [
        Path(filename),
        PROJECT_ROOT / filename,
        PROJECT_ROOT / "models" / filename,
        PROJECT_ROOT.parent / filename,
        PROJECT_ROOT.parent / "SuperProject" / filename,
    ]
    for c in candidates:
        if c.is_file():
            return c.resolve()
    raise FileNotFoundError(
        f"[!] GGUF მოდელი '{filename}' ვერ მოიძებნა. "
        f"გთხოვთ, ჩააკოპიროთ {PROJECT_ROOT} საქაღალდეში."
    )


def l2_normalize(vec: List[float]) -> List[float]:
    arr = np.asarray(vec, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr[-1]
    norm = float(np.linalg.norm(arr))
    return (arr / norm).tolist() if norm > 0.0 else arr.tolist()


def load_all_ast_articles() -> List[Dict[str, Any]]:
    if not AST_DIR.exists():
        raise FileNotFoundError(f"AST საქაღალდე ვერ მოიძებნა: {AST_DIR}. ჯერ გაუშვით step2_ast_parser.py")

    ast_files = sorted(AST_DIR.glob("ast_*.json"))
    if not ast_files:
        raise FileNotFoundError(f"{AST_DIR}-ში ast_*.json ფაილები არ არის.")

    all_articles: List[Dict[str, Any]] = []
    seen_ids = set()

    for fpath in ast_files:
        # გამოვტოვოთ აუდიტის რეპორტის ფაილი!
        if fpath.name == "ast_audit_report.json":
            continue

        records = json.loads(fpath.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            continue

        for r in records:
            if not isinstance(r, dict) or "node_id" not in r:
                continue
            nid = r["node_id"]
            body = (r.get("body_text") or "").strip()
            if not body or nid in seen_ids:
                continue
            seen_ids.add(nid)
            all_articles.append(r)

    return all_articles


# =============================================================================
# 1. SQLite + FTS5 Trigram ინდექსაცია
# =============================================================================
def build_sqlite_database(articles: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 95)
    print(f"🗄️ [1/2] იგება SQLite + FTS5 Trigram ბაზა: {DB_PATH}")
    print("=" * 95)

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()

    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()

    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=NORMAL;")

    cur.execute("""
        CREATE TABLE nodes (
            node_id TEXT PRIMARY KEY,
            code_slug TEXT NOT NULL,
            code_title TEXT NOT NULL,
            book TEXT,
            part TEXT,
            chapter TEXT,
            subchapter TEXT,
            article_no TEXT NOT NULL,
            title TEXT NOT NULL,
            body_text TEXT NOT NULL,
            breadcrumb TEXT NOT NULL,
            amendments TEXT
        );
    """)

    cur.execute("CREATE INDEX idx_code_article ON nodes(code_slug, article_no);")
    cur.execute("CREATE INDEX idx_article_only ON nodes(article_no);")
    cur.execute("CREATE INDEX idx_code_slug ON nodes(code_slug);")

    cur.execute("""
        CREATE VIRTUAL TABLE nodes_fts USING fts5(
            node_id UNINDEXED,
            title,
            body_text,
            breadcrumb,
            tokenize='trigram'
        );
    """)

    rows_nodes = []
    rows_fts = []
    for a in articles:
        rows_nodes.append((
            a["node_id"],
            a["code_slug"],
            a["code_title"],
            a.get("book", ""),
            a.get("part", ""),
            a.get("chapter", ""),
            a.get("subchapter", ""),
            a["article_no"],
            a["title"],
            a["body_text"],
            a["breadcrumb"],
            json.dumps(a.get("amendments", []), ensure_ascii=False),
        ))
        rows_fts.append((
            a["node_id"],
            a["title"],
            a["body_text"],
            a["breadcrumb"],
        ))

    cur.executemany("""
        INSERT INTO nodes (
            node_id, code_slug, code_title, book, part, chapter, subchapter,
            article_no, title, body_text, breadcrumb, amendments
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """, rows_nodes)

    cur.executemany("""
        INSERT INTO nodes_fts (node_id, title, body_text, breadcrumb)
        VALUES (?, ?, ?, ?);
    """, rows_fts)

    conn.commit()
    conn.close()
    print(f"[✓] SQLite ბაზა მზადაა! ჩაწერილია {len(articles)} სუფთა მუხლი + FTS5 Trigram ინდექსი.")


# =============================================================================
# 2. ChromaDB GPU ვექტორული ინდექსაცია (Resumable Batching)
# =============================================================================
def build_chroma_vector_store(articles: List[Dict[str, Any]], rebuild: bool = False) -> None:
    print("\n" + "=" * 95)
    print(f"🧠 [2/2] იგება ChromaDB ვექტორული საცავი GPU-ზე: {CHROMA_DIR}")
    print("=" * 95)

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    if rebuild:
        try:
            client.delete_collection(COLLECTION_NAME)
            print("[*] ძველი ChromaDB კოლექცია წაიშალა (--rebuild).")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    existing_ids = set()
    if collection.count() > 0:
        existing_data = collection.get(include=[])
        existing_ids = set(existing_data.get("ids") or [])

    pending_articles = [a for a in articles if a["node_id"] not in existing_ids]

    if not pending_articles:
        print(f"[✓] ყველა მუხლი ({collection.count()} ვექტორი) უკვე დაინდექსებულია ChromaDB-ში!")
        return

    print(
        f"[*] სულ მუხლები: {len(articles)} | უკვე ბაზაშია: {len(existing_ids)} | "
        f"დასაინდექსებელია: {len(pending_articles)}"
    )

    model_path = resolve_model_path(EMBED_FILENAME)
    print(f"[*] იტვირთება GGUF Embedder GPU-ზე: {model_path} ...")
    base_params = dict(
        model_path=str(model_path),
        embedding=True,
        n_gpu_layers=-1,
        n_ctx=4096,
        n_batch=1024,
        verbose=False,
    )
    try:
        embedder = Llama(**base_params, flash_attn=True)
        print("[✓] Qwen3-Embedding-8B მზადაა GPU-ზე (Flash Attention აქტიურია).")
    except Exception:
        embedder = Llama(**base_params, flash_attn=False)
        print("[✓] Qwen3-Embedding-8B მზადაა GPU-ზე.")

    t_start = time.perf_counter()
    total_pending = len(pending_articles)

    for start_idx in range(0, total_pending, BATCH_SIZE):
        batch = pending_articles[start_idx : start_idx + BATCH_SIZE]

        batch_ids: List[str] = []
        batch_docs: List[str] = []
        batch_embed_inputs: List[str] = []
        batch_metas: List[Dict[str, Any]] = []

        for art in batch:
            full_doc = f"{art['breadcrumb']}\n\n{art['body_text']}".strip()
            batch_ids.append(art["node_id"])
            batch_docs.append(full_doc)
            batch_embed_inputs.append(full_doc[:MAX_EMBED_CHARS])
            batch_metas.append({
                "node_id": art["node_id"],
                "code_slug": art["code_slug"],
                "code_title": art["code_title"],
                "article_no": art["article_no"],
                "title": art["title"],
                "breadcrumb": art["breadcrumb"],
            })

        try:
            raw_out = embedder.create_embedding(batch_embed_inputs)
            batch_embs = [l2_normalize(item["embedding"]) for item in raw_out["data"]]
        except Exception:
            # Fallback სათითაო ვექტორიზაციაზე, თუ რომელიმე მუხლი ძალიან გრძელია
            batch_embs = []
            for text_in in batch_embed_inputs:
                single_out = embedder.create_embedding(text_in[:1800])
                batch_embs.append(l2_normalize(single_out["data"][0]["embedding"]))

        collection.upsert(
            ids=batch_ids,
            documents=batch_docs,
            metadatas=batch_metas,
            embeddings=batch_embs,
        )

        done_count = min(start_idx + BATCH_SIZE, total_pending)
        if done_count % (BATCH_SIZE * 4) == 0 or done_count == total_pending:
            elapsed = time.perf_counter() - t_start
            rate = done_count / elapsed if elapsed > 0 else 0.0
            eta_sec = (total_pending - done_count) / rate if rate > 0 else 0.0
            print(
                f"  -> [{done_count:>4}/{total_pending}] ({done_count/total_pending*100:>5.1f}%) | "
                f"სიჩქარე: {rate:.1f} მუხლი/წმ | გავიდა: {elapsed:.0f}წმ | დარჩა (ETA): ~{eta_sec:.0f}წმ"
            )

    if hasattr(embedder, "close"):
        embedder.close()

    print(f"\n[🚀] ChromaDB ინდექსაცია დასრულდა! კოლექციაშია {collection.count()} ვექტორი.")


# =============================================================================
# 3. საბოლოო Sanity Check კრიტიკულ მუხლებზე
# =============================================================================
def verify_critical_articles() -> None:
    print("\n" + "=" * 95)
    print("🔬 [SANITY CHECK] კრიტიკული მუხლების გადამოწმება ახალ ბაზაში (21 კოდექსი):")
    print("=" * 95)

    test_ids = [
        ("criminal:a108", "განზრახ მკვლელობა"),
        ("criminal:a140", "სქესობრივი ხელშეუხებლობის ასაკი (16 წელი) - წინა ბაზაში ცარიელი იყო!"),
        ("criminal:a144^1", "წამება (პრიმ-მუხლი ^1)"),
        ("criminal:a126", "ძალადობა - წინა ბაზაში ცარიელი იყო!"),
        ("civil:a1108", "საქორწინო ასაკი (18 წელი)"),
        ("labor:a31", "შვებულების ხანგრძლივობა"),
        ("constitution:a1", "საქართველოს კონსტიტუცია მუხლი 1"),
        ("admin_offenses:a166", "წვრილმანი ხულიგნობა (ადმინისტრაციული)"),
        ("tax:a81", "საშემოსავლო გადასახადის განაკვეთი"),
    ]

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    for nid, label in test_ids:
        cur.execute(
            "SELECT title, length(body_text) AS blen, substr(body_text, 1, 80) AS preview FROM nodes WHERE node_id = ?",
            (nid,),
        )
        row = cur.fetchone()
        if row and row["blen"] > 0:
            preview_clean = row["preview"].replace("\n", " ")
            print(f"  ✅ {nid:<20} | {row['blen']:>4} სიმბოლო | {row['title'][:42]} -> '{preview_clean}...'")
        else:
            print(f"  ❌ {nid:<20} | ვერ მოიძებნა ან ცარიელია! ({label})")

    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Georgian Legal AI Dual Indexer")
    parser.add_argument("--sqlite-only", action="store_true", help="მხოლოდ SQLite + FTS5 ბაზის აგება")
    parser.add_argument("--rebuild", action="store_true", help="ChromaDB-ის ნულიდან თავიდან აგება")
    args = parser.parse_args()

    articles = load_all_ast_articles()
    print(f"[*] ჩატვირთულია {len(articles)} სუფთა მოქმედი მუხლი 21 კოდექსიდან.")

    build_sqlite_database(articles)
    verify_critical_articles()

    if not args.sqlite_only:
        build_chroma_vector_store(articles, rebuild=args.rebuild)


if __name__ == "__main__":
    main()