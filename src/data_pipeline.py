"""
src/data_pipeline.py
=============================================================================
Georgian Legal AI — Modular, Hardware-Agnostic ETL & Indexing Pipeline
Replaces legacy step1_scraper.py, step2_ast_parser.py, and step3_indexer.py.
Features:
  - Step 1 (`--step scrape`): Downloads & validates 21 Georgian codes from matsne.gov.ge
  - Step 2 (`--step parse`):  Lossless DOM-First Legal AST Parser (preserves prime ^1..^75)
  - Step 3 (`--step index`):  Dual Storage Indexer (SQLite FTS5 Trigram + ChromaDB)
                              Works with ANY model configured in .env (GGUF, HF, or Google)
                              via Resumable Batch Processing + Critical Sanity Check.
Usage Examples:
  python src/data_pipeline.py --step all
  python src/data_pipeline.py --step scrape --codes criminal,civil,labor
  python src/data_pipeline.py --step index --sqlite-only
  python src/data_pipeline.py --step index --rebuild --batch-size 16
=============================================================================
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

import chromadb
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

_THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _THIS_DIR.parent if _THIS_DIR.name in ("src", "backend", "utils") else _THIS_DIR
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

load_dotenv(PROJECT_ROOT / ".env")

from hybrid_retriever import ModularEmbedder

RAW_HTML_DIR = Path(os.getenv("RAW_HTML_DIR", str(PROJECT_ROOT / "data" / "raw_html")))
MANIFEST_PATH = RAW_HTML_DIR / "manifest.json"
AST_OUT_DIR = Path(os.getenv("AST_OUT_DIR", str(PROJECT_ROOT / "data" / "processed_ast")))
AUDIT_REPORT_PATH = AST_OUT_DIR / "ast_audit_report.json"
DB_PATH = Path(os.getenv("SQLITE_DB_PATH", str(PROJECT_ROOT / "data" / "legal_database.db")))
CHROMA_DIR = Path(os.getenv("CHROMA_DB_DIR", str(PROJECT_ROOT / "data" / "chroma_db")))
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "georgian_legal_dense")

DEFAULT_BATCH_SIZE = int(os.getenv("INDEX_BATCH_SIZE", "16"))
MAX_EMBED_CHARS = int(os.getenv("MAX_EMBED_CHARS", "2800"))

GEORGIAN_CODES_REGISTRY: List[Dict[str, str]] = [
    {"code_slug": "constitution", "code_title": "საქართველოს კონსტიტუცია", "doc_id": "30346"},
    {"code_slug": "criminal", "code_title": "საქართველოს სისხლის სამართლის კოდექსი", "doc_id": "16426"},
    {"code_slug": "criminal_proc", "code_title": "საქართველოს სისხლის სამართლის საპროცესო კოდექსი", "doc_id": "90034"},
    {"code_slug": "civil", "code_title": "საქართველოს სამოქალაქო კოდექსი", "doc_id": "31702"},
    {"code_slug": "civil_proc", "code_title": "საქართველოს სამოქალაქო საპროცესო კოდექსი", "doc_id": "29962"},
    {"code_slug": "labor", "code_title": "საქართველოს შრომის კოდექსი", "doc_id": "1155567"},
    {"code_slug": "gen_admin", "code_title": "საქართველოს ზოგადი ადმინისტრაციული კოდექსი", "doc_id": "16270"},
    {"code_slug": "admin_proc", "code_title": "საქართველოს ადმინისტრაციული საპროცესო კოდექსი", "doc_id": "16492"},
    {"code_slug": "admin_offenses", "code_title": "საქართველოს ადმინისტრაციულ სამართალდარღვევათა კოდექსი", "doc_id": "28216"},
    {"code_slug": "tax", "code_title": "საქართველოს საგადასახადო კოდექსი", "doc_id": "1043717"},
    {"code_slug": "customs", "code_title": "საქართველოს საბაჟო კოდექსი", "doc_id": "4598501"},
    {"code_slug": "election", "code_title": "საქართველოს ორგანული კანონი „საქართველოს საარჩევნო კოდექსი“", "doc_id": "1557168"},
    {"code_slug": "imprisonment", "code_title": "პატიმრობის კოდექსი", "doc_id": "91634"},
    {"code_slug": "juvenile", "code_title": "არასრულწლოვანთა მართლმსაჯულების კოდექსი", "doc_id": "2877281"},
    {"code_slug": "child_rights", "code_title": "ბავშვის უფლებათა კოდექსი", "doc_id": "4613854"},
    {"code_slug": "spatial_building", "code_title": "საქართველოს სივრცითი დაგეგმარების, არქიტექტურული და სამშენებლო საქმიანობის კოდექსი", "doc_id": "4276845"},
    {"code_slug": "product_safety", "code_title": "პროდუქტის უსაფრთხოებისა და თავისუფალი მიმოქცევის კოდექსი", "doc_id": "1659419"},
    {"code_slug": "forest", "code_title": "საქართველოს ტყის კოდექსი", "doc_id": "4874066"},
    {"code_slug": "env_assessment", "code_title": "გარემოსდაცვითი შეფასების კოდექსი", "doc_id": "3691981"},
    {"code_slug": "air", "code_title": "საქართველოს საჰაერო კოდექსი", "doc_id": "33298"},
    {"code_slug": "maritime", "code_title": "საქართველოს საზღვაო კოდექსი", "doc_id": "32476"},
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/131.0.0.0 Safari/537.36",
    "Accept-Language": "ka-GE,ka;q=0.9,en-US;q=0.8,en;q=0.7",
}

SUPERSCRIPT_MAP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
BOOK_RE = re.compile(r"^\s*(წიგნი\s+[^\n]{1,120})$", re.IGNORECASE)
PART_RE = re.compile(r"^\s*(კარი\s+[^\n]{1,140})$", re.IGNORECASE)
CHAPTER_RE = re.compile(r"^\s*(თავი\s+[^\n]{1,160})$", re.IGNORECASE)
SUBCHAPTER_RE = re.compile(r"^\s*(ქვეთავი\s+[^\n]{1,160})$", re.IGNORECASE)
ARTICLE_HEADER_RE = re.compile(
    r"^\s*\[?\s*მუხლი\s+"
    r"(\d{1,4}(?:\s*(?:\^\s*\d{1,3}|[⁰¹²³⁴⁵⁶⁷⁸⁹]+|\(\d+\))|\s*-\s*\d{1,3})?)"
    r"\s*(?:\.|-|–|—)?\s*(.*)$",
    re.IGNORECASE,
)
AMENDMENT_LINE_RE = re.compile(
    r"^(?:საქართველოს\s+\d{4}\s+წლის|საქართველოს\s+ორგანული\s+კანონი\s+№|"
    r"საქართველოს\s+კანონი\s+№|საქართველოს\s+პარლამენტის\s+\d{4}\s+წლის|"
    r"საქართველოს\s+საკონსტიტუციო\s+სასამართლოს\s+\d{4}\s+წლის)",
    re.IGNORECASE,
)


# =============================================================================
# STEP 1: MATSNE HTML SCRAPER & VALIDATOR
# =============================================================================
def validate_matsne_html(html_text: str) -> Tuple[bool, int]:
    if not html_text or len(html_text) < 10_000:
        return False, 0
    soup = BeautifulSoup(html_text, "html.parser")
    maindoc = soup.find("div", id="maindoc")
    if not maindoc:
        return False, 0
    text_content = maindoc.get_text("\n", strip=True)
    matches = re.findall(r"(?:^|\n)\s*მუხლი\s+\d+", text_content)
    return len(matches) >= 5, len(matches)


def try_recover_missing_html(code_slug: str, search_hint: str) -> Optional[Path]:
    target_path = RAW_HTML_DIR / f"{code_slug}.html"
    if target_path.exists() and target_path.stat().st_size > 15_000:
        return target_path

    candidate_urls = [
        "https://matsne.gov.ge/ka/document/view/91634?publication=35",
        "https://matsne.gov.ge/ka/document/view/91634?publication=30",
    ]
    try:
        search_url = f"https://matsne.gov.ge/ka/document/search?query={quote(search_hint)}"
        r = requests.get(search_url, headers=HEADERS, timeout=30)
        if r.status_code == 200:
            found_ids = re.findall(r"/ka/document/view/(\d+)(?:\?publication=\d+)?", r.text)
            for fid in found_ids[:5]:
                candidate_urls.insert(0, f"https://matsne.gov.ge/ka/document/view/{fid}")
    except Exception:
        pass

    for url in candidate_urls:
        try:
            resp = requests.get(url, headers=HEADERS, timeout=45)
            if resp.status_code == 200:
                ok, _ = validate_matsne_html(resp.text)
                if ok:
                    target_path.write_text(resp.text, encoding="utf-8")
                    return target_path
        except Exception:
            continue
    return None


def run_step1_scraper(filter_codes: Optional[List[str]] = None) -> None:
    RAW_HTML_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    manifest = []

    registry = [
        c for c in GEORGIAN_CODES_REGISTRY
        if not filter_codes or c["code_slug"] in filter_codes
    ]

    print("=" * 90)
    print(f"🚀 [STEP 1] საქართველოს კანონმდებლობის ჩამოტვირთვა ({len(registry)} კოდექსი)...")
    print("=" * 90)

    for idx, item in enumerate(registry, 1):
        slug, title, doc_id = item["code_slug"], item["code_title"], item["doc_id"]
        target_file = RAW_HTML_DIR / f"{slug}.html"
        print(f"[{idx}/{len(registry)}] {title} ({slug})...")

        if target_file.exists():
            html_data = target_file.read_text(encoding="utf-8", errors="replace")
            ok, art_cnt = validate_matsne_html(html_data)
            if ok:
                print(f"    [✓] უკვე არსებობს (~{art_cnt} მუხლი).")
                manifest.append({**item, "file_path": str(target_file), "approx_articles": art_cnt, "status": "ready"})
                continue

        downloaded = False
        for url in (
            f"https://matsne.gov.ge/ka/document/view/{doc_id}",
            f"https://matsne.gov.ge/ka/document/view/{doc_id}?publication=0",
        ):
            try:
                resp = session.get(url, headers=HEADERS, timeout=60)
                if resp.status_code == 200:
                    resp.encoding = "utf-8"
                    ok, art_cnt = validate_matsne_html(resp.text)
                    if ok:
                        target_file.write_text(resp.text, encoding="utf-8")
                        manifest.append({**item, "file_path": str(target_file), "approx_articles": art_cnt, "status": "ready"})
                        print(f"    [✓] ჩამოიტვირთა (~{art_cnt} მუხლი).")
                        downloaded = True
                        break
            except Exception as exc:
                print(f"    [!] ქსელური შეცდომა: {exc}")

        if not downloaded:
            recovered = try_recover_missing_html(slug, title)
            if recovered:
                manifest.append({**item, "file_path": str(recovered), "status": "ready"})
                print("    [✓] აღდგენილია Matsne Search-ით.")
            else:
                manifest.append({**item, "file_path": None, "status": "failed"})
                print("    [❌] ჩამოტვირთვა ვერ მოხერხდა.")
        time.sleep(1.0)

    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[✓] Step 1 დასრულდა. მანიფესტი: {MANIFEST_PATH}")


# =============================================================================
# STEP 2: DOM-FIRST LEGAL AST PARSER
# =============================================================================
def normalize_unicode(text: str) -> str:
    if not text:
        return ""
    s = unicodedata.normalize("NFC", text).replace("\u00a0", " ").replace("\u200b", "").replace("\ufeff", "")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"(\d)\s*\^\s*(\d+)", r"\1^\2", s)
    s = re.sub(r"(\d)\s+([⁰¹²³⁴⁵⁶⁷⁸⁹]+)", r"\1\2", s)
    return re.sub(r"\s+\.", ".", s).strip()


def normalize_article_no(raw_num: str) -> str:
    s = normalize_unicode(raw_num)
    s = re.sub(r"\(\s*(\d+)\s*\)", r"^\1", s)
    if "^" not in s:
        m = re.match(r"^(\d+)([⁰¹²³⁴⁵⁶⁷⁸⁹]+)$", s)
        if m:
            s = f"{m.group(1)}^{m.group(2).translate(SUPERSCRIPT_MAP)}"
    else:
        s = s.translate(SUPERSCRIPT_MAP)
    return re.sub(r"\s+", "", s)


def parse_html_to_ast(html_path: Path, code_slug: str, code_title: str) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    soup = BeautifulSoup(html_path.read_text(encoding="utf-8", errors="replace"), "html.parser")
    maindoc = soup.find("div", id="maindoc") or soup

    for sup in maindoc.find_all("sup"):
        sup_txt = sup.get_text(strip=True)
        if sup_txt:
            sup.replace_with(f"^{sup_txt}")
    for br in maindoc.find_all("br"):
        br.replace_with("\n")
    for blk in maindoc.find_all(("p", "div", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote")):
        blk.insert_before("\n")
        blk.insert_after("\n")

    lines = [normalize_unicode(line) for line in maindoc.get_text(" ").splitlines() if normalize_unicode(line)]
    current_book = current_part = current_chapter = current_subchapter = ""
    raw_articles: List[Dict[str, Any]] = []
    curr_art: Optional[Dict[str, Any]] = None

    def flush():
        nonlocal curr_art
        if curr_art is not None:
            raw_articles.append(curr_art)
            curr_art = None

    for line in lines:
        m_art = ARTICLE_HEADER_RE.match(line)
        if m_art:
            raw_num, rest_title = m_art.group(1), m_art.group(2).strip()
            if (
                not rest_title.startswith(("ჯერ", "-ჯერ"))
                and not re.match(r"^(?:და|ან|მუხლის|მუხლში|მუხლით|ნაწილი|პუნქტი)\b", rest_title.casefold())
                and len(line) < 290
            ):
                flush()
                art_no = normalize_article_no(raw_num)
                full_title = f"მუხლი {art_no}. {rest_title}".strip() if rest_title else f"მუხლი {art_no}"
                curr_art = {
                    "code_slug": code_slug, "code_title": code_title,
                    "book": current_book, "part": current_part,
                    "chapter": current_chapter, "subchapter": current_subchapter,
                    "article_no": art_no, "title": full_title,
                    "body_lines": [], "amendments": [],
                }
                continue

        if len(line) < 200:
            if BOOK_RE.match(line):
                flush()
                current_book, current_part, current_chapter, current_subchapter = line, "", "", ""
                continue
            if PART_RE.match(line):
                flush()
                current_part, current_chapter, current_subchapter = line, "", ""
                continue
            if CHAPTER_RE.match(line):
                flush()
                current_chapter, current_subchapter = line, ""
                continue
            if SUBCHAPTER_RE.match(line):
                flush()
                current_subchapter = line
                continue

        if curr_art is not None:
            if AMENDMENT_LINE_RE.match(line):
                curr_art["amendments"].append(line)
            elif not curr_art["body_lines"] or curr_art["body_lines"][-1] != line:
                curr_art["body_lines"].append(line)

    flush()
    best_by_art_no: Dict[str, Dict[str, Any]] = {}
    repealed = 0

    for art in raw_articles:
        art_no, title = art["article_no"], art["title"]
        body = "\n".join(art["body_lines"]).strip()
        if "ამოღებულია" in title.casefold() or "ძალადაკარგულია" in title.casefold() or (
            len(body) < 95 and ("ამოღებულია" in body.casefold() or "ძალადაკარგულია" in body.casefold())
        ):
            repealed += 1
            continue
        cand = {
            "node_id": f"{code_slug}:a{art_no}",
            "code_slug": code_slug, "code_title": code_title,
            "book": art["book"], "part": art["part"],
            "chapter": art["chapter"], "subchapter": art["subchapter"],
            "article_no": art_no, "title": title,
            "body_text": body, "amendments": art["amendments"],
        }
        if art_no not in best_by_art_no or len(body) > len(best_by_art_no[art_no]["body_text"]):
            best_by_art_no[art_no] = cand

    final_articles = []
    for rec in best_by_art_no.values():
        if not rec["body_text"].strip():
            continue
        parts = [rec["code_title"], rec["book"], rec["part"], rec["chapter"], rec["subchapter"], rec["title"]]
        rec["breadcrumb"] = " > ".join(p for p in parts if p and p.strip())
        final_articles.append(rec)

    return final_articles, {
        "raw_detected": len(raw_articles),
        "repealed_dropped": repealed,
        "clean_active_articles": len(final_articles),
    }


def run_step2_parser(filter_codes: Optional[List[str]] = None) -> None:
    AST_OUT_DIR.mkdir(parents=True, exist_ok=True)
    registry = [
        c for c in GEORGIAN_CODES_REGISTRY
        if not filter_codes or c["code_slug"] in filter_codes
    ]

    print("\n" + "=" * 90)
    print("🛠️ [STEP 2] HTML -> Legal AST დაპარსვა (პრიმ-მუხლების ^1..^75 შენარჩუნებით)...")
    print("=" * 90)

    total_clean = 0
    audit_summary = []

    for item in registry:
        slug, title = item["code_slug"], item["code_title"]
        html_path = RAW_HTML_DIR / f"{slug}.html"
        if not html_path.exists():
            recovered = try_recover_missing_html(slug, title)
            if recovered:
                html_path = recovered
            else:
                print(f"  ❌ {slug:<18} -> HTML ვერ მოიძებნა")
                continue

        articles, stats = parse_html_to_ast(html_path, slug, title)
        out_file = AST_OUT_DIR / f"ast_{slug}.json"
        out_file.write_text(json.dumps(articles, ensure_ascii=False, indent=2), encoding="utf-8")
        total_clean += stats["clean_active_articles"]
        audit_summary.append({"code_slug": slug, "code_title": title, **stats})
        print(f"  ✅ {slug:<18} | აღმოჩენილი: {stats['raw_detected']:<4} | გაუქმებული: {stats['repealed_dropped']:<4} | სუფთა: {stats['clean_active_articles']}")

    AUDIT_REPORT_PATH.write_text(
        json.dumps({"total_codes": len(audit_summary), "total_clean_articles": total_clean, "codes": audit_summary}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[✓] Step 2 დასრულდა. სულ სუფთა მოქმედი მუხლები: {total_clean}")


# =============================================================================
# STEP 3: MODULAR DUAL INDEXER (SQLite FTS5 + ChromaDB Batch Embedder)
# =============================================================================
def verify_critical_articles() -> None:
    print("\n🔬 [SANITY CHECK] კრიტიკული მუხლების გადამოწმება SQLite ბაზაში:")
    test_ids = [
        "criminal:a108", "criminal:a140", "criminal:a144^1", "criminal:a126",
        "civil:a1108", "labor:a31", "constitution:a1", "admin_offenses:a166", "tax:a81",
    ]
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    for nid in test_ids:
        row = cur.execute("SELECT title, length(body_text) AS blen FROM nodes WHERE node_id = ?", (nid,)).fetchone()
        if row and row["blen"] > 0:
            print(f"  ✅ {nid:<20} | {row['blen']:>4} სიმბოლო | {row['title'][:50]}")
        else:
            print(f"  ❌ {nid:<20} | ვერ მოიძებნა ან ცარიელია!")
    conn.close()


def run_step3_indexer(sqlite_only: bool = False, rebuild: bool = False, batch_size: int = DEFAULT_BATCH_SIZE) -> None:
    ast_files = sorted(p for p in AST_OUT_DIR.glob("ast_*.json") if p.name != "ast_audit_report.json")
    if not ast_files:
        raise FileNotFoundError(f"AST ფაილები ვერ მოიძებნა {AST_OUT_DIR}-ში. ჯერ გაუშვით --step parse")

    articles: List[Dict[str, Any]] = []
    seen = set()
    for fpath in ast_files:
        for r in json.loads(fpath.read_text(encoding="utf-8")):
            if isinstance(r, dict) and r.get("node_id") and r["node_id"] not in seen and r.get("body_text", "").strip():
                seen.add(r["node_id"])
                articles.append(r)

    print("\n" + "=" * 90)
    print(f"🗄️ [STEP 3.1] იგება SQLite + FTS5 Trigram ბაზა ({len(articles)} მუხლი): {DB_PATH}")
    print("=" * 90)

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()

    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=NORMAL;")
    cur.execute("""
        CREATE TABLE nodes (
            node_id TEXT PRIMARY KEY, code_slug TEXT NOT NULL, code_title TEXT NOT NULL,
            book TEXT, part TEXT, chapter TEXT, subchapter TEXT,
            article_no TEXT NOT NULL, title TEXT NOT NULL, body_text TEXT NOT NULL,
            breadcrumb TEXT NOT NULL, amendments TEXT
        );
    """)
    cur.execute("CREATE INDEX idx_code_article ON nodes(code_slug, article_no);")
    cur.execute("CREATE INDEX idx_article_only ON nodes(article_no);")
    cur.execute("CREATE INDEX idx_code_slug ON nodes(code_slug);")
    cur.execute("""
        CREATE VIRTUAL TABLE nodes_fts USING fts5(
            node_id UNINDEXED, title, body_text, breadcrumb, tokenize='trigram'
        );
    """)

    cur.executemany(
        "INSERT INTO nodes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [(
            a["node_id"], a["code_slug"], a["code_title"], a.get("book", ""), a.get("part", ""),
            a.get("chapter", ""), a.get("subchapter", ""), a["article_no"], a["title"],
            a["body_text"], a["breadcrumb"], json.dumps(a.get("amendments", []), ensure_ascii=False),
        ) for a in articles],
    )
    cur.executemany(
        "INSERT INTO nodes_fts VALUES (?, ?, ?, ?)",
        [(a["node_id"], a["title"], a["body_text"], a["breadcrumb"]) for a in articles],
    )
    conn.commit()
    conn.close()
    print(f"[✓] SQLite ბაზა მზადაა ({len(articles)} მუხლი).")
    verify_critical_articles()

    if sqlite_only:
        return

    print("\n" + "=" * 90)
    print(f"🧠 [STEP 3.2] იგება ChromaDB ვექტორული საცავი (Provider={os.getenv('EMBED_PROVIDER', 'gguf')}): {CHROMA_DIR}")
    print("=" * 90)

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if rebuild:
        try:
            client.delete_collection(COLLECTION_NAME)
            print("[*] ძველი ChromaDB კოლექცია წაიშალა (--rebuild).")
        except Exception:
            pass

    collection = client.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    existing_ids = set(collection.get(include=[]).get("ids") or []) if collection.count() > 0 else set()
    pending = [a for a in articles if a["node_id"] not in existing_ids]

    if not pending:
        print(f"[✓] ყველა მუხლი ({collection.count()} ვექტორი) უკვე დაინდექსებულია ChromaDB-ში!")
        return

    print(f"[*] სულ: {len(articles)} | უკვე ბაზაშია: {len(existing_ids)} | დასაინდექსებელია: {len(pending)}")
    embedder = ModularEmbedder()
    t_start = time.perf_counter()
    total_pending = len(pending)

    for start_idx in range(0, total_pending, batch_size):
        batch = pending[start_idx : start_idx + batch_size]
        batch_ids = [a["node_id"] for a in batch]
        batch_docs = [f"{a['breadcrumb']}\n\n{a['body_text']}".strip() for a in batch]
        batch_metas = [{
            "node_id": a["node_id"], "code_slug": a["code_slug"],
            "code_title": a["code_title"], "article_no": a["article_no"],
            "title": a["title"], "breadcrumb": a["breadcrumb"],
        } for a in batch]

        batch_embs = embedder.embed_documents(batch_docs, max_chars=MAX_EMBED_CHARS)

        collection.upsert(
            ids=batch_ids,
            documents=batch_docs,
            metadatas=batch_metas,
            embeddings=batch_embs,
        )

        done = min(start_idx + batch_size, total_pending)
        if done % (batch_size * 4) == 0 or done == total_pending:
            elapsed = time.perf_counter() - t_start
            rate = done / elapsed if elapsed > 0 else 0.0
            eta = (total_pending - done) / rate if rate > 0 else 0.0
            print(f"  -> [{done:>4}/{total_pending}] ({done/total_pending*100:>5.1f}%) | სიჩქარე: {rate:.1f} მუხლი/წმ | ETA: ~{eta:.0f}წმ")

    embedder.close()
    print(f"[🚀] ChromaDB ინდექსაცია დასრულდა! სულ კოლექციაშია: {collection.count()} ვექტორი.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Georgian Legal AI — Modular Data Pipeline")
    parser.add_argument("--step", choices=["all", "scrape", "parse", "index"], default="all")
    parser.add_argument("--codes", type=str, default="", help="კონკრეტული კოდექსების სია მძიმით (მაგ: criminal,civil)")
    parser.add_argument("--sqlite-only", action="store_true", help="მხოლოდ SQLite + FTS5 ბაზის აგება")
    parser.add_argument("--rebuild", action="store_true", help="ChromaDB კოლექციის ნულიდან გადაწყობა")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch ზომა ინდექსაციისას")
    args = parser.parse_args()

    filter_list = [x.strip() for x in args.codes.split(",") if x.strip()] if args.codes else None

    if args.step in ("all", "scrape"):
        run_step1_scraper(filter_codes=filter_list)
    if args.step in ("all", "parse"):
        run_step2_parser(filter_codes=filter_list)
    if args.step in ("all", "index"):
        run_step3_indexer(sqlite_only=args.sqlite_only, rebuild=args.rebuild, batch_size=args.batch_size)