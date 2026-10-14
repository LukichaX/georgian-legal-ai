"""
step2_ast_parser.py
=============================================================================
Georgian Legal AI (v4 Production Pipeline) — ნაბიჯი 2 (v2.1 Fixed)
უნივერსალური, უდანაკარგო HTML -> Legal AST პარსერი და ვალიდატორი.
- ასწორებს ინლაინ ტეგების (<a>, <sup>, <span>) გაწყვეტას, რათა ყველა პრიმ-მუხლი
  (მაგ. 126^1, 144^1, 21^1..21^75) 100%-ით ცალ-ცალკე შენარჩუნდეს.
- მყარად აბამს data/ საქაღალდეს პროექტის მთავარ დირექტორიას (PROJECT_ROOT).
=============================================================================
"""

from __future__ import annotations

import json
import re
import shutil
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup

# პროექტის მთავარი საქაღალდის ავტომატური განსაზღვრა (როგორც root-იდან, ისე src/-დან გაშვებისას)
_THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _THIS_DIR.parent if _THIS_DIR.name == "src" else _THIS_DIR

RAW_HTML_DIR = PROJECT_ROOT / "data" / "raw_html"
MANIFEST_PATH = RAW_HTML_DIR / "manifest.json"
AST_OUT_DIR = PROJECT_ROOT / "data" / "processed_ast"
AUDIT_REPORT_PATH = AST_OUT_DIR / "ast_audit_report.json"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ka-GE,ka;q=0.9,en-US;q=0.8,en;q=0.7",
}

SUPERSCRIPT_MAP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")

BOOK_RE = re.compile(r"^\s*(წიგნი\s+[^\n]{1,120})$", re.IGNORECASE)
PART_RE = re.compile(r"^\s*(კარი\s+[^\n]{1,140})$", re.IGNORECASE)
CHAPTER_RE = re.compile(r"^\s*(თავი\s+[^\n]{1,160})$", re.IGNORECASE)
SUBCHAPTER_RE = re.compile(r"^\s*(ქვეთავი\s+[^\n]{1,160})$", re.IGNORECASE)

# იჭერს: "მუხლი 108.", "მუხლი 144^1.", "მუხლი 21^69.", "მუხლი 42^10."
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


def _consolidate_src_data_if_needed():
    """თუ წინა გაშვებისას ფაილები src/data/-ში ჩაიწერა, გადმოაქვს მთავარ data/-ში."""
    alt_raw = _THIS_DIR / "data" / "raw_html"
    if _THIS_DIR.name == "src" and alt_raw.exists() and not MANIFEST_PATH.exists():
        RAW_HTML_DIR.mkdir(parents=True, exist_ok=True)
        for f in alt_raw.glob("*"):
            target = RAW_HTML_DIR / f.name
            if not target.exists():
                shutil.copy2(f, target)


def normalize_unicode(text: str) -> str:
    if not text:
        return ""
    s = unicodedata.normalize("NFC", text)
    s = s.replace("\u00a0", " ").replace("\u200b", "").replace("\ufeff", "")
    s = re.sub(r"[ \t]+", " ", s)
    # შევაერთოთ გაწყვეტილი პრიმ-ინდექსები: "144 ^1" -> "144^1", "144 ^ 1" -> "144^1"
    s = re.sub(r"(\d)\s*\^\s*(\d+)", r"\1^\2", s)
    s = re.sub(r"(\d)\s+([⁰¹²³⁴⁵⁶⁷⁸⁹]+)", r"\1\2", s)
    s = re.sub(r"\s+\.", ".", s)
    return s.strip()


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


def is_repealed_article(title: str, body_text: str) -> bool:
    t_low = title.casefold()
    b_low = body_text.casefold().strip()
    if "ამოღებულია" in t_low or "ძალადაკარგულია" in t_low:
        return True
    if len(b_low) < 95 and ("ამოღებულია" in b_low or "ძალადაკარგულია" in b_low):
        return True
    return False


def try_recover_missing_html(code_slug: str, search_hint: str) -> Optional[Path]:
    target_path = RAW_HTML_DIR / f"{code_slug}.html"
    if target_path.exists() and target_path.stat().st_size > 15_000:
        return target_path

    print(f"  [*] მიმდინარეობს '{search_hint}'-ის მოძიება matsne.gov.ge-ზე...")
    candidate_urls = [
        "https://matsne.gov.ge/ka/document/view/91634?publication=35",
        "https://matsne.gov.ge/ka/document/view/91634?publication=30",
        "https://matsne.gov.ge/ka/document/view/91634?publication=25",
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
            if resp.status_code == 200 and 'id="maindoc"' in resp.text:
                if len(re.findall(r"მუხლი\s+\d+", resp.text)) >= 10:
                    target_path.write_text(resp.text, encoding="utf-8")
                    return target_path
        except Exception:
            continue
    return None


def extract_clean_lines_from_maindoc(soup: BeautifulSoup) -> List[str]:
    """
    აგროვებს ბლოკურ სტრიქონებს ისე, რომ ინლაინ ტეგები (<a>, <sup>, <span>)
    არასდროს გაწყდეს სხვადასხვა ხაზზე!
    """
    maindoc = soup.find("div", id="maindoc") or soup

    # 1. <sup>N</sup> გადავაქციოთ ^N ტექსტად
    for sup in maindoc.find_all("sup"):
        sup_txt = sup.get_text(strip=True)
        if sup_txt:
            sup.replace_with(f"^{sup_txt}")

    # 2. <br> გადავაქციოთ ახალ ხაზად
    for br in maindoc.find_all("br"):
        br.replace_with("\n")

    # 3. ყველა ბლოკურ ტეგს დავუსვათ ხაზის გამყოფი (\n), ხოლო ინლაინ ტეგები შევაერთოთ ჰარით (" ")
    block_tags = ("p", "div", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote")
    for blk in maindoc.find_all(block_tags):
        blk.insert_before("\n")
        blk.insert_after("\n")

    raw_text = maindoc.get_text(" ")
    lines: List[str] = []
    for raw_line in raw_text.splitlines():
        clean = normalize_unicode(raw_line)
        if clean:
            lines.append(clean)
    return lines


def parse_html_to_ast(
    html_path: Path,
    code_slug: str,
    code_title: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    html_text = html_path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(html_text, "html.parser")
    lines = extract_clean_lines_from_maindoc(soup)

    current_book = ""
    current_part = ""
    current_chapter = ""
    current_subchapter = ""

    raw_articles: List[Dict[str, Any]] = []
    curr_art: Optional[Dict[str, Any]] = None

    def flush_current_article():
        nonlocal curr_art
        if curr_art is not None:
            raw_articles.append(curr_art)
            curr_art = None

    for line in lines:
        m_art = ARTICLE_HEADER_RE.match(line)
        if m_art:
            raw_num = m_art.group(1)
            rest_title = m_art.group(2).strip()

            # გამოვრიცხოთ ზმნიზედა "-ჯერ" და გრძელი წინადადებები, სადაც "მუხლი" შუაში ან კაზუსშია
            if (
                not rest_title.startswith("ჯერ")
                and not rest_title.startswith("-ჯერ")
                and not re.match(r"^(?:და|ან|მუხლის|მუხლში|მუხლით|ნაწილი|პუნქტი)\b", rest_title.casefold())
                and len(line) < 290
            ):
                flush_current_article()
                art_no = normalize_article_no(raw_num)
                full_title = f"მუხლი {art_no}. {rest_title}".strip() if rest_title else f"მუხლი {art_no}"

                curr_art = {
                    "code_slug": code_slug,
                    "code_title": code_title,
                    "book": current_book,
                    "part": current_part,
                    "chapter": current_chapter,
                    "subchapter": current_subchapter,
                    "article_no": art_no,
                    "title": full_title,
                    "body_lines": [],
                    "amendments": [],
                }
                continue

        if len(line) < 200:
            if BOOK_RE.match(line):
                flush_current_article()
                current_book = line
                current_part = ""
                current_chapter = ""
                current_subchapter = ""
                continue
            if PART_RE.match(line):
                flush_current_article()
                current_part = line
                current_chapter = ""
                current_subchapter = ""
                continue
            if CHAPTER_RE.match(line):
                flush_current_article()
                current_chapter = line
                current_subchapter = ""
                continue
            if SUBCHAPTER_RE.match(line):
                flush_current_article()
                current_subchapter = line
                continue

        if curr_art is not None:
            if AMENDMENT_LINE_RE.match(line):
                curr_art["amendments"].append(line)
            else:
                if not curr_art["body_lines"] or curr_art["body_lines"][-1] != line:
                    curr_art["body_lines"].append(line)

    flush_current_article()

    best_by_art_no: Dict[str, Dict[str, Any]] = {}
    repealed_count = 0

    for art in raw_articles:
        art_no = art["article_no"]
        title = art["title"]
        body_text = "\n".join(art["body_lines"]).strip()

        if is_repealed_article(title, body_text):
            repealed_count += 1
            continue

        candidate = {
            "node_id": f"{code_slug}:a{art_no}",
            "code_slug": code_slug,
            "code_title": code_title,
            "book": art["book"],
            "part": art["part"],
            "chapter": art["chapter"],
            "subchapter": art["subchapter"],
            "article_no": art_no,
            "title": title,
            "body_text": body_text,
            "amendments": art["amendments"],
        }

        existing = best_by_art_no.get(art_no)
        if existing is None or len(body_text) > len(existing["body_text"]):
            best_by_art_no[art_no] = candidate

    final_articles: List[Dict[str, Any]] = []
    empty_dropped = 0

    for art_no, rec in best_by_art_no.items():
        if not rec["body_text"].strip():
            empty_dropped += 1
            continue

        hierarchy_parts = [
            rec["code_title"],
            rec["book"],
            rec["part"],
            rec["chapter"],
            rec["subchapter"],
            rec["title"],
        ]
        rec["breadcrumb"] = " > ".join(p for p in hierarchy_parts if p and p.strip())
        final_articles.append(rec)

    stats = {
        "raw_detected": len(raw_articles),
        "repealed_dropped": repealed_count,
        "empty_toc_dropped": empty_dropped,
        "clean_active_articles": len(final_articles),
    }
    return final_articles, stats


def main():
    _consolidate_src_data_if_needed()
    AST_OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"მანიფესტი ვერ მოიძებნა: {MANIFEST_PATH}. ჯერ გაუშვით step1_scraper.py")

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    print("=" * 100)
    print("🛠️ ნაბიჯი 2 (v2.1): ყველა კოდექსის დაპარსვა პრიმ-მუხლების (144^1, 126^1) სრული შენარჩუნებით...")
    print("=" * 100)
    print(f"{'კოდექსი (Slug)':<20} | {'აღმოჩენილი':<12} | {'გაუქმებული':<12} | {'სუფთა მოქმედი':<15} | {'სტატუსი'}")
    print("-" * 100)

    audit_summary: List[Dict[str, Any]] = []
    total_clean_articles = 0

    for item in manifest:
        slug = item["code_slug"]
        title = item["code_title"]
        html_path = RAW_HTML_DIR / f"{slug}.html"

        if not html_path.exists():
            recovered_path = try_recover_missing_html(slug, item.get("search_hint", title))
            if recovered_path:
                html_path = recovered_path
            else:
                print(f"{slug:<20} | {'—':<12} | {'—':<12} | {'0':<15} | ❌ HTML ვერ მოიძებნა")
                continue

        articles, stats = parse_html_to_ast(html_path, slug, title)
        out_file = AST_OUT_DIR / f"ast_{slug}.json"
        out_file.write_text(json.dumps(articles, ensure_ascii=False, indent=2), encoding="utf-8")

        total_clean_articles += stats["clean_active_articles"]
        print(
            f"{slug:<20} | {stats['raw_detected']:<12} | "
            f"{stats['repealed_dropped']:<12} | "
            f"{stats['clean_active_articles']:<15} | ✅ შენახულია ({out_file.name})"
        )

        audit_summary.append({
            "code_slug": slug,
            "code_title": title,
            **stats,
            "ast_file": str(out_file),
        })

    report_data = {
        "total_codes_parsed": len(audit_summary),
        "total_clean_active_articles": total_clean_articles,
        "codes": audit_summary,
    }
    AUDIT_REPORT_PATH.write_text(json.dumps(report_data, ensure_ascii=False, indent=2), encoding="utf-8")

    print("-" * 100)
    print(f"[🚀] ნაბიჯი 2 წარმატებით დასრულდა!")
    print(f"[📊] სულ დამუშავდა: {len(audit_summary)} კოდექსი | სუფთა მოქმედი მუხლების ჯამი: {total_clean_articles} მუხლი.")
    print(f"[📁] AST საქაღალდე: {AST_OUT_DIR}")
    print("=" * 100)


if __name__ == "__main__":
    main()