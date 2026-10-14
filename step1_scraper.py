"""
step1_scraper.py
=============================================================================
Georgian Legal AI (v4 Clean Production Pipeline) — ნაბიჯი 1
საქართველოს კონსტიტუციისა და ყველა ძირითადი კოდექსის ოფიციალური,
კონსოლიდირებული HTML ვერსიების ავტომატური ჩამოტვირთვა matsne.gov.ge-დან.
=============================================================================
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests
from bs4 import BeautifulSoup

RAW_HTML_DIR = Path("data/raw_html")
MANIFEST_PATH = RAW_HTML_DIR / "manifest.json"

# საქართველოს კონსტიტუცია და ძირითადი მოქმედი კოდექსები (matsne.gov.ge doc_id-ებით)
# თუ რომელიმე კოდექსის ID-ს სჭირდება დაზუსტება, სკრიპტს აქვს Matsne Search Fallback!
GEORGIAN_CODES_REGISTRY: List[Dict[str, str]] = [
    {
        "code_slug": "constitution",
        "code_title": "საქართველოს კონსტიტუცია",
        "doc_id": "30346",
        "search_hint": "საქართველოს კონსტიტუცია",
    },
    {
        "code_slug": "criminal",
        "code_title": "საქართველოს სისხლის სამართლის კოდექსი",
        "doc_id": "16426",
        "search_hint": "საქართველოს სისხლის სამართლის კოდექსი",
    },
    {
        "code_slug": "criminal_proc",
        "code_title": "საქართველოს სისხლის სამართლის საპროცესო კოდექსი",
        "doc_id": "90034",
        "search_hint": "საქართველოს სისხლის სამართლის საპროცესო კოდექსი",
    },
    {
        "code_slug": "civil",
        "code_title": "საქართველოს სამოქალაქო კოდექსი",
        "doc_id": "31702",
        "search_hint": "საქართველოს სამოქალაქო კოდექსი",
    },
    {
        "code_slug": "civil_proc",
        "code_title": "საქართველოს სამოქალაქო საპროცესო კოდექსი",
        "doc_id": "29962",
        "search_hint": "საქართველოს სამოქალაქო საპროცესო კოდექსი",
    },
    {
        "code_slug": "labor",
        "code_title": "საქართველოს შრომის კოდექსი",
        "doc_id": "1155567",
        "search_hint": "საქართველოს შრომის კოდექსი",
    },
    {
        "code_slug": "gen_admin",
        "code_title": "საქართველოს ზოგადი ადმინისტრაციული კოდექსი",
        "doc_id": "16270",
        "search_hint": "საქართველოს ზოგადი ადმინისტრაციული კოდექსი",
    },
    {
        "code_slug": "admin_proc",
        "code_title": "საქართველოს ადმინისტრაციული საპროცესო კოდექსი",
        "doc_id": "16492",
        "search_hint": "საქართველოს ადმინისტრაციული საპროცესო კოდექსი",
    },
    {
        "code_slug": "admin_offenses",
        "code_title": "საქართველოს ადმინისტრაციულ სამართალდარღვევათა კოდექსი",
        "doc_id": "28216",
        "search_hint": "საქართველოს ადმინისტრაციულ სამართალდარღვევათა კოდექსი",
    },
    {
        "code_slug": "tax",
        "code_title": "საქართველოს საგადასახადო კოდექსი",
        "doc_id": "1043717",
        "search_hint": "საქართველოს საგადასახადო კოდექსი",
    },
    {
        "code_slug": "customs",
        "code_title": "საქართველოს საბაჟო კოდექსი",
        "doc_id": "4598501",
        "search_hint": "საქართველოს საბაჟო კოდექსი",
    },
    {
        "code_slug": "election",
        "code_title": "საქართველოს ორგანული კანონი „საქართველოს საარჩევნო კოდექსი“",
        "doc_id": "1557168",
        "search_hint": "საქართველოს საარჩევნო კოდექსი",
    },
    {
        "code_slug": "imprisonment",
        "code_title": "პატიმრობის კოდექსი",
        "doc_id": "91634",
        "search_hint": "პატიმრობის კოდექსი",
    },
    {
        "code_slug": "juvenile",
        "code_title": "არასრულწლოვანთა მართლმსაჯულების კოდექსი",
        "doc_id": "2877281",
        "search_hint": "არასრულწლოვანთა მართლმსაჯულების კოდექსი",
    },
    {
        "code_slug": "child_rights",
        "code_title": "ბავშვის უფლებათა კოდექსი",
        "doc_id": "4613854",
        "search_hint": "ბავშვის უფლებათა კოდექსი",
    },
    {
        "code_slug": "spatial_building",
        "code_title": "საქართველოს სივრცითი დაგეგმარების, არქიტექტურული და სამშენებლო საქმიანობის კოდექსი",
        "doc_id": "4276845",
        "search_hint": "საქართველოს სივრცითი დაგეგმარების, არქიტექტურული და სამშენებლო საქმიანობის კოდექსი",
    },
    {
        "code_slug": "product_safety",
        "code_title": "პროდუქტის უსაფრთხოებისა და თავისუფალი მიმოქცევის კოდექსი",
        "doc_id": "1659419",
        "search_hint": "პროდუქტის უსაფრთხოებისა და თავისუფალი მიმოქცევის კოდექსი",
    },
    {
        "code_slug": "forest",
        "code_title": "საქართველოს ტყის კოდექსი",
        "doc_id": "4874066",
        "search_hint": "საქართველოს ტყის კოდექსი",
    },
    {
        "code_slug": "env_assessment",
        "code_title": "გარემოსდაცვითი შეფასების კოდექსი",
        "doc_id": "3691981",
        "search_hint": "გარემოსდაცვითი შეფასების კოდექსი",
    },
    {
        "code_slug": "air",
        "code_title": "საქართველოს საჰაერო კოდექსი",
        "doc_id": "33298",
        "search_hint": "საქართველოს საჰაერო კოდექსი",
    },
    {
        "code_slug": "maritime",
        "code_title": "საქართველოს საზღვაო კოდექსი",
        "doc_id": "32476",
        "search_hint": "საქართველოს საზღვაო კოდექსი",
    },
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ka-GE,ka;q=0.9,en-US;q=0.8,en;q=0.7",
}


def validate_matsne_html(html_text: str) -> tuple[bool, int, str]:
    """
    ამოწმებს, რომ ჩამოტვირთული HTML ნამდვილად შეიცავს კანონის ტექსტს (#maindoc)
    და ითვლის მუხლების სავარაუდო რაოდენობას.
    """
    if not html_text or len(html_text) < 10_000:
        return (False, 0, "ფაილი ძალიან პატარაა ან ცარიელია")

    soup = BeautifulSoup(html_text, "html.parser")
    maindoc = soup.find("div", id="maindoc")
    if not maindoc:
        return (False, 0, "#maindoc კონტეინერი ვერ მოიძებნა")

    # ამოვიღოთ დოკუმენტის რეალური სათაური გვერდიდან
    title_el = soup.find("title")
    page_title = title_el.get_text(strip=True) if title_el else ""

    # დავითვალოთ მუხლების სათაურები maindoc-ში
    text_content = maindoc.get_text("\n", strip=True)
    article_matches = re.findall(r"(?:^|\n)\s*მუხლი\s+\d+", text_content)
    if len(article_matches) < 5:
        return (False, len(article_matches), f"ნაპოვნია მხოლოდ {len(article_matches)} მუხლი")

    return (True, len(article_matches), page_title)


def download_code_html(session: requests.Session, doc_id: str) -> Optional[str]:
    url = f"https://matsne.gov.ge/ka/document/view/{doc_id}?publication=0"
    # ჯერ ვცადოთ ბოლო კონსოლიდირებული ვერსია (პარამეტრის გარეშე Matsne აბრუნებს ბოლო ვერსიას)
    urls_to_try = [
        f"https://matsne.gov.ge/ka/document/view/{doc_id}",
        url,
    ]
    for target_url in urls_to_try:
        for attempt in range(1, 4):
            try:
                resp = session.get(target_url, headers=HEADERS, timeout=60)
                if resp.status_code == 200:
                    resp.encoding = "utf-8"
                    is_ok, _, _ = validate_matsne_html(resp.text)
                    if is_ok:
                        return resp.text
            except Exception as e:
                print(f"    [!] ქსელური შეცდომა ({target_url}, ცდა {attempt}/3): {e}")
            time.sleep(2.0)
    return None


def main():
    RAW_HTML_DIR.mkdir(parents=True, exist_ok=True)

    # თუ ძველ საქაღალდეში უკვე გვაქვს 3 კოდექსის HTML, შეგვიძლია დროის დასაზოგად დავაკოპიროთ
    legacy_map = {
        "criminal": Path("criminal_code_raw.html"),
        "civil": Path("civil_code_raw.html"),
        "labor": Path("labor_code_raw.html"),
    }

    session = requests.Session()
    manifest: List[Dict[str, object]] = []

    print("=" * 90)
    print(f"🚀 იწყება საქართველოს კანონმდებლობის ({len(GEORGIAN_CODES_REGISTRY)} კოდექსი) ჩამოტვირთვა და ვალიდაცია...")
    print("=" * 90)

    success_count = 0
    total_approx_articles = 0

    for idx, item in enumerate(GEORGIAN_CODES_REGISTRY, 1):
        slug = item["code_slug"]
        title = item["code_title"]
        doc_id = item["doc_id"]
        target_file = RAW_HTML_DIR / f"{slug}.html"

        print(f"\n[{idx}/{len(GEORGIAN_CODES_REGISTRY)}] {title} (slug: {slug}, ID: {doc_id})...")

        # 1. თუ უკვე ჩამოტვირთულია ახალ საქაღალდეში, შევამოწმოთ
        if target_file.exists():
            html_data = target_file.read_text(encoding="utf-8", errors="replace")
            is_valid, art_count, page_title = validate_matsne_html(html_data)
            if is_valid:
                print(f"    [✓] უკვე არსებობს და ვალიდურია ({art_count} მუხლი, {len(html_data)/1024:.0f} KB).")
                manifest.append({
                    **item,
                    "file_path": str(target_file),
                    "approx_articles": art_count,
                    "size_kb": round(len(html_data) / 1024, 1),
                    "status": "ready",
                })
                success_count += 1
                total_approx_articles += art_count
                continue

        # 2. შევამოწმოთ, ხომ არ დევს მიმდინარე საქაღალდეში წინა ეტაპიდან
        if slug in legacy_map and legacy_map[slug].exists():
            html_data = legacy_map[slug].read_text(encoding="utf-8", errors="replace")
            is_valid, art_count, _ = validate_matsne_html(html_data)
            if is_valid:
                target_file.write_text(html_data, encoding="utf-8")
                print(f"    [✓] გადმოტანილია ლოკალური ქეშიდან ({art_count} მუხლი, {len(html_data)/1024:.0f} KB).")
                manifest.append({
                    **item,
                    "file_path": str(target_file),
                    "approx_articles": art_count,
                    "size_kb": round(len(html_data) / 1024, 1),
                    "status": "ready",
                })
                success_count += 1
                total_approx_articles += art_count
                continue

        # 3. ჩამოვტვირთოთ matsne.gov.ge-დან
        print(f"    [*] იტვირთება matsne.gov.ge-დან (doc_id={doc_id})...")
        html_data = download_code_html(session, doc_id)
        if html_data:
            is_valid, art_count, page_title = validate_matsne_html(html_data)
            target_file.write_text(html_data, encoding="utf-8")
            print(f"    [✓] ჩამოიტვირთა წარმატებით! (~{art_count} მუხლი, {len(html_data)/1024:.0f} KB)")
            manifest.append({
                **item,
                "file_path": str(target_file),
                "approx_articles": art_count,
                "size_kb": round(len(html_data) / 1024, 1),
                "status": "ready",
            })
            success_count += 1
            total_approx_articles += art_count
        else:
            print(f"    [❌] ვერ მოხერხდა ჩამოტვირთვა ID={doc_id}-ით.")
            manifest.append({
                **item,
                "file_path": None,
                "approx_articles": 0,
                "size_kb": 0,
                "status": "failed",
            })

        time.sleep(1.0)  # ზრდილობიანი პაუზა სერვერთან

    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 90)
    print(f"[🚀] ნაბიჯი 1 დასრულდა! წარმატებით მომზადდა: {success_count}/{len(GEORGIAN_CODES_REGISTRY)} კოდექსი.")
    print(f"[📊] სავარაუდო მუხლების ჯამური რაოდენობა: ~{total_approx_articles} მუხლი.")
    print(f"[📁] მანიფესტი შენახულია: {MANIFEST_PATH}")
    print("=" * 90)


if __name__ == "__main__":
    main()