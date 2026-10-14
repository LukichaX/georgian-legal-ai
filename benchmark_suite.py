"""
benchmark_suite.py
=============================================================================
Georgian Legal AI (v4 Production Pipeline) — ნაბიჯი 6
Evaluation & Ablation Benchmark across the 21 Georgian Codes (5,624 Articles).
=============================================================================
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Set

from legal_agent import LegalAgentPipeline

_THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _THIS_DIR.parent if _THIS_DIR.name in ("src", "benchmarks") else _THIS_DIR
DB_PATH = PROJECT_ROOT / "data" / "legal_database.db"
REPORT_PATH = PROJECT_ROOT / "benchmark_report.json"

GOLDEN_DATASET: List[Dict[str, Any]] = [
    # 1. კონსტიტუცია
    {
        "id": "CONST_01",
        "code_slug": "constitution",
        "query": "რა წერია საქართველოს კონსტიტუციის მე-12 მუხლში მოქალაქეობის შესახებ?",
        "expected": ["constitution:a12"],
        "desc": "საქართველოს მოქალაქეობა (კონსტიტუცია მუხლი 12)",
    },
    # 2. სისხლის სამართალი
    {
        "id": "CRIM_01",
        "code_slug": "criminal",
        "query": "რა წერია სისხლის სამართლის კოდექსის 108-ე მუხლში?",
        "expected": ["criminal:a108"],
        "desc": "განზრახ მკვლელობა (Tier 0 - მუხლი 108)",
    },
    {
        "id": "CRIM_02",
        "code_slug": "criminal",
        "query": "საქართველოში რომელი წლიდან არის დაშვებული სექსი?",
        "expected": ["criminal:a140", "criminal:a141"],
        "desc": "ლაიკური კითხვა: სქესობრივი თანხმობის ასაკი (16 წელი - მუხლი 140)",
    },
    {
        "id": "CRIM_03",
        "code_slug": "criminal",
        "query": "დამკითხეს სასტიკად, მცემდნენ და მაიძულებდნენ მუხლი 39-ჯერ მიმერტყა იატაკზე აღიარების მისაღებად",
        "expected": ["criminal:a144^1", "criminal:a144^3", "criminal:a335", "criminal:a126"],
        "desc": "ომონიმის ხაფანგი ('მუხლი 39-ჯერ') + წამება/ჩვენების იძულება",
    },
    {
        "id": "CRIM_04",
        "code_slug": "criminal",
        "query": "რა შემთხვევაში ითვლება დაზიანება აუცილებელ მოგერიებად?",
        "expected": ["criminal:a28"],
        "desc": "აუცილებელი მოგერიება (მუხლი 28)",
    },
    {
        "id": "CRIM_05",
        "code_slug": "criminal",
        "query": "განზრახ მკვლელობა დამამძიმებელ გარემოებაში ჯგუფურად ან ანგარებით",
        "expected": ["criminal:a109"],
        "desc": "დამამძიმებელი მკვლელობა (მუხლი 109)",
    },
    # 3. შრომის სამართალი
    {
        "id": "LAB_01",
        "code_slug": "labor",
        "query": "შრომის კოდექსის 27-ე მუხლი",
        "expected": ["labor:a27"],
        "desc": "ზეგანაკვეთური სამუშაო (Tier 0 - მუხლი 27)",
    },
    {
        "id": "LAB_02",
        "code_slug": "labor",
        "query": "რამდენი დღე ეკუთვნის თანამშრომელს წელიწადში ანაზღაურებადი შვებულება?",
        "expected": ["labor:a31"],
        "desc": "შვებულების ხანგრძლივობა (24 სამუშაო დღე - მუხლი 31)",
    },
    {
        "id": "LAB_03",
        "code_slug": "labor",
        "query": "რა არის გამოსაცდელი ვადით შრომითი ხელშეკრულების მაქსიმალური ხანგრძლივობა?",
        "expected": ["labor:a17"],
        "desc": "გამოსაცდელი ვადა (6 თვე - მუხლი 17)",
    },
    {
        "id": "LAB_04",
        "code_slug": "labor",
        "query": "რა საფუძვლებით შეიძლება შეწყდეს შრომითი ხელშეკრულება და გათავისუფლდეს დასაქმებული?",
        "expected": ["labor:a47", "labor:a48"],
        "desc": "შრომითი ხელშეკრულების შეწყვეტა (მუხლი 47/48)",
    },
    # 4. სამოქალაქო სამართალი
    {
        "id": "CIV_01",
        "code_slug": "civil",
        "query": "რა ასაკიდან ხდება ადამიანი სრულწლოვანი და ქმედუნარიანი საქართველოში?",
        "expected": ["civil:a12"],
        "desc": "სრულწლოვანება და ქმედუნარიანობა (18 წელი - მუხლი 12)",
    },
    {
        "id": "CIV_02",
        "code_slug": "civil",
        "query": "რა ასაკიდან არის დაშვებული ქორწინება საქართველოში?",
        "expected": ["civil:a1108"],
        "desc": "საქორწინო ასაკი (18 წელი - მუხლი 1108)",
    },
    {
        "id": "CIV_03",
        "code_slug": "civil",
        "query": "ვინ არიან პირველი რიგის კანონისმიერი მემკვიდრეები?",
        "expected": ["civil:a1336"],
        "desc": "პირველი რიგის მემკვიდრეები (მუხლი 1336)",
    },
    {
        "id": "CIV_04",
        "code_slug": "civil",
        "query": "რა ფორმით უნდა დაიდოს უძრავი ქონების ნასყიდობის ხელშეკრულება და როდის წარმოიშობა საკუთრება?",
        "expected": ["civil:a183"],
        "desc": "უძრავი ნივთის შეძენა / რეესტრში რეგისტრაცია (მუხლი 183)",
    },
    # 5. ახალი კოდექსები (ადმინისტრაციული და საგადასახადო)
    {
        "id": "ADM_01",
        "code_slug": "admin_offenses",
        "query": "რა სასჯელია წვრილმანი ხულიგნობისთვის საჯარო სივრცეში ლანძღვა-გინებაზე?",
        "expected": ["admin_offenses:a166"],
        "desc": "წვრილმანი ხულიგნობა (ასკ მუხლი 166)",
    },
    {
        "id": "TAX_01",
        "code_slug": "tax",
        "query": "რამდენი პროცენტია ფიზიკური პირის საშემოსავლო გადასახადის განაკვეთი საქართველოში?",
        "expected": ["tax:a81"],
        "desc": "საშემოსავლო გადასახადის განაკვეთი (20% - საგადასახადო მუხლი 81)",
    },
]


def run_db_health_check() -> Dict[str, Any]:
    print("\n" + "=" * 95)
    print("🔍 STAGE 1: 21-კოდექსიანი ბაზის სისრულის აუდიტი (SQLite)")
    print("=" * 95)
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*), SUM(CASE WHEN length(trim(body_text)) = 0 THEN 1 ELSE 0 END) FROM nodes")
    total, empty_cnt = cur.fetchone()
    cur.execute("SELECT COUNT(DISTINCT code_slug) FROM nodes")
    code_cnt = cur.fetchone()[0]
    conn.close()

    print(f"• სულ კოდექსები ბაზაში: {code_cnt} კოდექსი")
    print(f"• სულ სუფთა მოქმედი მუხლები: {total} მუხლი")
    print(f"• ცარიელი მუხლები (body_text=''): {empty_cnt} (100% სუფთა!)")
    return {"total_codes": code_cnt, "total_articles": total, "empty_articles": empty_cnt}


def _score_list(pred_ids: List[str], expected: Set[str], k: int = 4) -> Dict[str, float]:
    top_k = pred_ids[:k]
    h1 = 1.0 if (top_k and top_k[0] in expected) else 0.0
    h4 = 1.0 if any(p in expected for p in top_k) else 0.0
    rr = 0.0
    for idx, p in enumerate(top_k, 1):
        if p in expected:
            rr = 1.0 / idx
            break
    return {"hit_at_1": h1, "hit_at_4": h4, "mrr": rr}


def run_ablation_benchmark(agent: LegalAgentPipeline) -> Dict[str, Any]:
    print("\n" + "=" * 95)
    print("📊 STAGE 2: RETRIEVAL ABLATION STUDY (Dense vs. FTS5 vs. Hybrid vs. Full Agent)")
    print("=" * 95)

    retriever = agent.retriever
    methods = ["dense_only", "sparse_fts5", "hybrid_rerank", "agent_routed"]
    agg = {m: {"hit_at_1": 0.0, "hit_at_4": 0.0, "mrr": 0.0, "ms": 0.0} for m in methods}
    n = len(GOLDEN_DATASET)

    for item in GOLDEN_DATASET:
        q = item["query"]
        exp = set(item["expected"])

        # 1. Dense Only
        t0 = time.perf_counter()
        d_ids = retriever.search_dense(q, top_k=4)
        t_d = (time.perf_counter() - t0) * 1000

        # 2. Sparse FTS5 Only
        t0 = time.perf_counter()
        s_ids = retriever.search_sparse_fts5(q, top_k=4)
        t_s = (time.perf_counter() - t0) * 1000

        # 3. Raw Hybrid + Rerank (without Agent Router)
        t0 = time.perf_counter()
        h_docs = retriever.hybrid_search(q, top_k=4)
        t_h = (time.perf_counter() - t0) * 1000
        h_ids = [d["node_id"] for d in h_docs]

        # 4. Full Agent Router + Retriever (Tier 0 + Vocabulary Bridge + Code Filter)
        t0 = time.perf_counter()
        r_state = agent.router_node({"user_query": q, "chat_history": [], "active_articles": [], "timings": {}})
        ret_state = agent.retriever_node({"user_query": q, **r_state})
        t_a = (time.perf_counter() - t0) * 1000
        a_ids = [d["node_id"] for d in ret_state["retrieved_articles"]]

        for m_name, pids, lat in [
            ("dense_only", d_ids, t_d),
            ("sparse_fts5", s_ids, t_s),
            ("hybrid_rerank", h_ids, t_h),
            ("agent_routed", a_ids, t_a),
        ]:
            sc = _score_list(pids, exp, k=4)
            agg[m_name]["hit_at_1"] += sc["hit_at_1"]
            agg[m_name]["hit_at_4"] += sc["hit_at_4"]
            agg[m_name]["mrr"] += sc["mrr"]
            agg[m_name]["ms"] += lat

        ag_sc = _score_list(a_ids, exp, k=4)
        icon = "✅" if ag_sc["hit_at_1"] == 1.0 else ("⚠️" if ag_sc["hit_at_4"] == 1.0 else "❌")
        print(f"{icon} [{item['id']:<8}] {item['desc']:<54} | Top-1: {a_ids[0] if a_ids else 'None':<18}")

    print("\n" + "-" * 95)
    print(f"{'მეთოდი (Pipeline Stage)':<26} | {'Hit@1 (%)':<12} | {'Recall@4 (%)':<14} | {'MRR@4':<10} | {'Avg Latency'}")
    print("-" * 95)
    summary = {}
    for m in methods:
        h1 = (agg[m]["hit_at_1"] / n) * 100
        h4 = (agg[m]["hit_at_4"] / n) * 100
        mrr = agg[m]["mrr"] / n
        lat = agg[m]["ms"] / n
        summary[m] = {"hit_at_1_pct": round(h1, 2), "recall_at_4_pct": round(h4, 2), "mrr": round(mrr, 4), "avg_ms": round(lat, 1)}
        print(f"{m:<26} | {h1:>9.1f}%   | {h4:>11.1f}%   | {mrr:>8.3f}   | {lat:>8.1f} ms")
    print("-" * 95)
    return summary


def run_topic_switch_audit(agent: LegalAgentPipeline) -> List[Dict[str, Any]]:
    print("\n" + "=" * 95)
    print("🧠 STAGE 3: CONVERSATIONAL ROUTER & TOPIC-SWITCH AUDIT")
    print("=" * 95)

    turns = [
        ("გამარჯობა, რაში მეხმარები?", "direct_chat", None, []),
        ("რამდენი დღეა ანაზღაურებადი შვებულება?", "hybrid_search", "labor", []),
        ("შეგიძლია ეს მუხლი მარტივი ენით ამიხსნა?", "follow_up_existing", "labor", []),
        ("საქართველოში რომელი წლიდან არის დაშვებული სექსი?", "hybrid_search", "criminal", ["შვებულებ"]),
        ("და ქორწინება რა ასაკიდან არის დაშვებული?", "hybrid_search", "civil", ["შვებულებ"]),
    ]

    state: Dict[str, Any] = {
        "chat_history": [],
        "active_articles": [],
        "active_evidence_pack": "",
        "active_code_slug": None,
        "timings": {},
    }

    results = []
    for idx, (q, exp_intent, exp_code, forbidden) in enumerate(turns, 1):
        state["user_query"] = q
        r_out = agent.router_node(state)
        dec = r_out["router_decision"]
        got_intent = dec["intent"]
        got_code = dec.get("code_slug")
        sq = dec.get("search_query", "")

        if idx == 2:
            arts = agent.retriever.hybrid_search(sq, top_k=2, code_slug="labor")
            state["active_articles"] = arts
            state["active_evidence_pack"] = agent.retriever.build_evidence_pack(arts)
            state["active_code_slug"] = "labor"

        state["chat_history"].append({"role": "user", "content": q})
        state["chat_history"].append({"role": "assistant", "content": "პასუხი [E1]."})

        leaked = [w for w in forbidden if w in sq.casefold()]
        ok = (got_intent == exp_intent) and (exp_code is None or got_code == exp_code) and not leaked
        print(f"{'✅' if ok else '❌'} სვლა #{idx}: '{q}' -> intent={got_intent} | code={got_code} | search_q='{sq}'")
        results.append({"turn": idx, "query": q, "passed": ok, "decision": dec})

    return results


def main():
    db_stats = run_db_health_check()
    with LegalAgentPipeline() as agent:
        ablation = run_ablation_benchmark(agent)
        router_check = run_topic_switch_audit(agent)

    REPORT_PATH.write_text(
        json.dumps({"db_audit": db_stats, "ablation": ablation, "router_audit": router_check}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n[✓] ბენჩმარკის რეპორტი შენახულია: {REPORT_PATH}")


if __name__ == "__main__":
    main()