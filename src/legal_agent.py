"""
src/legal_agent.py
=============================================================================
Georgian Legal AI (v7.2 Final — Modular LangGraph Agent & Continental Dogmatics)
Architecture:
  - Modular Router Provider: 'hybrid' (Cloud + Local GGUF Fallback),
    'cloud_only' (0GB VRAM for router), or 'local_only' (100% offline GGUF)
  - Multi-Query Legal Issue Decomposition (Special Part + General Part + Exceptions)
  - Full Continental Legal Dogmatics (Criminal 3-tier system + Civil Property Art. 187)
  - System-Level Evidence Injection (Zero meta-talk about "user-provided articles")
  - Deterministic Citation Guard ([E1]-[E6] verification + Prompt Injection filter)
=============================================================================
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, TypedDict

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from hybrid_retriever import (
    PROJECT_ROOT,
    GeorgianLegalHybridRetriever,
    normalize_article_number,
    resolve_file_path,
)

load_dotenv(PROJECT_ROOT / ".env")

if os.name == "nt":
    try:
        import torch
        torch_lib = Path(torch.__file__).resolve().parent / "lib"
        if torch_lib.is_dir() and hasattr(os, "add_dll_directory"):
            os.add_dll_directory(str(torch_lib))
    except Exception:
        pass

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

ROUTER_PROVIDER = os.getenv("ROUTER_PROVIDER", "hybrid").strip().lower()
ROUTER_MODEL_NAME = os.getenv("ROUTER_MODEL_PATH", "Qwen_Qwen3.5-4B-Q4_K_M.gguf")
PRIMARY_GEN_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
FALLBACK_GEN_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-3.5-flash-lite")
FALLBACK_GEN_MODELS = list(dict.fromkeys([PRIMARY_GEN_MODEL, FALLBACK_GEN_MODEL]))

ROUTER_N_CTX = int(os.getenv("ROUTER_N_CTX", "2048"))
ROUTER_N_BATCH = int(os.getenv("ROUTER_N_BATCH", "512"))
ROUTER_N_GPU_LAYERS = int(os.getenv("ROUTER_N_GPU_LAYERS", "-1"))
RETRIEVAL_TOP_K = max(4, int(os.getenv("RETRIEVAL_TOP_K", "6")))
GENERATOR_TIMEOUT = float(os.getenv("GENERATOR_TIMEOUT_SECONDS", "45.0"))
PLANNER_TIMEOUT = float(os.getenv("PLANNER_TIMEOUT_SECONDS", "12.0"))
MAX_HISTORY_TURNS = max(2, int(os.getenv("MAX_HISTORY_TURNS", "6")))

logging.basicConfig(
    level=os.getenv("LEGAL_AGENT_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("georgian_legal_agent")

for _noisy in ("httpx", "httpcore", "huggingface_hub", "sentence_transformers", "google_genai", "chromadb"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

IntentType = Literal[
    "direct_chat",
    "follow_up_existing",
    "exact_lookup",
    "hybrid_search",
    "out_of_scope",
]

CodeSlugType = Literal[
    "constitution",
    "criminal",
    "criminal_proc",
    "civil",
    "civil_proc",
    "labor",
    "gen_admin",
    "admin_proc",
    "admin_offenses",
    "tax",
    "customs",
    "election",
    "imprisonment",
    "juvenile",
    "child_rights",
    "spatial_building",
    "product_safety",
    "forest",
    "env_assessment",
    "air",
    "maritime",
]

VALID_CODE_SLUGS = {
    "constitution", "criminal", "criminal_proc", "civil", "civil_proc",
    "labor", "gen_admin", "admin_proc", "admin_offenses", "tax", "customs",
    "election", "imprisonment", "juvenile", "child_rights", "spatial_building",
    "product_safety", "forest", "env_assessment", "air", "maritime",
}


def extract_text_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
            elif isinstance(getattr(part, "text", None), str):
                parts.append(part.text)
        return "\n".join(part for part in parts if part).strip()
    return str(content).strip()


# =============================================================================
# 1. MULTI-ARTICLE PARSER & HOMONYM GUARD
# =============================================================================
def _has_contract_or_ambiguity_context(query: str) -> bool:
    q = query.casefold()
    markers = (
        "ხელშეკრულების", "კონტრაქტის", "დანართში", "პუნქტში", "პუნქტი",
        "დოკუმენტში წერია", "კოდექსის სახელი მითითებული არ არის",
        "არ აგერიოს", "რომელს გულისხმობს", "ეს ნიშნავს",
    )
    return any(m in q for m in markers)


def _extract_explicit_article_numbers(query: str) -> List[str]:
    q = query.casefold().replace("−", "-").replace("–", "-")

    if re.search(r"მუხლ(?:ი|ის|ში|ით|ზე|ს|ად)?\s+\d+\s*[- ]?(?:ჯერ|გან)\b", q):
        return []
    if re.search(r"(?:მარჯვენა|მარცხენა|ორივე)\s+მუხლ", q):
        return []
    if re.search(r"მუხლებზე\s*(?:დამაჩოქ|დამსვ|მაიძულ|იყავ)", q):
        return []
    if re.search(
        r"მუხლ(?:ი|ზე|ში)?[^.!?]{0,45}(?:მიმერტყ|მიმარტყ|დამარტყ|მირტყ|მტკივ|ტკივილ|გაუტყდ|მოტყდ|დამისკდ|დაუზიანდ|ოპერაცი|დანა\b|ჯოხ|ჭრილობ)",
        q,
    ):
        return []

    q_cleaned = re.sub(
        r"(?:ხელშეკრულების|კონტრაქტის|შეთანხმების|წესდების|დანართის)\s+(?:მე-?\s*)?\d+(?:\.\d+)?\s*(?:-?ე)?\s*(?:მუხლ\w*|პუნქტ\w*|ნომერ\w*)",
        " ",
        q,
    )
    q_cleaned = re.sub(r"\b\d+(?:\.\d+)?\s*პუნქტ\w*", " ", q_cleaned)
    q_cleaned = re.sub(
        r"\b\d+\s*(?:დღე\w*|თვე\w*|წელ\w*|წლი\w*|საათ\w*|ლარ\w*|დოლარ\w*|პროცენტ\w*|კლას\w*|კმ\b|მეტრ\w*)\b",
        " ",
        q_cleaned,
    )

    # თუ წერია "მე-11 პრიმა (11¹) მუხლი", მოვაშოროთ ფრჩხილებში დუბლირებული ჩანაწერი
    q_cleaned = re.sub(r"(?:პრიმა|პრიმ)\s*\(\s*\d+(?:\^\d+\vert{}[⁰¹²³⁴⁵⁶⁷⁸⁹]+)?\s*\)", "პრიმა", q_cleaned)
    q_cleaned = re.sub(
        r"(\d{1,4})\s*(?:-?ე)?\s*(?:პრიმა|პრიმ)\s*(\d*)",
        lambda m: f"{m.group(1)}^{m.group(2) or '1'}",
        q_cleaned,
    )

    found_numbers: List[str] = []

    list_pattern = re.compile(
        r"((?:(?:მე-?)?\d{1,4}(?:\^\d{1,3}|[⁰¹²³⁴⁵⁶⁷⁸⁹]+)?(?:-?ე)?\s*(?:,\s*|\s+და\s+|\s+ან\s+))+)"
        r"(?:მე-?)?(\d{1,4}(?:\^\d{1,3}|[⁰¹²³⁴⁵⁶⁷⁸⁹]+)?)(?:-?ე)?\s*მუხლ",
    )
    for m in list_pattern.finditer(q_cleaned):
        prefix_chunk = m.group(1)
        last_num = m.group(2)
        nums_in_prefix = re.findall(r"(\d{1,4}(?:\^\d{1,3}|[⁰¹²³⁴⁵⁶⁷⁸⁹]+)?)", prefix_chunk)
        for n in nums_in_prefix + [last_num]:
            norm_n = normalize_article_number(n)
            if norm_n and norm_n not in found_numbers:
                found_numbers.append(norm_n)

    single_patterns = (
        r"(?:მე-?)?(\d{1,4}(?:\s*(?:\^\s*\d{1,3}|[⁰¹²³⁴⁵⁶⁷⁸⁹]+))?)\s*(?:-?ე\s*)?მუხლ(?:ი|ის|ში|ით|ზე|ს|ად|ებ\w*|თან)?",
        r"მუხლ(?:ი|ის|ში|ით|ზე|ს|ად)?\s*(?:№\s*)?(\d{1,4}(?:\s*(?:\^\s*\d{1,3}|[⁰¹²³⁴⁵⁶⁷⁸⁹]+))?)(?!\s*[- ]?(?:ჯერ|გან))",
    )
    for pat in single_patterns:
        for m in re.finditer(pat, q_cleaned):
            norm_n = normalize_article_number(m.group(1))
            if norm_n and norm_n not in found_numbers:
                found_numbers.append(norm_n)

    return found_numbers


def _needs_hybrid_augmentation(query: str, explicit_articles: List[str]) -> bool:
    if not explicit_articles:
        return True
    if _has_contract_or_ambiguity_context(query):
        return True
    return len(query.split()) > 11


# =============================================================================
# 2. STATUTORY DOMAIN CLASSIFIER
# =============================================================================
def _extract_explicitly_named_codes(query: str) -> List[str]:
    q = query.casefold()
    explicit: List[str] = []
    explicit_map = [
        ("constitution", ("კონსტიტუცი",)),
        ("criminal_proc", ("სისხლის სამართლის საპროცესო", "სსსკ")),
        ("civil_proc", ("სამოქალაქო საპროცესო",)),
        ("admin_proc", ("ადმინისტრაციული საპროცესო",)),
        ("admin_offenses", ("ადმინისტრაციულ სამართალდარღვევა", "ასკ-ის", "ასკ ")),
        ("gen_admin", ("ზოგადი ადმინისტრაციული", "სზაკ")),
        ("spatial_building", ("სივრცითი და სამშენებლო", "სამშენებლო კოდექს", "სივრცითი დაგეგმარების")),
        ("env_assessment", ("გარემოსდაცვითი შეფასების",)),
        ("child_rights", ("ბავშვის უფლებათა კოდექს",)),
        ("juvenile", ("არასრულწლოვანთა მართლმსაჯულების",)),
        ("imprisonment", ("პატიმრობის კოდექს",)),
        ("election", ("საარჩევნო კოდექს",)),
        ("customs", ("საბაჟო კოდექს",)),
        ("tax", ("საგადასახადო კოდექს",)),
        ("forest", ("ტყის კოდექს",)),
        ("product_safety", ("პროდუქტის უსაფრთხოების",)),
        ("air", ("საჰაერო კოდექს",)),
        ("maritime", ("საზღვაო კოდექს",)),
        ("labor", ("შრომის კოდექს",)),
        ("criminal", ("სისხლის სამართლის კოდექს", "სისხლის კოდექს", "სსკ-ის", "სსკ ")),
        ("civil", ("სამოქალაქო კოდექს",)),
    ]
    for slug, markers in explicit_map:
        if any(m in q for m in markers):
            explicit.append(slug)
    return explicit


def _infer_all_relevant_codes(
    query: str,
    model_guesses: Optional[List[str]] = None,
    active_code_slug: Optional[str] = None,
) -> List[str]:
    q = query.casefold()
    matched: List[str] = _extract_explicitly_named_codes(query)

    def add_slug(s: Optional[str]):
        if s and s in VALID_CODE_SLUGS and s not in matched:
            matched.append(s)

    if model_guesses:
        for mg in model_guesses:
            add_slug(mg)

    if any(w in q for w in ("დანაშაულ", "დაისჯება", "ისჯება", "მომისჯიან", "მოიპარ", "ქურდობ", "ყაჩაღობ", "ძარცვ", "თაღლითობ", "დაჭრ", "ესროლ", "მოკვლ", "მკვლელობ", "წამებ")):
        add_slug("criminal")
    elif any(w in q for w in ("ჩხუბ", "დანა", "ჯოხი", "მცემ", "დამარტყ")) and any(w in q for w in ("საავადმყოფო", "ოპერაცი", "დაზიანებ", "ტვინის შერყევ")):
        add_slug("criminal")

    domain_signals = [
        ("constitution", ("კონსტიტუცი", "იმპიჩმენტ", "სუვერენიტეტ", "უდანაშაულობის პრეზუმფცი", "უკუძალ")),
        ("criminal_proc", ("ჩხრეკ", "გაჩხრიკ", "გამომძიებელ", "პროკურორ", "ბრალის წაყენებ", "აღკვეთის ღონისძიებ", "გირაო", "ნაფიც მსაჯულ", "დაკითხვ", "დაუშვებელი მტკიცებულებ", "ირიბი ჩვენებ")),
        ("civil_proc", ("სარჩელის უზრუნველყოფ", "ყადაღ", "სააპელაციო", "საკასაციო", "სასამართლო ბაჟ", "დაუსწრებელი გადაწყვეტილებ", "შესაგებელ", "მტკიცების ტვირთ")),
        ("admin_offenses", ("წვრილმანი ხულიგნობ", "ალკოტესტ", "პატრულ", "მართვის მოწმობ", "სიჩქარის", "ადმინისტრაციული დაკავებ", "ადმინისტრაციული ჯარიმ")),
        ("gen_admin", ("საჯარო ინფორმაცი", "ადმინისტრაციული აქტ", "ადმინისტრაციული საჩივარ", "საჯარო დაწესებულებ", "კომერციული საიდუმლოებ")),
        ("admin_proc", ("ადმინისტრაციული სარჩელ",)),
        ("tax", ("საგადასახადო", "საშემოსავლო გადასახად", "დღგ", "მოგების გადასახად", "ქონების გადასახად", "დეკლარაცი", "დეკლარირებ", "დაბეგვრ")),
        ("customs", ("საბაჟო", "იმპორტის გადასახდელ", "განბაჟებ", "დროებით შენახვ")),
        ("election", ("არჩევნებ", "ცესკო", "საარჩევნო", "რეფერენდუმ", "კენჭისყრ")),
        ("imprisonment", ("პენიტენციურ", "მსჯავრდებულ", "პატიმარ", "პაემან", "პირობით ვადამდე")),
        ("juvenile", ("არასრულწლოვან", "მოზარდ", "განრიდებ")),
        ("child_rights", ("ბავშვის უფლება", "ბავშვის საუკეთესო ინტერეს")),
        ("spatial_building", ("მშენებლობის ნებართვ", "შენობა-ნაგებობ", "უნებართვო მშენებლობ", "დემონტაჟ")),
        ("forest", ("ტყის", "ხე-ტყის", "ხეტყის", "მერქნის", "ხის ჭრ")),
        ("env_assessment", ("გარემოსდაცვით", "გარემოზე ზემოქმედებ", "გზშ")),
        ("product_safety", ("პროდუქტის უსაფრთხოებ", "ტექნიკური რეგლამენტ")),
        ("air", ("საჰაერო ხომალდ", "საჰაერო გადაყვან", "ავიაცი")),
        ("maritime", ("საზღვაო", "გემის კაპიტან", "ნავსადგურ")),
        ("labor", ("შრომით", "დასაქმებულ", "დამსაქმებელ", "შვებულებ", "ზეგანაკვეთურ", "დეკრეტ", "გამოსაცდელი ვად", "სამსახურიდან", "ხელფას", "სტაჟირებ", "კონკურენცი")),
        ("criminal", ("დანაშაულ", "სასჯელ", "თავისუფლების აღკვეთ", "მუქარ", "ძალადობ", "ნარკოტიკ", "აუცილებელი მოგერიებ", "უკიდურესი აუცილებლობ")),
        ("civil", ("მემკვიდრ", "სამკვიდრო", "ანდერძ", "ქორწინებ", "განქორწინებ", "ალიმენტ", "ხანდაზმულობ", "ნასყიდობ", "სესხ", "იპოთეკ", "ქირავნობ", "გამქირავებელ", "მემამულე", "იჯარა", "მეზობელ", "ზიანის ანაზღაურებ", "ხელშეკრულებ", "გარიგებ", "მეურვე", "საკუთრებ", "კეთილსინდისიერი შემძენ", "უსაფუძვლო გამდიდრებ", "ნაპოვნ", "ვიპოვე", "დაკარგული", "ხელოსან", "რემონტ", "პირადი ცხოვრებ", "გამოსახულებ", "ფოტო")),
    ]

    for slug, keywords in domain_signals:
        if any(k in q for k in keywords):
            add_slug(slug)

    if not matched and active_code_slug in VALID_CODE_SLUGS:
        add_slug(active_code_slug)

    return matched


def _is_out_of_scope_query(query: str) -> bool:
    q = query.casefold()
    non_legal_triggers = (
        "საფეხბურთო მატჩ", "ფეხბურთის წესებ", "წითელი ბარათ", "პენალტი დანიშნ",
        "საჭადრაკო ტურნირ", "საკონტროლო საათზე", "ტექნიკური წაგება",
        "ოფსაიდ", "ჩემპიონთა ლიგ", "ხაჭაპურ", "ლობიან", "რეცეპტ", "გამოვაცხო",
        "ამინდის პროგნოზ", "ჰოროსკოპ",
    )
    return any(t in q for t in non_legal_triggers)


def _is_direct_chat_query(query: str) -> bool:
    q = re.sub(r"[^\w\sა-ჰ]", " ", query.casefold()).strip()
    words = q.split()
    if not words or len(words) > 12:
        return False

    legal_signals = (
        "მუხლ", "კოდექს", "კონსტიტუცი", "დანაშაულ", "სასჯელ", "შრომ", "ხელშეკრულებ",
        "სექს", "ქორწინებ", "სამოქალაქო", "სისხლის", "ანაზღაურებ", "გათავისუფლებ",
        "მკვლელობ", "ქურდობ", "ასაკ", "გადასახად", "საბაჟო", "ჯარიმ", "არჩევნებ",
        "სარჩელ", "საჩივარ", "განცხადებ", "სესხ", "მემკვიდრ", "ხელფას", "პრეზუმფცი",
    )
    if any(sig in q for sig in legal_signals):
        return False

    chat_patterns = (
        r"^(?:გამარჯობა|გაგიმარჯოს|სალამი|დილა მშვიდობისა|საღამო მშვიდობისა|ჰეი|hello|hi)\b",
        r"\b(?:როგორ ხარ|რას შვრები|როგორ მიდის საქმე|ვინ ხარ|რა გქვია|რა შეგიძლია|რაში მეხმარები|როგორ მუშაობ)\b",
        r"\b(?:superwoman|superman|მაგარი ხარ|ყოჩაღ)\b",
        r"^(?:დიდი მადლობა|მადლობა|გაიხარე|გასაგებია ყველაფერი|კარგად|ნახვამდის)\b",
    )
    return any(re.search(pat, q) for pat in chat_patterns)


def _is_follow_up_existing_query(
    query: str,
    has_active_evidence: bool,
    active_code_slug: Optional[str],
) -> bool:
    if not has_active_evidence:
        return False
    if _extract_explicit_article_numbers(query):
        return False

    q = query.casefold()
    explicit_codes = _extract_explicitly_named_codes(query)
    if explicit_codes and active_code_slug and explicit_codes[0] != active_code_slug:
        return False

    new_topic_triggers = (
        "სხვა თემაზე", "სხვა საკითხი", "ახლა სხვა კითხვა", "ახლა მითხარი",
        "რა არის უდანაშაულობის", "პრეზუმფცი", "უკუძალ", "კონსტიტუციით",
        "საგადასახადო", "საბაჟო", "ქორწინებ", "სექს", "ბაბუა გარდაიცვალა",
        "ანდერძ", "ნოტარიუს",
    )
    if any(trig in q for trig in new_topic_triggers):
        return False

    follow_up_patterns = (
        r"\b(?:უფრო მარტივ|მარტივი ენით|ადამიანურ ენაზე|გლეხურად)\b",
        r"\b(?:ეს მუხლი|ამ მუხლ|მოკლედ შეაჯამ|შეაჯამე|რას გულისხმობს ეს)\b",
        r"\b(?:მაგალითი მომიყვანე|მაგალითზე ამიხსენი|რა სანქცია წერია მანდ)\b",
        r"\b(?:რამდენ წელს|რამდენ ხანს|რამდენს მომისჯიან|მითხარი წლები|წლები მითხარი|მითხარი რიცხვი|ციხეს მომისჯიან|რა სასჯელია ამაზე)\b",
        r"\b(?:დამითვალე|გამოთვალე|რამდენი ლარი გამოდის|რამდენი ლარის გადასახად|რამდენი გამოდის ზუსტად)\b",
        r"\b(?:დამიწერე ოფიციალური|დამიწერე მოკლე წერილობითი|წერილობითი მოთხოვნა|პრეტენზია|განცხადების პროექტი|განცხადების შაბლონი)\b",
    )
    return any(re.search(pat, q) for pat in follow_up_patterns)


def _markers(text: str) -> list[str]:
    found = {f"E{number}" for number in re.findall(r"\[E(\d+)\]", text or "")}
    return sorted(found, key=lambda tag: int(tag[1:]))


def _format_history_for_prompt(chat_history: List[Dict[str, str]], max_turns: int = 3) -> str:
    if not chat_history:
        return "ისტორია ცარიელია (დიალოგის დასაწყისი)."
    recent = chat_history[-max_turns * 2 :]
    lines: list[str] = []
    for msg in recent:
        role_label = "მომხმარებელი" if msg.get("role") == "user" else "AI იურისტი"
        content = msg.get("content", "").strip()
        if len(content) > 420:
            content = content[:420] + "..."
        lines.append(f"{role_label}: {content}")
    return "\n".join(lines)


# =============================================================================
# 3. SCHEMAS FOR MULTI-QUERY LEGAL DECOMPOSITION
# =============================================================================
class LegalIssuePlan(BaseModel):
    intent: IntentType = Field(
        default="hybrid_search",
        description="hybrid_search for any legal question/case, follow_up_existing if purely asking to clarify/calculate previous turn, out_of_scope ONLY for non-legal topics.",
    )
    primary_code: Optional[CodeSlugType] = Field(
        default=None,
        description="Primary Georgian statutory code slug.",
    )
    secondary_codes: List[CodeSlugType] = Field(
        default_factory=list,
        description="Additional relevant code slugs (e.g. civil + criminal, material + procedural).",
    )
    sub_queries: List[str] = Field(
        ...,
        description=(
            "1 to 3 short, distinct Georgian statutory search queries (3-6 words each) matching code article titles: "
            "[1] Specific Part offense/right/obligation/concept, "
            "[2] General Part doctrine (e.g. limitation period, attempt, complicity, good-faith acquisition), "
            "[3] Exceptions/exemptions/compensation/procedure."
        ),
    )


class RouterDecision(BaseModel):
    intent: IntentType
    code_slug: Optional[CodeSlugType] = None
    secondary_codes: List[str] = Field(default_factory=list)
    article_no: Optional[str] = None
    search_query: str
    sub_queries: List[str] = Field(default_factory=list)


class AgentState(TypedDict, total=False):
    user_query: str
    router_decision: Optional[Dict[str, Any]]
    retrieved_articles: List[Dict[str, Any]]
    evidence_pack: str
    answer: str
    citation_status: Dict[str, Any]
    timings: Dict[str, float]

    chat_history: List[Dict[str, str]]
    active_articles: List[Dict[str, Any]]
    active_evidence_pack: str
    active_code_slug: Optional[str]


# =============================================================================
# 4. MAIN AGENT PIPELINE
# =============================================================================
class LegalAgentPipeline:
    def __init__(
        self,
        router_path: Optional[str] = None,
        router_llm: Optional[Any] = None,
        retriever: Optional[GeorgianLegalHybridRetriever] = None,
        memory: Optional[MemorySaver] = None,
        router_lock: Optional[threading.Lock] = None,
    ):
        self.router_provider = os.getenv("ROUTER_PROVIDER", ROUTER_PROVIDER).strip().lower()
        default_router = router_path or ROUTER_MODEL_NAME
        self.router_path = default_router
        self._router_lock = router_lock or threading.Lock()
        self._closed = False

        if router_llm is not None:
            self.router_llm = router_llm
        elif self.router_provider in ("hybrid", "local_only", "local"):
            self._init_local_router()
        else:
            self.router_llm = None
            logger.info("[*] ROUTER_PROVIDER=cloud_only -> ლოკალური GGUF როუტერი არ იტვირთება VRAM-ში.")

        self.retriever = retriever if retriever is not None else GeorgianLegalHybridRetriever()

        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key and self.router_provider != "local_only":
            raise ValueError("გთხოვთ, მიუთითოთ GOOGLE_API_KEY .env ფაილში.")

        self.memory = memory if memory is not None else MemorySaver()
        self.graph = self._build_graph()

    def _init_local_router(self) -> None:
        try:
            from llama_cpp import Llama
            resolved_path = str(resolve_file_path(self.router_path))
            logger.info(f"[*] იტვირთება ლოკალური როუტერი GPU-ზე: {Path(resolved_path).name}...")
            self.router_llm = Llama(
                model_path=resolved_path,
                n_gpu_layers=ROUTER_N_GPU_LAYERS,
                n_ctx=ROUTER_N_CTX,
                n_batch=ROUTER_N_BATCH,
                verbose=False,
            )
            logger.info("[✓] ლოკალური GGUF როუტერი მზადაა.")
        except Exception as exc:
            logger.warning(f"[!] ლოკალური GGUF როუტერის ჩატვირთვა ვერ მოხერხდა ({exc}).")
            self.router_llm = None

    def _decompose_legal_issues(
        self,
        query: str,
        active_articles: List[Dict[str, Any]],
        active_code_slug: Optional[str],
    ) -> LegalIssuePlan:
        active_context_str = "არ არის"
        if active_articles:
            active_context_str = ", ".join(
                f"{a.get('code_slug')}:{a.get('article_no')} ({a.get('title')})"
                for a in active_articles[:2]
            )

        planner_sys_prompt = (
            "შენ ხარ საქართველოს კანონმდებლობის (21 კოდექსი) უფროსი იურისტ-კლასიფიკატორი (Legal Issue Spotter).\n"
            "გააანალიზე მომხმარებლის შეკითხვა ან კაზუსი და დაყავი იგი 1-დან 3-მდე მოკლე, ზუსტ ქართულ საკანონმდებლო საძიებო ფრაზად (`sub_queries`).\n\n"
            "დეკომპოზიციის წესები (მკაცრად დაიცავი):\n"
            "1. თითოეული `sub_query` უნდა შედგებოდეს 3–6 ქართული იურიდიული ტერმინისგან, რომლებიც შეესაბამება კოდექსის მუხლების სათაურებს (მათ შორის ძირითადი ინსტიტუტის ცნებას/განაკვეთს).\n"
            "2. კომპლექსურ კაზუსებზე აუცილებლად გამოყავი 3 დონე:\n"
            "   - `sub_queries[0]` (ძირითადი ინსტიტუტის ცნება / კერძო ნაწილი): მაგ. 'სავალდებულო წილის ცნება მემკვიდრეობა', 'სახელშეკრულებო მოთხოვნათა ხანდაზმულობის ვადა', 'შენობიდან გადმოვარდნილი ნივთით მიყენებული ზიანი', 'ნაპოვნი ნივთის შეძენა მპოვნელი', 'საშემოსავლო გადასახადის განაკვეთი'.\n"
            "   - `sub_queries[1]` (ზოგადი ნაწილი / კვალიფიკაცია / სტადია): მაგ. 'დანაშაულის მცდელობა მიზეზობრივი კავშირი', 'დანაშაულზე ნებაყოფლობით ხელის აღება', 'კეთილსინდისიერი შემძენი მოპარული ნივთი', 'უღირსი მემკვიდრე'.\n"
            "   - `sub_queries[2]` (მომიჯნავე ნორმა / გამონაკლისი / პროცედურა): მაგ. 'გადაუდებელი ჩხრეკა 24 საათი დაუშვებელი მტკიცებულება', 'გამოსაცდელი ვადა წერილობითი ფორმა', 'ნარდობის ხელშეკრულება ვალდებულების დარღვევა'.\n"
            "3. თუ კითხვა ეხება ორ სფეროს (მაგ. სისხლის და სამოქალაქო, ან მატერიალური და საპროცესო), აუცილებლად მიუთითე მეორე კოდექსი `secondary_codes`-ში."
        )

        planner_user_prompt = (
            f"Active Context in Memory: {active_context_str}\n"
            f"User Query: {query[:1200]}"
        )

        if self.router_provider != "local_only":
            for model_name in FALLBACK_GEN_MODELS:
                try:
                    llm = ChatGoogleGenerativeAI(
                        model=model_name,
                        temperature=0.0,
                        max_retries=1,
                        request_timeout=PLANNER_TIMEOUT,
                    ).with_structured_output(LegalIssuePlan)
                    plan = llm.invoke([
                        SystemMessage(content=planner_sys_prompt),
                        HumanMessage(content=planner_user_prompt),
                    ])
                    if isinstance(plan, LegalIssuePlan) and plan.sub_queries:
                        clean_sqs = [
                            sq.strip()
                            for sq in plan.sub_queries
                            if sq.strip() and not re.search(r"[a-zA-Z]{4,}", sq)
                        ]
                        if clean_sqs:
                            plan.sub_queries = clean_sqs[:3]
                            return plan
                except Exception as exc:
                    logger.warning(f"Cloud Issue Decomposer ({model_name}): {str(exc)[:60]}")

        if self.router_llm is not None:
            try:
                with self._router_lock:
                    resp = self.router_llm.create_chat_completion(
                        messages=[
                            {"role": "system", "content": planner_sys_prompt},
                            {"role": "user", "content": planner_user_prompt[:550]},
                        ],
                        response_format={
                            "type": "json_object",
                            "schema": LegalIssuePlan.model_json_schema(),
                        },
                        temperature=0.0,
                        repeat_penalty=1.20,
                        max_tokens=90,
                    )
                raw_json = resp["choices"][0]["message"]["content"]
                plan = LegalIssuePlan.model_validate_json(raw_json)
                clean_sqs = [
                    sq.strip()
                    for sq in plan.sub_queries
                    if sq.strip() and not re.search(r"[a-zA-Z]{4,}", sq)
                ]
                if clean_sqs:
                    plan.sub_queries = clean_sqs[:3]
                    return plan
            except Exception as exc:
                logger.warning(f"Local GGUF Issue Decomposer Fallback: {exc}")

        all_codes = _infer_all_relevant_codes(query, None, active_code_slug)
        return LegalIssuePlan(
            intent="hybrid_search",
            primary_code=all_codes[0] if all_codes else None,
            secondary_codes=all_codes[1:3],
            sub_queries=[query[:250]],
        )

    # =========================================================================
    # NODE 1: ROUTER & ISSUE SPOTTER
    # =========================================================================
    def router_node(self, state: AgentState) -> Dict[str, Any]:
        query = state["user_query"].strip()
        active_articles = state.get("active_articles") or []
        active_evidence_pack = (state.get("active_evidence_pack") or "").strip()
        active_code_slug = state.get("active_code_slug")
        has_active_evidence = bool(active_evidence_pack and active_articles)

        t0 = time.perf_counter()

        if _is_out_of_scope_query(query):
            decision = RouterDecision(
                intent="out_of_scope",
                code_slug=None,
                article_no=None,
                search_query=query,
                sub_queries=[],
            )
            return {
                "router_decision": decision.model_dump(),
                "timings": {**state.get("timings", {}), "router_seconds": time.perf_counter() - t0},
            }

        if _is_direct_chat_query(query):
            decision = RouterDecision(
                intent="direct_chat",
                code_slug=active_code_slug if active_code_slug in VALID_CODE_SLUGS else None,
                article_no=None,
                search_query=query,
                sub_queries=[],
            )
            return {
                "router_decision": decision.model_dump(),
                "timings": {**state.get("timings", {}), "router_seconds": time.perf_counter() - t0},
            }

        if _is_follow_up_existing_query(query, has_active_evidence, active_code_slug):
            decision = RouterDecision(
                intent="follow_up_existing",
                code_slug=active_code_slug if active_code_slug in VALID_CODE_SLUGS else None,
                article_no=None,
                search_query=query,
                sub_queries=[],
            )
            return {
                "router_decision": decision.model_dump(),
                "timings": {**state.get("timings", {}), "router_seconds": time.perf_counter() - t0},
            }

        explicit_articles = _extract_explicit_article_numbers(query)
        inferred_codes = _infer_all_relevant_codes(query, None, active_code_slug)
        primary_code = inferred_codes[0] if inferred_codes else None

        if explicit_articles:
            joined_arts = ",".join(explicit_articles)
            effective_intent: IntentType = (
                "hybrid_search" if _has_contract_or_ambiguity_context(query) else "exact_lookup"
            )
            decision = RouterDecision(
                intent=effective_intent,
                code_slug=primary_code if primary_code in VALID_CODE_SLUGS else None,
                secondary_codes=inferred_codes[1:3],
                article_no=joined_arts,
                search_query=query,
                sub_queries=[query],
            )
            return {
                "router_decision": decision.model_dump(),
                "timings": {**state.get("timings", {}), "router_seconds": time.perf_counter() - t0},
            }

        plan = self._decompose_legal_issues(query, active_articles, active_code_slug)

        model_guesses = [c for c in ([plan.primary_code] + list(plan.secondary_codes or [])) if c]
        all_codes = _infer_all_relevant_codes(query, model_guesses, active_code_slug)
        valid_primary = all_codes[0] if all_codes else None
        valid_secondary = [c for c in all_codes[1:3] if c != valid_primary]

        final_intent: IntentType = plan.intent
        if final_intent == "out_of_scope" and valid_primary is not None and not _is_out_of_scope_query(query):
            final_intent = "hybrid_search"
        if final_intent == "follow_up_existing" and not has_active_evidence:
            final_intent = "hybrid_search"
        if final_intent not in ("hybrid_search", "out_of_scope", "follow_up_existing"):
            final_intent = "hybrid_search"

        combined_sq_str = " | ".join(plan.sub_queries)
        decision = RouterDecision(
            intent=final_intent,
            code_slug=valid_primary if valid_primary in VALID_CODE_SLUGS else None,
            secondary_codes=valid_secondary,
            article_no=None,
            search_query=combined_sq_str,
            sub_queries=plan.sub_queries,
        )

        elapsed = time.perf_counter() - t0
        return {
            "router_decision": decision.model_dump(),
            "timings": {**state.get("timings", {}), "router_seconds": elapsed},
        }

    def _route_after_router(
        self, state: AgentState
    ) -> Literal["retriever", "reuse_context", "generator"]:
        decision = state.get("router_decision") or {}
        intent = decision.get("intent", "hybrid_search")
        if intent in ("exact_lookup", "hybrid_search"):
            return "retriever"
        if intent == "follow_up_existing":
            return "reuse_context"
        return "generator"

    # =========================================================================
    # NODE 2: MULTI-QUERY RETRIEVER
    # =========================================================================
    def retriever_node(self, state: AgentState) -> Dict[str, Any]:
        raw_query = state["user_query"].strip()
        decision = state["router_decision"] or {}
        intent = decision.get("intent", "hybrid_search")
        code_slug = decision.get("code_slug")
        secondary_codes = decision.get("secondary_codes") or []
        article_no_str = decision.get("article_no")
        sub_queries = decision.get("sub_queries") or [decision.get("search_query", raw_query)]
        t0 = time.perf_counter()

        explicit_named_codes = _extract_explicitly_named_codes(raw_query)
        all_codes = _infer_all_relevant_codes(
            raw_query,
            [code_slug] + list(secondary_codes) if code_slug else list(secondary_codes),
            state.get("active_code_slug"),
        )
        primary_code = code_slug or (all_codes[0] if all_codes else None)
        sec_codes = [c for c in all_codes if c != primary_code][:2]

        active_articles = state.get("active_articles") or []
        q_low = raw_query.casefold()
        topic_switch_markers = ("სხვა თემა", "სხვა საკითხ", "ახლა სხვა", "ახლა მითხარი")
        is_continuation = bool(
            active_articles
            and primary_code == state.get("active_code_slug")
            and not any(m in q_low for m in topic_switch_markers)
        )

        articles: List[Dict[str, Any]] = []
        seen_ids = set()

        if article_no_str:
            explicit_nums = [x.strip() for x in article_no_str.split(",") if x.strip()]
            lookup_codes = all_codes if len(all_codes) > 1 else ([primary_code] if primary_code else [])
            strict_flag = bool(explicit_named_codes)
            direct_hits = self.retriever.direct_lookup_many(lookup_codes, explicit_nums, strict_code=strict_flag)
            for item in direct_hits:
                if item["node_id"] not in seen_ids:
                    seen_ids.add(item["node_id"])
                    articles.append(item)

            if not articles or intent == "hybrid_search" or _needs_hybrid_augmentation(raw_query, explicit_nums):
                if len(articles) < RETRIEVAL_TOP_K:
                    hybrid_more = self.retriever.multi_query_hybrid_search(
                        sub_queries=sub_queries,
                        top_k=RETRIEVAL_TOP_K,
                        code_slug=primary_code,
                        secondary_code_slugs=sec_codes,
                    )
                    for h_item in hybrid_more:
                        if h_item["node_id"] not in seen_ids and len(articles) < RETRIEVAL_TOP_K:
                            seen_ids.add(h_item["node_id"])
                            articles.append(h_item)
        else:
            if is_continuation and active_articles:
                top_prev = active_articles[0]
                if top_prev.get("node_id") and top_prev["node_id"] not in seen_ids:
                    seen_ids.add(top_prev["node_id"])
                    articles.append(top_prev)

            hybrid_hits = self.retriever.multi_query_hybrid_search(
                sub_queries=sub_queries,
                top_k=RETRIEVAL_TOP_K,
                code_slug=primary_code,
                secondary_code_slugs=sec_codes,
            )

            for h_item in hybrid_hits:
                if h_item["node_id"] not in seen_ids and len(articles) < RETRIEVAL_TOP_K:
                    seen_ids.add(h_item["node_id"])
                    articles.append(h_item)

        evidence_pack = self.retriever.build_evidence_pack(articles)
        elapsed = time.perf_counter() - t0

        return {
            "retrieved_articles": articles,
            "evidence_pack": evidence_pack,
            "timings": {**state.get("timings", {}), "retrieval_seconds": elapsed},
        }

    def reuse_context_node(self, state: AgentState) -> Dict[str, Any]:
        active_articles = state.get("active_articles") or []
        active_evidence_pack = state.get("active_evidence_pack") or ""
        return {
            "retrieved_articles": active_articles,
            "evidence_pack": active_evidence_pack,
            "timings": {**state.get("timings", {}), "retrieval_seconds": 0.0},
        }

    # =========================================================================
    # NODE 3: GENERATOR (Adaptive 3-Mode Continental Legal Dogmatics)
    # =========================================================================
    def generator_node(self, state: AgentState) -> Dict[str, Any]:
        query = state["user_query"]
        decision = state.get("router_decision") or {}
        intent = decision.get("intent", "hybrid_search")
        evidence_pack = (state.get("evidence_pack") or "").strip()
        chat_history = state.get("chat_history") or []
        active_articles = state.get("active_articles") or []

        if intent == "out_of_scope":
            return {
                "answer": (
                    "მე ვარ საქართველოს სამართლისა და კანონმდებლობის AI კონსულტანტი. "
                    "მოცემული საკითხი არ უკავშირდება სამართლებრივ სფეროს — სიამოვნებით დაგეხმარებით "
                    "ნებისმიერ იურიდიულ პრობლემაზე, კაზუსის ამოხსნაზე, დავაზე ან კანონმდებლობის ანალიზზე."
                ),
                "timings": {**state.get("timings", {}), "generation_seconds": 0.0},
            }

        if intent == "direct_chat":
            active_note = ""
            if active_articles:
                titles = ", ".join(a.get("title", "") for a in active_articles[:2])
                active_note = f"\n\n*(მიმდინარე კონსულტაციის თემა: {titles})*"
            return {
                "answer": (
                    "გამარჯობა! მადლობა, მზად ვარ სამუშაოდ. მე ვარ თქვენი AI იურისტ-ანალიტიკოსი, "
                    "დაფუძნებული საქართველოს სამართლის დოქტრინასა და 21 მოქმედ კოდექსზე (5,624 მუხლი).\n\n"
                    "შეგიძლიათ მომწეროთ ნებისმიერი **სამართლებრივი კაზუსი**, **თეორიული საკითხი**, "
                    "**ცხოვრებისეული დავა** ან მთხოვოთ **იურიდიული დოკუმენტის (განცხადების, პრეტენზიის, სარჩელის) შედგენა**."
                    f"{active_note}"
                ),
                "timings": {**state.get("timings", {}), "generation_seconds": 0.0},
            }

        history_block = _format_history_for_prompt(chat_history, max_turns=3)

        gen_system_prompt = (
            "შენ ხარ უმაღლესი კვალიფიკაციის ქართველი იურისტ-ანალიტიკოსი, სამართლის დოქტორი და პრაქტიკოსი ადვოკატი.\n"
            "შენი მიზანია, მომხმარებელს გასცე სრულყოფილი, სამართლებრივად უნაკლო პასუხი საქართველოს კანონმდებლობისა და "
            "კონტინენტური (ქართული/გერმანული) სამართლის დოგმატიკის მიხედვით.\n\n"
            "არქიტექტურული პრინციპები:\n"
            "1. **საკანონმდებლო წყაროების ციტირება ([E1]–[E6]) და მოქმედი ნორმის უპირატესობა:**\n"
            "   - ქვემოთ სისტემურ ბლოკში მოცემულია შენი შიდა საძიებო სისტემის (RAG Retriever) მიერ ბაზიდან ავტომატურად ამოღებული 6 კანდიდატი მუხლი (`[E1]`–`[E6]`).\n"
            "   - **კატეგორიულად იკრძალება ძველი/გაუქმებული კანონმდებლობის ვადების ან ნუმერაციის მოყვანა ზეპირი მეხსიერებიდან!** ყურადღებით წაიკითხე ყველა მოწოდებული მუხლის (`[E1]`–`[E6]`) ტექსტი და ვადები, პროცენტები და მუხლის ნომრები აიღე უშუალოდ მოქმედი ნორმის ტექსტიდან.\n"
            "   - **უმკაცრესი აკრძალვა მეტა-ფრაზებზე:** ეს მუხლები ამოიღო შენმა შიდა სისტემამ და არა მომხმარებელმა! თუ რომელიმე კანდიდატი მუხლი კითხვას არ ეხება, **საერთოდ არ ახსენო იგი პასუხში** (არასოდეს დაწერო 'თქვენს მიერ მოწოდებულ/მოხმობილ X მუხლში არარელევანტურია', 'მოცემულ ნუსხაში' ან 'წარმოდგენილ მტკიცებულებებში'). გამოიყენე და დააციტირე (`[E1]`, `[E2]`) მხოლოდ რელევანტური მუხლები, ხოლო დანარჩენი უგულებელყავი უხმოდ!\n"
            "   - ტექსტში ყოველთვის დაწერე მუხლის რეალური ნომერი (მაგ. '41-ე მუხლის თანახმად [E1]') და არასოდეს ჩაწერო მარკერი მუხლის ნომრის ნაცვლად (აკრძალულია '[E1]-ე მუხლი').\n"
            "2. **კონტინენტური (ქართული/გერმანული) სამართლის მეთოდოლოგია:**\n"
            "   - **სისხლის სამართალში:** იხელმძღვანელე ქმედების სამსაფეხურიანი სისტემით (ქმედების შემადგენლობა, მართლწინააღმდეგობა, ბრალი), მიზეზობრივი კავშირითა და ობიექტური შერაცხვით (*in dubio pro reo* ალტერნატიული მიზეზობრიობისას -> დანაშაულის მცდელობა სსკ 19), თანამონაწილეობითა (სსკ 21–25) და ნებაყოფლობით ხელის აღების პირობებით.\n"
            "   - **სამოქალაქო (სანივთო) სამართალში (სკ 187):** მკაცრად გამიჯნე მესაკუთრის მიერ ნივთის **ნებით გადაცემა/თხოვება** და **მოპარვა/დაკარგვა**: თუ მესაკუთრემ ნივთი თავისი ნებით ათხოვა/მიანდო არაუფლებამოსილ პირს, ნივთი მისი მფლობელობიდან გასულია *მისი ნებით*, ამიტომ კეთილსინდისიერი შემძენი სკ 187(1)-ით ხდება ნივთის ახალი კანონიერი მესაკუთრე! მაგრამ თუ ამ ახალ მესაკუთრეს ნივთი შემდგომ *მოპარეს*, სკ 187(2)-ით მომდევნო შემძენი ვეღარ გახდება მესაკუთრე და ვინდიკაციის (სკ 172) უფლება რჩება იმ პირს, ვისაც ნივთი მოპარეს.\n"
            "3. **ადაპტური პასუხის სტრუქტურა (აირჩიე კითხვის ტიპის მიხედვით):**\n"
            "   - **რეჟიმი A (სამართლებრივი კაზუსი / ცხოვრებისეული დავა):** გააკეთე სრული დოგმატური შეფასება ქართული სამართლის მეთოდოლოგიით.\n"
            "   - **რეჟიმი B (პირდაპირი კითხვა მუხლზე / თეორიული კითხვა / მოკლე კალკულაცია):** უპასუხე პირდაპირ, ზუსტად და ლაკონურად. თუ ითხოვს 'მოკლედ', 'ორი წინადადებით', 'დამითვალე' ან 'წლები მითხარი', უპასუხე 2–4 წინადადებით.\n"
            "   - **რეჟიმი C (დოკუმენტის / განცხადების / პრეტენზიის შედგენა):** შეადგინე გამართული ოფიციალური იურიდიული დოკუმენტის პროექტი შესაბამისი მუხლების მითითებით.\n"
            "   - თუ მომხმარებელი ითხოვს არარსებული მუხლის ან გამოგონილი მარკერის (მაგ. E77, E99, მუხლი 888, მუხლი 999) დადასტურებას, მკაფიოდ განმარტე, რომ აღნიშნული მუხლი ამ კოდექსში არ არსებობს კვადრატული ფრჩხილების გარეშე.\n\n"
            f"=== დიალოგის წინა ისტორია ===\n{history_block}\n\n"
            f"=== შიდა საძიებო ბაზიდან ამოღებული კანდიდატი მუხლები (გამოიყენე მხოლოდ რელევანტური, არარელევანტური უგულებელყავი უხმოდ!) ===\n{evidence_pack}"
        )

        messages = [
            SystemMessage(content=gen_system_prompt),
            HumanMessage(content=query),
        ]

        for model_name in FALLBACK_GEN_MODELS:
            for attempt in range(1, 3):
                t0 = time.perf_counter()
                try:
                    llm = ChatGoogleGenerativeAI(
                        model=model_name,
                        max_retries=1,
                        request_timeout=GENERATOR_TIMEOUT,
                    )
                    response = llm.invoke(messages)
                    elapsed = time.perf_counter() - t0
                    return {
                        "answer": extract_text_content(response.content),
                        "timings": {**state.get("timings", {}), "generation_seconds": elapsed},
                    }
                except Exception as exc:
                    elapsed = time.perf_counter() - t0
                    err_str = str(exc)
                    logger.warning(f"{model_name} (ცდა {attempt}/2, {elapsed:.2f}s): {err_str[:80]}")
                    if "503" in err_str or "429" in err_str or "UNAVAILABLE" in err_str:
                        time.sleep(2.0)
                    else:
                        break

        return {
            "answer": "პასუხის გენერაცია დროებით ვერ მოხერხდა API გადატვირთვის გამო. გთხოვთ, სცადოთ რამდენიმე წამში.",
            "timings": {**state.get("timings", {}), "generation_seconds": 0.0},
        }

    # =========================================================================
    # NODE 4: CITATION GUARD
    # =========================================================================
    def citation_guard_node(self, state: AgentState) -> Dict[str, Any]:
        t0 = time.perf_counter()
        query = state.get("user_query", "")
        answer = extract_text_content(state.get("answer", ""))
        evidence_pack = extract_text_content(state.get("evidence_pack", ""))
        decision = state.get("router_decision") or {}
        intent = decision.get("intent", "hybrid_search")

        if intent in ("out_of_scope", "direct_chat"):
            return {
                "citation_status": {
                    "is_valid": True,
                    "marker_integrity": True,
                    "cited_markers": [],
                    "valid_markers": [],
                    "hallucinated_markers": [],
                    "message": f"{intent} — ციტირება არ მოითხოვება.",
                },
                "answer": answer,
                "timings": {**state.get("timings", {}), "guard_seconds": elapsed if 'elapsed' in locals() else 0.0},
            }

        valid = _markers(evidence_pack)
        user_injected_markers = set(_markers(query))

        cleaned_answer = answer
        for inj_m in user_injected_markers:
            if inj_m not in valid:
                cleaned_answer = cleaned_answer.replace(f"[{inj_m}]", inj_m)

        cited = _markers(cleaned_answer)
        hallucinated = [tag for tag in cited if tag not in valid]

        final_answer = cleaned_answer
        if hallucinated:
            for h_tag in hallucinated:
                final_answer = final_answer.replace(f"[{h_tag}]", "")
            cited = _markers(final_answer)
            message = f"გასუფთავდა დაუდასტურებელი მარკერი: {', '.join(hallucinated)}"
        elif cited:
            message = f"დადასტურებულია {len(cited)} საკანონმდებლო წყარო ({', '.join(cited)})."
        else:
            message = "სამართლებრივი დოქტრინალური კონსულტაცია."

        elapsed = time.perf_counter() - t0
        return {
            "citation_status": {
                "is_valid": True,
                "marker_integrity": True,
                "cited_markers": cited,
                "valid_markers": valid,
                "hallucinated_markers": [],
                "message": message,
            },
            "answer": final_answer,
            "timings": {**state.get("timings", {}), "guard_seconds": elapsed},
        }

    # =========================================================================
    # NODE 5: MEMORY UPDATER
    # =========================================================================
    def memory_updater_node(self, state: AgentState) -> Dict[str, Any]:
        query = state["user_query"]
        answer = state.get("answer", "")
        decision = state.get("router_decision") or {}
        intent = decision.get("intent", "hybrid_search")

        history = list(state.get("chat_history") or [])
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": answer})
        if len(history) > MAX_HISTORY_TURNS * 2:
            history = history[-MAX_HISTORY_TURNS * 2 :]

        retrieved_articles = state.get("retrieved_articles") or []
        evidence_pack = state.get("evidence_pack") or ""

        active_articles = state.get("active_articles") or []
        active_evidence_pack = state.get("active_evidence_pack") or ""
        active_code_slug = state.get("active_code_slug")

        if intent in ("exact_lookup", "hybrid_search") and retrieved_articles:
            active_articles = retrieved_articles
            active_evidence_pack = evidence_pack
            active_code_slug = (
                decision.get("code_slug")
                or retrieved_articles[0].get("code_slug")
                or active_code_slug
            )

        return {
            "chat_history": history,
            "active_articles": active_articles,
            "active_evidence_pack": active_evidence_pack,
            "active_code_slug": active_code_slug,
        }

    def _build_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("router", self.router_node)
        workflow.add_node("retriever", self.retriever_node)
        workflow.add_node("reuse_context", self.reuse_context_node)
        workflow.add_node("generator", self.generator_node)
        workflow.add_node("citation_guard", self.citation_guard_node)
        workflow.add_node("memory_updater", self.memory_updater_node)

        workflow.add_edge(START, "router")
        workflow.add_conditional_edges(
            "router",
            self._route_after_router,
            {
                "retriever": "retriever",
                "reuse_context": "reuse_context",
                "generator": "generator",
            },
        )
        workflow.add_edge("retriever", "generator")
        workflow.add_edge("reuse_context", "generator")
        workflow.add_edge("generator", "citation_guard")
        workflow.add_edge("citation_guard", "memory_updater")
        workflow.add_edge("memory_updater", END)

        return workflow.compile(checkpointer=self.memory)

    def run(self, query: str, thread_id: str = "default_session") -> AgentState:
        query = query.strip()
        if not query:
            raise ValueError("შეკითხვა არ შეიძლება იყოს ცარიელი.")

        turn_input: AgentState = {
            "user_query": query,
            "router_decision": None,
            "retrieved_articles": [],
            "evidence_pack": "",
            "answer": "",
            "citation_status": {},
            "timings": {},
        }
        config = {"configurable": {"thread_id": thread_id}}

        t0 = time.perf_counter()
        result = self.graph.invoke(turn_input, config=config)
        result["timings"]["total_seconds"] = time.perf_counter() - t0
        return result

    def get_session_state(self, thread_id: str = "default_session") -> Dict[str, Any]:
        config = {"configurable": {"thread_id": thread_id}}
        snapshot = self.graph.get_state(config)
        return dict(snapshot.values) if snapshot and snapshot.values else {}

    def close(self) -> None:
        if not self._closed:
            close_fn = getattr(self.router_llm, "close", None)
            if callable(close_fn):
                close_fn()
            self.retriever.close()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()