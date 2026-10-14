"""
legal_agent.py
=============================================================================
Georgian Legal AI (v4 Production Pipeline) — ნაბიჯი 5
Stateful Conversational LangGraph Agent across all 21 Georgian Codes.
=============================================================================
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, TypedDict

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

if os.name == "nt":
    try:
        import torch
        torch_lib = Path(torch.__file__).resolve().parent / "lib"
        if torch_lib.is_dir() and hasattr(os, "add_dll_directory"):
            os.add_dll_directory(str(torch_lib))
    except Exception:
        pass

from llama_cpp import Llama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from hybrid_retriever import GeorgianLegalHybridRetriever, resolve_file_path

ROUTER_MODEL_NAME = os.getenv("ROUTER_MODEL_PATH", "Qwen_Qwen3.5-4B-Q4_K_M.gguf")
PRIMARY_GEN_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
FALLBACK_GEN_MODELS = [
    PRIMARY_GEN_MODEL,
    "gemini-3.5-flash-lite" if PRIMARY_GEN_MODEL != "gemini-3.5-flash-lite" else "gemini-3.1-flash-lite",
]

ROUTER_N_CTX = int(os.getenv("ROUTER_N_CTX", "2048"))
ROUTER_N_BATCH = int(os.getenv("ROUTER_N_BATCH", "512"))
ROUTER_N_GPU_LAYERS = int(os.getenv("ROUTER_N_GPU_LAYERS", "-1"))
RETRIEVAL_TOP_K = max(1, int(os.getenv("RETRIEVAL_TOP_K", "4")))
GENERATOR_TIMEOUT = float(os.getenv("GENERATOR_TIMEOUT_SECONDS", "35.0"))
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


def _extract_explicit_article_number(query: str) -> Optional[str]:
    """დეტერმინისტულად ამოიცნობს მუხლის ნომერს და ფილტრავს ომონიმებს ('მუხლი 39-ჯერ')."""
    q = query.casefold().replace("−", "-")
    if re.search(r"მუხლ(?:ი|ის|ში|ით|ზე|ს|ად)?\s+\d+\s*[- ]?ჯერ", q):
        return None
    if re.search(r"მუხლებზე\s*(?:დამაჩოქ|დამსვ|მაიძულ|იყავ)", q):
        return None
    if re.search(r"მუხლ(?:ი|ზე|ში)?[^.!?]{0,45}(?:მიმერტყ|მიმარტყ|დამარტყ|მირტყ|მტკივ|ტკივილ)", q):
        return None

    article_num = r"(\d{1,4}(?:\s*(?:\^\s*\d{1,3}|[⁰¹²³⁴⁵⁶⁷⁸⁹]+))?)"
    article_word = r"მუხლ(?:ი|ის|ში|ით|ზე|ს|ად)?"

    patterns = (
        rf"{article_num}\s*(?:-?ე\s*)?{article_word}",
        rf"{article_word}\s*(?:№\s*)?{article_num}(?!\s*[- ]?ჯერ)",
    )
    for pattern in patterns:
        match = re.search(pattern, q)
        if match:
            number = re.sub(r"\s+", "", match.group(1))
            superscripts = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
            if "^" not in number:
                match_suffix = re.match(r"^(\d+)([⁰¹²³⁴⁵⁶⁷⁸⁹]+)$", number)
                if match_suffix:
                    number = f"{match_suffix.group(1)}^{match_suffix.group(2).translate(superscripts)}"
            return number
    return None


def _expand_colloquial_legal_query(raw_query: str) -> tuple[Optional[str], Optional[str]]:
    """
    სალაპარაკო (ლაიკური) ენით დასმულ იურიდიულ კითხვებს უმატებს კოდექსის ზუსტ ტერმინოლოგიას.
    """
    q = raw_query.casefold()

    if any(w in q for w in ("სექს", "სქესობრივ", "გაუპატიურებ")):
        if any(w in q for w in ("წლიდან", "ასაკ", "დაშვებულ", "არასრულწლოვ", "16", "თექვსმეტ")):
            return (
                "criminal",
                "სექსუალური ხასიათის შეღწევა თექვსმეტი წლის ასაკს მიუღწევლის სხეულში მუხლი 140 მუხლი 141",
            )
        return ("criminal", None)

    if any(w in q for w in ("დამკითხ", "აღიარებ", "წამებ", "მცემდნენ", "მიმერტყ")):
        return (
            "criminal",
            "წამება დამამცირებელი ან არაადამიანური მოპყრობა ჩვენების მიცემის იძულება მუხლი 144^1 მუხლი 144^3 მუხლი 335",
        )

    if "აუცილებელ" in q and "მოგერიებ" in q:
        return ("criminal", "აუცილებელი მოგერიება მართლწინააღმდეგობის გამომრიცხველი გარემოება მუხლი 28")

    if "ქორწინებ" in q and any(w in q for w in ("ასაკ", "წლიდან", "დაშვებულ")):
        return ("civil", "საქორწინო ასაკი ქორწინება დაიშვება 18 წლის ასაკიდან მუხლი 1108")

    if any(w in q for w in ("სრულწლოვან", "ქმედუნარიან")):
        return ("civil", "სრულწლოვანება ქმედუნარიანობა 18 წლის ასაკიდან მუხლი 12")

    if "უძრავ" in q and any(w in q for w in ("ნასყიდობ", "შეძენ", "საკუთრებ", "ფორმ")):
        return ("civil", "უძრავი ნივთის შესაძენად გარიგების წერილობითი ფორმა საჯარო რეესტრში რეგისტრაცია მუხლი 183")

    if any(w in q for w in ("შეწყდეს შრომითი", "შრომითი ხელშეკრულების შეწყვეტ", "სამსახურიდან გათავისუფლებ")):
        return ("labor", "შრომითი ხელშეკრულების შეწყვეტის საფუძვლები მუხლი 47 მუხლი 48")

    if "წვრილმანი ხულიგნობ" in q or ("ლანძღვა" in q and "საჯარო" in q):
        return ("admin_offenses", "წვრილმანი ხულიგნობა საჯარო სივრცეში ლანძღვა-გინება მუხლი 166")

    if "საშემოსავლო გადასახად" in q and ("პროცენტ" in q or "განაკვეთ" in q):
        return ("tax", "ფიზიკური პირის დასაბეგრი შემოსავალი იბეგრება 20 პროცენტით საშემოსავლო გადასახადის განაკვეთი მუხლი 81")

    return (None, None)


def _infer_code_slug(
    query: str,
    model_guess: Optional[str] = None,
    active_code_slug: Optional[str] = None,
) -> Optional[str]:
    q = query.casefold()
    colloq_code, _ = _expand_colloquial_legal_query(query)
    if colloq_code:
        return colloq_code

    # 21-ვე კოდექსის დეტერმინისტული საკვანძო სიტყვები
    code_keywords = [
        ("constitution", ("კონსტიტუცი", "პრეზიდენტის იმპიჩმენტ", "სახელმწიფო სუვერენიტეტ")),
        ("criminal_proc", ("სისხლის სამართლის საპროცესო", "ბრალდებულის დაკავებ", "ნაფიც მსაჯულ", "აღკვეთის ღონისძიებ", "გირაო")),
        ("civil_proc", ("სამოქალაქო საპროცესო", "სარჩელის უზრუნველყოფ", "დაუსწრებელი გადაწყვეტილებ")),
        ("admin_proc", ("ადმინისტრაციული საპროცესო",)),
        ("admin_offenses", ("ადმინისტრაციულ სამართალდარღვევა", "წვრილმანი ხულიგნობ", "ადმინისტრაციული პატიმრობ", "ჯარიმა საგზაო")),
        ("gen_admin", ("ზოგადი ადმინისტრაციული", "საჯარო ინფორმაცი", "ადმინისტრაციული აქტ")),
        ("tax", ("საგადასახადო", "საშემოსავლო გადასახად", "დღგ", "მოგების გადასახად", "ქონების გადასახად")),
        ("customs", ("საბაჟო", "დეკლარაცი", "იმპორტის გადასახდელ", "საბაჟო კონტროლ")),
        ("election", ("საარჩევნო", "არჩევნებ", "ცესკო", "საარჩევნო უბან", "კანდიდატის რეგისტრაცი")),
        ("imprisonment", ("პატიმრობის კოდექს", "პენიტენციურ", "მსჯავრდებულის პაემან")),
        ("juvenile", ("არასრულწლოვანთა მართლმსაჯულებ", "განრიდება")),
        ("child_rights", ("ბავშვის უფლებათა კოდექს",)),
        ("spatial_building", ("სამშენებლო", "მშენებლობის ნებართვ", "სივრცითი დაგეგმარებ")),
        ("product_safety", ("პროდუქტის უსაფრთხოებ", "ტექნიკური რეგლამენტ")),
        ("forest", ("ტყის კოდექს", "ხე-ტყის დამზადებ")),
        ("env_assessment", ("გარემოსდაცვითი შეფასებ", "გზშ")),
        ("air", ("საჰაერო კოდექს", "საჰაერო ხომალდ", "ავიაცი")),
        ("maritime", ("საზღვაო კოდექს", "გემის კაპიტან", "ნავსადგურ")),
        ("labor", ("შრომის კოდექს", "შრომითი", "დასაქმებულ", "დამსაქმებელ", "შვებულებ", "ზეგანაკვეთურ", "დეკრეტ", "გამოსაცდელი ვად")),
        ("criminal", ("სისხლის სამართლის", "სისხლის კოდექს", "სსკ", "დანაშაულ", "სასჯელ", "მკვლელობ", "ქურდობ", "ძარცვ", "ყაჩაღობ", "თავისუფლების აღკვეთ")),
        ("civil", ("სამოქალაქო კოდექს", "სამოქალაქო სამართლის", "მემკვიდრ", "ქორწინებ", "სრულწლოვან", "ხანდაზმულობ", "უძრავი ქონებ", "ნასყიდობ", "სესხ", "იპოთეკ")),
    ]

    for slug, keywords in code_keywords:
        if any(k in q for k in keywords):
            return slug

    if model_guess in VALID_CODE_SLUGS:
        return model_guess

    if active_code_slug in VALID_CODE_SLUGS and any(
        w in q for w in ("ამ კოდექს", "მაგ კოდექს", "იგივე დანაშაულ", "ამ მუხლ", "ეს მუხლ")
    ):
        return active_code_slug

    return None


def _is_direct_chat_query(query: str) -> bool:
    q = re.sub(r"[^\w\sა-ჰ]", " ", query.casefold()).strip()
    words = q.split()
    if not words or len(words) > 12:
        return False

    legal_signals = (
        "მუხლ", "კოდექს", "კონსტიტუცი", "დანაშაულ", "სასჯელ", "შრომ", "ხელშეკრულებ",
        "სექს", "ქორწინებ", "სამოქალაქო", "სისხლის", "ანაზღაურებ", "გათავისუფლებ",
        "მკვლელობ", "ქურდობ", "ასაკ", "გადასახად", "საბაჟო", "ჯარიმ", "არჩევნებ",
    )
    if any(sig in q for sig in legal_signals):
        return False

    chat_patterns = (
        r"^(?:გამარჯობა|გაგიმარჯოს|სალამი|დილა მშვიდობისა|საღამო მშვიდობისა|ჰეი|hello|hi)\b",
        r"\b(?:ვინ ხარ|რა გქვია|რა შეგიძლია|რაში მეხმარები|რაში შეგიძლია დამეხმარო|როგორ მუშაობ)\b",
        r"^(?:დიდი მადლობა|მადლობა|გაიხარე|გასაგებია ყველაფერი|კარგად|ნახვამდის)\b",
    )
    return any(re.search(pat, q) for pat in chat_patterns)


def _is_follow_up_existing_query(query: str, has_active_evidence: bool) -> bool:
    if not has_active_evidence:
        return False
    if _extract_explicit_article_number(query):
        return False

    q = query.casefold()
    new_search_triggers = (
        "და თუ ", "მაგრამ თუ", "დამამძიმებელ", "ჯგუფურად", "ანგარებით",
        "სხვა მუხლ", "კიდევ რა დანაშაულ", "მცემდნენ", "დამკითხეს",
        "რომელი წლიდან", "რა ასაკიდან", "ქორწინებ", "სექს", "გადასახად",
    )
    if any(trig in q for trig in new_search_triggers):
        return False

    follow_up_patterns = (
        r"\b(?:უფრო მარტივ|მარტივად|მარტივი ენით|ადამიანურ ენაზე|გლეხურად)\b",
        r"\b(?:ეს მუხლი|ამ მუხლ|მოკლედ შეაჯამ|შეაჯამე|რას გულისხმობს ეს)\b",
        r"\b(?:მაგალითი მომიყვანე|მაგალითზე ამიხსენი|რა სანქცია წერია მანდ)\b",
        r"\b(?:მე-\d+ ნაწილ|პირველ ნაწილ|მეორე ნაწილ|მესამე ნაწილ)\b",
    )
    return any(re.search(pat, q) for pat in follow_up_patterns)


def _enrich_contextual_search_query(
    raw_query: str,
    router_search_query: str,
    active_articles: List[Dict[str, Any]],
    current_code_slug: Optional[str],
    active_code_slug: Optional[str],
) -> str:
    _, colloq_expanded = _expand_colloquial_legal_query(raw_query)
    if colloq_expanded:
        return colloq_expanded

    base_q = (router_search_query or raw_query).strip()

    if active_articles and current_code_slug and active_code_slug and current_code_slug != active_code_slug:
        old_title = re.sub(r"^მუხლი\s+[^\s.]+\.?\s*", "", active_articles[0].get("title", "")).strip()
        if old_title and old_title.casefold() in base_q.casefold():
            base_q = raw_query.strip()
        return base_q

    if not active_articles:
        return base_q

    q_lower = raw_query.casefold()
    strict_continuation_markers = (
        "იგივე დანაშაულ", "ამავე დანაშაულ", "ეს დანაშაულ", "ამ დანაშაულ",
        "დამამძიმებელ გარემოებ", "მსუბუქ გარემოებ", "მაგ შემთხვევაში", "ამ შემთხვევაში",
    )
    if any(m in q_lower for m in strict_continuation_markers):
        top_title = active_articles[0].get("title", "")
        clean_title = re.sub(r"^მუხლი\s+[^\s.]+\.?\s*", "", top_title).strip()
        if clean_title and clean_title.casefold() not in base_q.casefold():
            return f"{clean_title} — {base_q}"

    return base_q


def _markers(text: str) -> list[str]:
    found = {f"E{number}" for number in re.findall(r"\[E(\d+)\]", text or "")}
    return sorted(found, key=lambda tag: int(tag[1:]))


def _format_history_for_prompt(chat_history: List[Dict[str, str]], max_turns: int = 3) -> str:
    if not chat_history:
        return "ისტორია ცარიელია (პირველი შეკითხვა)."
    recent = chat_history[-max_turns * 2 :]
    lines: list[str] = []
    for msg in recent:
        role_label = "მომხმარებელი" if msg.get("role") == "user" else "ასისტენტი"
        content = msg.get("content", "").strip()
        if len(content) > 260:
            content = content[:260] + "..."
        lines.append(f"{role_label}: {content}")
    return "\n".join(lines)


class RouterDecision(BaseModel):
    intent: IntentType = Field(
        ...,
        description=(
            "direct_chat (მისალმება/მადლობა), "
            "follow_up_existing (უკვე ნაპოვნი მუხლის გამარტივება/განმარტება), "
            "exact_lookup (კონკრეტული მუხლის ნომრის მოთხოვნა), "
            "hybrid_search (ნებისმიერი სამართლებრივი შეკითხვა ან კაზუსი), "
            "out_of_scope (მხოლოდ კულინარია, ამინდი, სპორტი)."
        ),
    )
    code_slug: Optional[CodeSlugType] = Field(
        default=None,
        description="საქართველოს კოდექსის slug ან null.",
    )
    article_no: Optional[str] = Field(
        default=None,
        description="მუხლის ნომერი (მაგ. '108', '144^1'). მხოლოდ exact_lookup-ის დროს.",
    )
    search_query: str = Field(
        ...,
        description="სუფთა იურიდიული საძიებო შეკითხვა ქართულ ენაზე.",
    )


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


class LegalAgentPipeline:
    def __init__(self, router_path: str = ROUTER_MODEL_NAME):
        self.router_path = str(resolve_file_path(router_path))
        self._router_lock = threading.Lock()
        self._closed = False

        self._init_local_router()
        self.retriever = GeorgianLegalHybridRetriever()

        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("გთხოვთ, მიუთითოთ GOOGLE_API_KEY .env ფაილში.")

        logger.info(f"გენერაციის კასკადური მოდელები: {FALLBACK_GEN_MODELS}")
        self.memory = MemorySaver()
        self.graph = self._build_graph()

    def _init_local_router(self) -> None:
        logger.info(f"იტვირთება ლოკალური კონტექსტური როუტერი GPU-ზე: {Path(self.router_path).name}...")
        self.router_llm = Llama(
            model_path=self.router_path,
            n_gpu_layers=ROUTER_N_GPU_LAYERS,
            n_ctx=ROUTER_N_CTX,
            n_batch=ROUTER_N_BATCH,
            verbose=False,
        )
        logger.info("ლოკალური კონტექსტური როუტერი მზადაა.")

    def router_node(self, state: AgentState) -> Dict[str, Any]:
        query = state["user_query"].strip()
        chat_history = state.get("chat_history") or []
        active_articles = state.get("active_articles") or []
        active_evidence_pack = (state.get("active_evidence_pack") or "").strip()
        active_code_slug = state.get("active_code_slug")
        has_active_evidence = bool(active_evidence_pack and active_articles)

        t0 = time.perf_counter()
        logger.info("[1/5 Router] კონტექსტური ანალიზი და მარშრუტიზაცია...")

        if _is_direct_chat_query(query):
            decision = RouterDecision(
                intent="direct_chat",
                code_slug=active_code_slug if active_code_slug in VALID_CODE_SLUGS else None,
                article_no=None,
                search_query=query,
            )
            elapsed = time.perf_counter() - t0
            return {
                "router_decision": decision.model_dump(),
                "timings": {**state.get("timings", {}), "router_seconds": elapsed},
            }

        if _is_follow_up_existing_query(query, has_active_evidence):
            decision = RouterDecision(
                intent="follow_up_existing",
                code_slug=active_code_slug if active_code_slug in VALID_CODE_SLUGS else None,
                article_no=None,
                search_query=query,
            )
            elapsed = time.perf_counter() - t0
            return {
                "router_decision": decision.model_dump(),
                "timings": {**state.get("timings", {}), "router_seconds": elapsed},
            }

        colloq_code, colloq_sq = _expand_colloquial_legal_query(query)
        explicit_article = _extract_explicit_article_number(query)

        if explicit_article:
            inferred_code = _infer_code_slug(query, colloq_code, active_code_slug)
            decision = RouterDecision(
                intent="exact_lookup",
                code_slug=inferred_code if inferred_code in VALID_CODE_SLUGS else None,
                article_no=explicit_article,
                search_query=query,
            )
            elapsed = time.perf_counter() - t0
            return {
                "router_decision": decision.model_dump(),
                "timings": {**state.get("timings", {}), "router_seconds": elapsed},
            }

        active_titles = ", ".join(
            f"{a.get('code_slug', '')} {a.get('title', '')}" for a in active_articles[:2]
        ) or "არ არის"
        history_snippet = _format_history_for_prompt(chat_history, max_turns=2)

        system_prompt = (
            "You are a strict, ultra-fast Conversational Legal Router for Georgian Law (21 Codes).\n"
            "Output ONLY valid JSON adhering to the schema.\n\n"
            "CRITICAL RULES:\n"
            "1. Questions about age of consent/sex, marriage age, adulthood, crimes, contracts, labor, taxes, customs, elections, administrative fines, constitution ARE LEGAL ('hybrid_search'), NEVER 'out_of_scope'!\n"
            "2. If the user switches to a NEW legal topic, DO NOT mix old topic words into 'search_query'!\n"
            "3. Only combine history into 'search_query' if the user says 'იგივე დანაშაული', 'დამამძიმებელ გარემოებაში', etc.\n\n"
            "FEW-SHOT EXAMPLES:\n"
            "User: 'რა წერია სისხლის სამართლის კოდექსის 108-ე მუხლში?'\n"
            "{\"intent\": \"exact_lookup\", \"code_slug\": \"criminal\", \"article_no\": \"108\", \"search_query\": \"განზრახ მკვლელობა მუხლი 108\"}\n\n"
            "History: Discussed Article 108 (განზრახ მკვლელობა) | User: 'და თუ იგივე დანაშაული ჯგუფურად ან ანგარებით მოხდა?'\n"
            "{\"intent\": \"hybrid_search\", \"code_slug\": \"criminal\", \"article_no\": null, \"search_query\": \"განზრახ მკვლელობა დამამძიმებელ გარემოებაში ჯგუფურად ანგარებით\"}\n\n"
            "History: Discussed Labor Vacation | User: 'საქართველოში რომელი წლიდან არის დაშვებული სექსი?'\n"
            "{\"intent\": \"hybrid_search\", \"code_slug\": \"criminal\", \"article_no\": null, \"search_query\": \"სექსუალური ხასიათის შეღწევა თექვსმეტი წლის ასაკს მიუღწევლის სხეულში მუხლი 140\"}\n\n"
            "History: Discussed Labor Vacation | User: 'და ქორწინება რა ასაკიდან არის დაშვებული?'\n"
            "{\"intent\": \"hybrid_search\", \"code_slug\": \"civil\", \"article_no\": null, \"search_query\": \"საქორწინო ასაკი ქორწინება 18 წლის ასაკიდან მუხლი 1108\"}\n\n"
            "User: 'როგორ გამოვაცხო იმერული ხაჭაპური?'\n"
            "{\"intent\": \"out_of_scope\", \"code_slug\": null, \"article_no\": null, \"search_query\": \"ხაჭაპური\"}"
        )

        user_prompt = (
            f"Active Articles in Memory: {active_titles}\n"
            f"Recent History:\n{history_snippet}\n\n"
            f"Current User Message: {query}"
        )

        try:
            with self._router_lock:
                response = self.router_llm.create_chat_completion(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={
                        "type": "json_object",
                        "schema": RouterDecision.model_json_schema(),
                    },
                    temperature=0.0,
                    max_tokens=95,
                )
            raw_json = response["choices"][0]["message"]["content"]
            decision = RouterDecision.model_validate_json(raw_json)

            inferred_code = _infer_code_slug(query, decision.code_slug, active_code_slug)
            valid_code = inferred_code if inferred_code in VALID_CODE_SLUGS else None

            if decision.intent == "out_of_scope" and (valid_code is not None or colloq_sq is not None):
                decision = decision.model_copy(
                    update={"intent": "hybrid_search", "code_slug": valid_code}
                )

            if decision.intent == "exact_lookup" and not decision.article_no:
                decision = decision.model_copy(
                    update={"intent": "hybrid_search", "article_no": None, "code_slug": valid_code}
                )
            elif decision.intent == "follow_up_existing" and not has_active_evidence:
                decision = decision.model_copy(
                    update={"intent": "hybrid_search", "code_slug": valid_code}
                )

            if decision.intent == "hybrid_search":
                enriched_q = _enrich_contextual_search_query(
                    raw_query=query,
                    router_search_query=colloq_sq or decision.search_query,
                    active_articles=active_articles,
                    current_code_slug=valid_code,
                    active_code_slug=active_code_slug,
                )
                decision = decision.model_copy(
                    update={"search_query": enriched_q, "code_slug": valid_code}
                )

        except Exception as exc:
            logger.warning(f"როუტერის გამონაკლისი: {exc}. ჩაირთო დეტერმინისტული Fallback.")
            inferred_code = _infer_code_slug(query, None, active_code_slug)
            valid_code = inferred_code if inferred_code in VALID_CODE_SLUGS else None
            enriched_q = _enrich_contextual_search_query(
                query, colloq_sq or query, active_articles, valid_code, active_code_slug
            )
            decision = RouterDecision(
                intent="hybrid_search",
                code_slug=valid_code,
                article_no=None,
                search_query=enriched_q,
            )

        elapsed = time.perf_counter() - t0
        logger.info(
            f"როუტერის გადაწყვეტილება: intent={decision.intent} | "
            f"code={decision.code_slug} | art={decision.article_no} | "
            f"search_q='{decision.search_query}' ({elapsed:.2f} წმ)"
        )
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

    def retriever_node(self, state: AgentState) -> Dict[str, Any]:
        decision = state["router_decision"] or {}
        intent = decision.get("intent", "hybrid_search")
        code_slug = decision.get("code_slug")
        article_no = decision.get("article_no")
        search_query = decision.get("search_query", state["user_query"])
        t0 = time.perf_counter()

        logger.info(f"[2/5 Retrieval] რეჟიმი: {intent.upper()} (კოდექსი: {code_slug or 'All 21 Codes'})")

        if intent == "exact_lookup" and article_no:
            articles = self.retriever.direct_lookup(code_slug, article_no)
            if not articles:
                articles = self.retriever.hybrid_search(search_query, top_k=RETRIEVAL_TOP_K, code_slug=code_slug)
        else:
            articles = self.retriever.hybrid_search(search_query, top_k=RETRIEVAL_TOP_K, code_slug=code_slug)

        evidence_pack = self.retriever.build_evidence_pack(articles)
        elapsed = time.perf_counter() - t0
        logger.info(f"ბაზიდან მოძიებულია {len(articles)} მუხლი ({elapsed:.2f} წმ)")

        return {
            "retrieved_articles": articles,
            "evidence_pack": evidence_pack,
            "timings": {**state.get("timings", {}), "retrieval_seconds": elapsed},
        }

    def reuse_context_node(self, state: AgentState) -> Dict[str, Any]:
        active_articles = state.get("active_articles") or []
        active_evidence_pack = state.get("active_evidence_pack") or ""
        logger.info(
            f"[2/5 Context Reuse] ახალი ძიება გამოტოვებულია (0 ms)! "
            f"გამოიყენება მეხსიერებაში არსებული {len(active_articles)} მუხლი."
        )
        return {
            "retrieved_articles": active_articles,
            "evidence_pack": active_evidence_pack,
            "timings": {**state.get("timings", {}), "retrieval_seconds": 0.0},
        }

    def generator_node(self, state: AgentState) -> Dict[str, Any]:
        query = state["user_query"]
        decision = state.get("router_decision") or {}
        intent = decision.get("intent", "hybrid_search")
        evidence_pack = (state.get("evidence_pack") or "").strip()
        articles = state.get("retrieved_articles") or []
        chat_history = state.get("chat_history") or []
        active_articles = state.get("active_articles") or []

        if intent == "out_of_scope":
            return {
                "answer": (
                    "მოცემული შეკითხვა სცდება ჩემს სამართლებრივ კომპეტენციას. "
                    "სისტემაში ინდექსირებულია საქართველოს კონსტიტუცია და 20 მოქმედი კოდექსი (5,624 მუხლი). "
                    "სიამოვნებით გიპასუხებთ საქართველოს კანონმდებლობასთან დაკავშირებულ ნებისმიერ შეკითხვაზე."
                ),
                "timings": {**state.get("timings", {}), "generation_seconds": 0.0},
            }

        if intent == "direct_chat":
            active_note = ""
            if active_articles:
                titles = ", ".join(a.get("title", "") for a in active_articles[:2])
                active_note = f"\n\n*(მიმდინარე დიალოგში განვიხილავთ: {titles})*"
            return {
                "answer": (
                    "გამარჯობა! მე ვარ საქართველოს კანონმდებლობის ინტელექტუალური AI ასისტენტი (Georgian Legal AI).\n\n"
                    "ჩემს საძიებო ბაზაში სრულადაა ინდექსირებული **საქართველოს კონსტიტუცია და 20 მოქმედი კოდექსი (5,624 მუხლი)** — "
                    "მათ შორის სისხლის, სამოქალაქო, შრომის, ადმინისტრაციული, საგადასახადო, საბაჟო და საარჩევნო კოდექსები.\n\n"
                    "შემიძლია:\n"
                    "• მოგიძებნოთ კონკრეტული მუხლები ზუსტი ნომრით (Tier 0 SQL);\n"
                    "• გავაანალიზო რთული სამართლებრივი კაზუსები (Hybrid RRF + Cross-Encoder);\n"
                    "• გესაუბროთ დიალოგის რეჟიმში და ყველა დასკვნა დავასაბუთო ზუსტი `[E1]`, `[E2]` წყაროებით."
                    f"{active_note}"
                ),
                "timings": {**state.get("timings", {}), "generation_seconds": 0.0},
            }

        if not evidence_pack or not articles:
            return {
                "answer": (
                    "მოძიებულ წყაროებში შესაბამისი სამართლებრივი ნორმა ვერ მოიძებნა. "
                    "არსებული მონაცემების საფუძველზე დასკვნას ვერ გავაკეთებ."
                ),
                "timings": {**state.get("timings", {}), "generation_seconds": 0.0},
            }

        history_block = _format_history_for_prompt(chat_history, max_turns=3)

        gen_system_prompt = (
            "შენ ხარ საქართველოს კანონმდებლობის მაღალკვალიფიციური კონვერსაციული იურიდიული AI ასისტენტი.\n"
            "აწარმოე ბუნებრივი, თანმიმდევრული დიალოგი მომხმარებელთან ჩატის ისტორიის გათვალისწინებით, "
            "მაგრამ ყველა სამართლებრივი ფაქტი დააფუძნე მხოლოდ და მხოლოდ მოწოდებულ EVIDENCE PACK-ზე.\n\n"
            "მკაცრი წესები:\n"
            "1. ყოველი იურიდიული დასკვნა, პუნქტი, სანქცია ან განმარტება დაასაბუთე შესაბამისი მარკერით: [E1], [E2] და ა.შ.\n"
            "2. კატეგორიულად აკრძალულია მუხლების, ციფრების ან ნორმების გამოგონება საკუთარი მეხსიერებიდან.\n"
            "3. არ გამოიყენო ისეთი მარკერი, რომელიც ქვემოთ მოცემულ მტკიცებულებებში არ არსებობს.\n"
            "4. ყურადღება მიაქციე მხოლოდ იმ მუხლებს, რომლებიც პირდაპირ პასუხობს მიმდინარე შეკითხვას; არარელევანტური კანდიდატი მუხლები დააიგნორე და მათზე კომენტარი არ გააკეთო."
        )

        user_content = (
            f"=== დიალოგის წინა ისტორია ===\n{history_block}\n\n"
            f"=== მომხმარებლის მიმდინარე შეკითხვა ===\n{query}\n\n"
            f"{evidence_pack}\n\n"
            "ჩამოაყალიბე პროფესიული, კონტექსტზე მორგებული პასუხი [E1], [E2] ციტირებით:"
        )

        messages = [
            SystemMessage(content=gen_system_prompt),
            HumanMessage(content=user_content),
        ]

        for model_name in FALLBACK_GEN_MODELS:
            logger.info(f"[3/5 Generation] ცდა მოდელით: {model_name}...")
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
                    logger.info(f"გენერაცია დასრულდა წარმატებით ({elapsed:.2f} წმ) [{model_name}]")
                    return {
                        "answer": extract_text_content(response.content),
                        "timings": {**state.get("timings", {}), "generation_seconds": elapsed},
                    }
                except Exception as exc:
                    elapsed = time.perf_counter() - t0
                    err_str = str(exc)
                    logger.warning(f"{model_name} (ცდა {attempt}/2 ვერ შედგა {elapsed:.2f} წმ-ში): {err_str[:80]}...")
                    if "503" in err_str or "429" in err_str or "UNAVAILABLE" in err_str:
                        time.sleep(2.5)
                    else:
                        break

        return {
            "answer": "პასუხის გენერაცია დროებით ვერ მოხერხდა API გადატვირთვის გამო. მოძიებული მუხლები იხილეთ ზემოთ.",
            "timings": {**state.get("timings", {}), "generation_seconds": 0.0},
        }

    def citation_guard_node(self, state: AgentState) -> Dict[str, Any]:
        t0 = time.perf_counter()
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
                "timings": {**state.get("timings", {}), "guard_seconds": 0.0},
            }

        cited = _markers(answer)
        valid = _markers(evidence_pack)
        hallucinated = [tag for tag in cited if tag not in valid]
        missing_citations = bool(evidence_pack) and not cited
        marker_integrity = bool(cited) and not hallucinated
        is_valid = marker_integrity and not missing_citations

        final_answer = answer
        if hallucinated:
            message = f"დაფიქსირდა გამოგონილი ციტატები: {', '.join(hallucinated)}"
            final_answer += f"\n\n[სისტემური გაფრთხილება]: დაუდასტურებელი მტკიცებულებები: {', '.join(hallucinated)}."
        elif missing_citations:
            message = "პასუხში ციტირების მარკერები არ არის მითითებული."
            final_answer += "\n\n[სისტემური გაფრთხილება]: პასუხში წყაროს მარკერები არ არის მითითებული."
        else:
            message = "ყველა ციტატა ვალიდურია."

        elapsed = time.perf_counter() - t0
        return {
            "citation_status": {
                "is_valid": is_valid,
                "marker_integrity": marker_integrity,
                "cited_markers": cited,
                "valid_markers": valid,
                "hallucinated_markers": hallucinated,
                "message": message,
            },
            "answer": final_answer,
            "timings": {**state.get("timings", {}), "guard_seconds": elapsed},
        }

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