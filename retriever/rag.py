import json
import logging
import os
import re
import threading
import asyncio
from typing import Any

import numpy as np
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Project root: .../retriever/rag.py -> parent -> parent
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(_PROJECT_ROOT, "data")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "amenify.index")
METADATA_PATH = os.path.join(DATA_DIR, "metadata.json")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "gemini-2.5-flash"
SIMILARITY_THRESHOLD = 1.35
MAX_CONTEXT_BLOCKS = 3
FAISS_PROBE_K = 20
LLM_MAX_OUTPUT_TOKENS = 420
LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", "25"))

# Lazy-loaded heavy resources (nothing loaded at import time)
_resources_lock = threading.Lock()
_embedding_model: Any = None
_index: Any = None
_metadata: list | None = None
_resources_load_error: str | None = None
_gemini_configured = False
_preload_started = False


def load_resources() -> str | None:
    """
    Load SentenceTransformer, FAISS index, and metadata once (thread-safe).
    Configure Gemini API when a key is present.
    Returns None on success, or a short user-facing error message on failure.
    """
    global _embedding_model, _index, _metadata, _resources_load_error, _gemini_configured

    with _resources_lock:
        if _resources_load_error is not None:
            return _resources_load_error
        if _embedding_model is not None and _index is not None and _metadata is not None:
            return None  # already loaded successfully

        if not os.path.isfile(FAISS_INDEX_PATH) or not os.path.isfile(METADATA_PATH):
            _resources_load_error = (
                "Knowledge base files are missing on the server. "
                "Please ensure data/amenify.index and data/metadata.json are deployed."
            )
            return _resources_load_error

        try:
            # Heavy imports only when needed (keeps uvicorn import + bind fast for Render)
            import faiss
            from sentence_transformers import SentenceTransformer

            emb = SentenceTransformer(EMBEDDING_MODEL)
            idx = faiss.read_index(FAISS_INDEX_PATH)
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                meta = json.load(f)

            if idx.ntotal != len(meta):
                logger.warning(
                    "FAISS ntotal (%s) != metadata rows (%s); re-run ingestion.",
                    idx.ntotal,
                    len(meta),
                )

            api_key = os.getenv("GOOGLE_GEMINI_API_KEY")
            if api_key and not _gemini_configured:
                import google.generativeai as genai

                genai.configure(api_key=api_key)
                _gemini_configured = True

            _embedding_model = emb
            _index = idx
            _metadata = meta
            logger.info("RAG resources loaded (%s vectors).", idx.ntotal)
            return None

        except Exception:
            logger.exception("Failed to load RAG resources")
            _embedding_model = None
            _index = None
            _metadata = None
            _resources_load_error = "The assistant could not start its search index. Please try again later."
            return _resources_load_error


def is_resources_ready() -> bool:
    return _embedding_model is not None and _index is not None and _metadata is not None


def preload_resources_in_background() -> bool:
    """
    Kick off non-blocking resource preload once per process.
    Returns True when a preload thread was started.
    """
    global _preload_started
    with _resources_lock:
        if _preload_started or is_resources_ready() or _resources_load_error is not None:
            return False
        _preload_started = True

    def _runner():
        load_resources()

    t = threading.Thread(target=_runner, name="rag-preload", daemon=True)
    t.start()
    return True


def get_query_embedding(query: str) -> list[float]:
    if _embedding_model is None:
        raise RuntimeError("Embedding model not loaded; call load_resources() first.")
    embedding = _embedding_model.encode(query, convert_to_tensor=False)
    return embedding.tolist()


def _chunk_key(text: str) -> str:
    return " ".join((text or "").split())


def dedupe_contexts(contexts: list) -> list:
    """Remove blocks with identical normalized text (e.g. repeated footers)."""
    seen: set[str] = set()
    out = []
    for c in contexts:
        k = _chunk_key(c.get("text", ""))
        if len(k) < 40 or k in seen:
            continue
        seen.add(k)
        out.append(c)
    return out


# Phrases repeated in scraped header/nav (Squarespace-style menus)
_NAV_PHRASES = (
    "Skip to Content",
    "Open Menu",
    "Close Menu",
    "Product Catalog",
    "Folder: Our Services",
    "Home Interior Services",
    "Office Interiors & Furnishing",
    "Office Interiors and Furnishing",
    "Modular Kitchen Blog",
    "0 0 login",
    "login",
)

# First substantial body markers (trim junk before these when far into the string)
_CONTENT_START_MARKERS = (
    r"Built Your Perfect Modular Kitchen",
    r"Your Go-To Turnkey Solution",
    r"Dream Big, We Furnish",
    r"At Amenify, we understand that your office",
    r"At Amenify, we understand that your home",
    r"At Amenify, we believe the kitchen",
    r"At Amenify, we",
    r"Interior Designing Solutions for Every Home",
    r"Interior Designing Solutions",
    r"Why Choose Amenify",
    r"Why Hire Our",
    r"How Amenify",
    r"Comprehensive Furnishing",
    r"We start with a detailed consultation",
    r"Amenify specializes in",
    r"Amenify specialise in",
    r"Amenify specialize in",
)


def _strip_repeated_nav(t: str) -> str:
    for _ in range(4):
        for phrase in _NAV_PHRASES:
            t = re.sub(re.escape(phrase), " ", t, flags=re.IGNORECASE)
    t = re.sub(
        r"(?:Our Services\s+){2,}",
        " ",
        t,
        flags=re.IGNORECASE,
    )
    t = re.sub(r"\bBack\s+", " ", t, flags=re.IGNORECASE)
    return t


def _trim_to_body_start(t: str) -> str:
    """Drop long leading title+menu runs; keep real page copy."""
    if len(t) < 80:
        return t
    cut = None
    for pat in _CONTENT_START_MARKERS:
        m = re.search(pat, t, re.IGNORECASE)
        if m and m.start() >= 35:
            if cut is None or m.start() < cut:
                cut = m.start()
    if cut is not None:
        t = t[cut:].strip()
    return t


_COMMON_FIRST_WORDS = frozenset(
    """a an as at be by do if in is it of on or so to up us we all and any are but can end for
    get got had has her him his how its may new not now off old one our out own say she ten the
    too try two use was way who why yes yet you your amenify their there these this that what when
    where which while using through during without within interior office offering tailored every
    select modern designed comprehensive experience leveraging overview environment creating bringing
    somewhere building services solutions project projects space spaces team design home corporate
    small large with from into about after also before being both each fewer more most much such
    than them then they those under very well were will work works working""".split()
)


def _trim_leading_chunk_fragment(t: str) -> str:
    """Remove orphan starts like 'ironment that...' from split chunks."""
    if not t or t[0].isupper():
        return t
    parts = t.split(maxsplit=1)
    if not parts:
        return t
    first = parts[0].lower().strip(",.;:\"'")
    if len(first) > 10 or first in _COMMON_FIRST_WORDS:
        return t
    m = re.search(r"\.\s+[A-Z]", t)
    if m is not None and m.start() < 280:
        return t[m.start() + 2 :].strip()
    return t


def sanitize_context_chunk(text: str) -> str:
    """Strip nav, carousels, and CTAs so the model answers from real copy."""
    if not text:
        return text
    t = text
    t = _strip_repeated_nav(t)
    t = re.sub(
        r"\bBook Your\s+[^.!?]{0,120}?Today!\s*",
        " ",
        t,
        flags=re.IGNORECASE,
    )
    t = re.sub(r"(?:View Project\s+3D VR\s*)+", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"\bView Project\b", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"\bView Design\b", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"\bRead More\b", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"\bKnow More\b", " ", t, flags=re.IGNORECASE)
    t = _trim_to_body_start(t)
    t = _trim_leading_chunk_fragment(t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def expand_query_for_embedding(query: str) -> str:
    """Bias retrieval toward the right page (decor spelling, kitchen, office)."""
    q = (query or "").strip()
    if not q:
        return q
    ql = q.lower()
    extra: list[str] = []
    if "decore" in ql or ("decor" in ql and "office" not in ql):
        extra.append("home interior designing furnishing residential")
    if "office" in ql and ("decore" in ql or "decor" in ql or "interior" in ql):
        extra.append("office interior furnishing decorating services")
    if "kitchen" in ql or "modular" in ql:
        extra.append("modular kitchen design services")
    if "office" in ql and (
        "service" in ql or "interior" in ql or "decor" in ql or "decore" in ql or "tell" in ql
    ):
        extra.append("office interior furnishing turnkey fit-out")
    if not extra:
        return q
    return f"{q} {' '.join(extra)}"


def _norm_query(q_raw: str) -> str:
    q = (q_raw or "").strip().lower()
    return re.sub(r"\s+", " ", q)


def handle_static_intent(q_raw: str) -> str | None:
    """
    Short, reliable answers without RAG (greetings, capabilities, contact, etc.).
    """
    q = _norm_query(q_raw)
    qn = q.replace("ypu", "you").replace("wat ", "what ")

    if not q:
        return None

    if q in {"hi", "hello", "hey", "hii", "howdy", "namaste"} or (
        len(q) <= 20 and q.split()[0] in {"hi", "hello", "hey"}
    ):
        return (
            "Hi—I'm Amenify's assistant. Ask about home/office interiors, modular kitchens, furniture, EMI, warranty, or contact details."
        )

    if "who are you" in q or q in {"who are you?", "who r u", "who are u"}:
        return "I'm Amenify's assistant for interior services, process, and policies from our published information."

    if any(
        x in qn
        for x in (
            "what you do",
            "what do you do",
            "what can you do",
            "what do you offer",
            "what you offer",
        )
    ) or (len(q) <= 28 and "what" in qn and "do" in qn and "you" in qn):
        return (
            "Amenify delivers end-to-end interiors: home and office design, modular kitchens/wardrobes, furniture, "
            "3D/VR previews, in-house factory execution, warranty, EMI, and post-completion support."
        )

    if q in {"offers", "offer", "services", "your services", "service"}:
        return (
            "Services include home/office design and build-out, modular kitchens, custom furniture, turnkey renovation, "
            "3D visualization, transparent pricing, EMI options, and after-sales support."
        )

    if q == "emi" or (len(q) <= 40 and "emi" in q and "option" in q):
        return (
            "Amenify promotes easy EMI with transparent pricing and a best-price assurance; exact plans depend on your quote—confirm when you book."
        )

    if q in {"location", "address", "where are you", "office location"} or "where are you located" in q:
        return (
            "Registered: Emaar Digital Greens, Sector 61, Gurugram, Haryana 122001. "
            "NCR (Gurugram, Delhi, Noida) plus Bangalore, Hyderabad, Mumbai, Pune, Kolkata, Chandigarh, Ahmedabad."
        )

    if q in {"contact", "contact us", "reach you", "phone", "email", "call you"} or (
        len(q) <= 35 and ("contact" in q or q == "phone" or q == "email")
    ):
        return (
            "Email india@amenify.com, phone +91 98731 23716. Address: Emaar Digital Greens, Sector 61, Gurugram, Haryana 122001. "
            "Use Contact/Login on the site to book or follow up."
        )

    if len(q) <= 90 and re.search(
        r"\b3d\b|\bvr\b|virtual\s+reality|visuali[sz]ation", q
    ):
        return (
            "Amenify uses VR-driven 3D previews and AR-style catalogs to finalize materials, with on-site capture and portal updates "
            "so you can align the design before execution."
        )

    return None


def polish_response(contexts: list, query: str, max_sentences: int = 4) -> str:
    """Readable answer from retrieved blocks without repeating the same sentence."""
    if not contexts:
        return "I don't know."

    noise_suffixes = (
        "View Project", "View Design", "Read More", "Know More",
        "Open Menu", "Close Menu", "Skip to Content",
    )
    seen: set[str] = set()
    sentences: list[str] = []

    for ctx in contexts:
        text = sanitize_context_chunk((ctx.get("text") or "").strip())
        parts = re.split(r"(?<=[.!?])\s+|\n+", text)
        for p in parts:
            s = p.strip()
            if len(s) < 35:
                continue
            if s.rstrip().endswith("?"):
                continue
            if any(s.rstrip(".").endswith(x) for x in noise_suffixes):
                continue
            sig = s.lower()[:240]
            if sig in seen:
                continue
            seen.add(sig)
            if s[-1] not in ".!?":
                s += "."
            sentences.append(s)
            if len(sentences) >= max_sentences:
                break
        if len(sentences) >= max_sentences:
            break

    if not sentences:
        blob = _chunk_key(sanitize_context_chunk(contexts[0].get("text") or ""))
        if len(blob) > 360:
            return blob[:360].rsplit(" ", 1)[0] + "..."
        return blob or "I don't know."

    return " ".join(sentences)


def retrieve_contexts(
    query: str,
    top_k: int | None = None,
    threshold: float | None = None,
    max_blocks: int | None = None,
):
    """Retrieve FAISS neighbors, filter by L2 distance, dedupe identical text."""
    if _index is None or _metadata is None or len(_metadata) == 0:
        return []

    k = min(top_k or FAISS_PROBE_K, len(_metadata))
    thr = SIMILARITY_THRESHOLD if threshold is None else threshold
    cap = max_blocks if max_blocks is not None else MAX_CONTEXT_BLOCKS

    q_emb = get_query_embedding(expand_query_for_embedding(query))
    q_emb_matrix = np.array([q_emb]).astype("float32")

    distances, indices = _index.search(q_emb_matrix, k)

    contexts = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1 or not (dist < thr):
            continue
        ctx = _metadata[idx].copy()
        ctx["distance"] = float(dist)
        contexts.append(ctx)
        logger.debug("Retrieved chunk distance=%.4f source=%s", dist, ctx.get("source", "")[:60])

    contexts = dedupe_contexts(contexts)[:cap]

    if not contexts and len(distances[0]):
        logger.debug(
            "No contexts under threshold; best_distance=%.4f threshold=%s",
            float(distances[0][0]),
            thr,
        )

    return contexts

async def generate_chat_stream(query: str, chat_history: list):
    """
    RAG Logic with Strict Anti-Hallucination.
    Heavy resources load only when a query needs retrieval (not for static intents).
    """
    static = handle_static_intent(query)
    if static:
        yield static
        return

    load_err = load_resources()
    if load_err:
        yield load_err
        return

    contexts = retrieve_contexts(query)

    if not contexts:
        logger.debug("No contexts for query=%s", query[:80])
        yield "I don't know."
        return

    blocks = [sanitize_context_chunk(ctx["text"]) for ctx in contexts]
    context_text = "\n\n".join(blocks)

    system_prompt = f"""You are Amenify's assistant.

Rules:
- Use ONLY sentences you can support with <context>. If it does not answer the question, say exactly: I don't know.
- Be direct: 1–3 short sentences for a simple question; at most 4 if the user asked for a list or multiple points.
- Answer what was asked—no preamble ("Certainly"), no closing offers, no lines like "Book Your … Today!".
- Do not copy FAQ question lines from context (lines ending with ?); turn facts into statements. End with a full stop, not a question.
- Do not invent numbers, cities, or policies not written in <context>.
- No URLs, no page titles with "— Amenify India", no "Skip to Content", menu labels, or carousel wording. Do not repeat the same idea twice.

<context>
{context_text}
</context>

Question: {query}"""

    try:
        import google.generativeai as genai

        def _generate_once() -> str:
            model = genai.GenerativeModel(LLM_MODEL)
            response = model.generate_content(
                system_prompt,
                stream=False,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.0,
                    max_output_tokens=LLM_MAX_OUTPUT_TOKENS,
                ),
                request_options={"timeout": LLM_TIMEOUT_SECONDS},
            )
            return (getattr(response, "text", "") or "").strip()

        text = await asyncio.wait_for(
            asyncio.to_thread(_generate_once),
            timeout=LLM_TIMEOUT_SECONDS + 3,
        )

        if text:
            yield text
        else:
            yield polish_response(contexts, query)
    except Exception as e:
        logger.exception("LLM generation failed")
        if contexts:
            yield polish_response(contexts, query)
        else:
            yield "I don't know."


# Backwards compatibility (e.g. scripts); prefer load_resources()
load_vector_db = load_resources
