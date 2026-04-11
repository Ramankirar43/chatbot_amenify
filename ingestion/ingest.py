import os
import re
import json
import unicodedata
from collections import Counter
import numpy as np
import requests
import faiss
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

URLS = [
    "https://www.amenify.in/",
    "https://www.amenify.in/home-interior-designing-services",
    "https://www.amenify.in/office-interiors-furnishing",
    "https://www.amenify.in/modular-kitchen-design-services",
    "https://www.amenify.in/blog",
]

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "amenify.index")
METADATA_PATH = os.path.join(DATA_DIR, "metadata.json")


def scrape_text_from_url(url: str) -> str:
    print(f"Scraping {url}...")
    try:
        response = requests.get(
            url,
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        response.raise_for_status()
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return ""

    soup = BeautifulSoup(response.content, "html.parser")
    for element in soup(["script", "style", "noscript"]):
        element.decompose()

    return soup.get_text(separator=" ", strip=True)


def _normalize_chunk_key(text: str) -> str:
    """
    Canonical key for deduplication: NFKC + strip invisible junk + collapse whitespace.
    Two chunks that differ only by unicode spaces or ZWSP must map to the same key.
    """
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", text)
    for ch in ("\ufeff", "\u200b", "\u200c", "\u200d"):
        t = t.replace(ch, "")
    t = t.replace("\u00a0", " ")
    return " ".join(t.split())


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """
    Word-safe windows; overlap always advances (no 1-char sliding that duplicates footers).
    Optionally extend end to the next sentence break within a short lookahead.
    """
    chunks: list[str] = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)

        if end < n and text[end] != " ":
            space_idx = text.rfind(" ", start, end)
            if space_idx != -1 and space_idx > start:
                end = space_idx

        if end < n:
            lookahead = text[end : min(end + 100, n)]
            sent_m = re.search(r"[.!?]\s+", lookahead)
            if sent_m is not None and sent_m.end() <= 90:
                end = end + sent_m.end()

        chunk = text[start:end].strip()
        if len(chunk) >= 45:
            chunks.append(chunk)

        if end >= n:
            break

        new_start = end - overlap
        if new_start <= start:
            new_start = end
        start = new_start

    return chunks


def dedupe_chunks(chunks: list[str]) -> list[str]:
    """Exact dedupe within one page (normalized key)."""
    seen: set[str] = set()
    out: list[str] = []
    for c in chunks:
        key = _normalize_chunk_key(c)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def _final_dedupe_globally(
    all_chunks: list[str], metadata: list[dict]
) -> tuple[list[str], list[dict], int]:
    """
    Safety net: drop any row whose normalized text was already kept (fixes bad historical runs).
    Returns (chunks, metadata, number_removed).
    """
    seen: set[str] = set()
    out_c: list[str] = []
    out_m: list[dict] = []
    removed = 0
    for ch, meta in zip(all_chunks, metadata):
        k = _normalize_chunk_key(ch)
        if k in seen:
            removed += 1
            continue
        seen.add(k)
        out_c.append(ch)
        out_m.append(meta)
    return out_c, out_m, removed


def get_embeddings(texts: list[str]) -> list[list[float]]:
    return model.encode(texts, convert_to_tensor=False).tolist()


def ingest_data():
    os.makedirs(DATA_DIR, exist_ok=True)

    all_chunks: list[str] = []
    metadata: list[dict] = []

    seen_globally: set[str] = set()
    per_url_kept: dict[str, int] = {u: 0 for u in URLS}
    failed_urls: list[str] = []

    for url in URLS:
        page_text = scrape_text_from_url(url)
        if not page_text:
            failed_urls.append(url)
            continue

        chunks = dedupe_chunks(chunk_text(page_text))
        idx = 0
        for chunk in chunks:
            gkey = _normalize_chunk_key(chunk)
            if gkey in seen_globally:
                continue
            seen_globally.add(gkey)
            all_chunks.append(chunk)
            metadata.append(
                {
                    "source": url,
                    "chunk_id": f"{url}_chunk_{idx}",
                    "text": chunk,
                }
            )
            idx += 1
            per_url_kept[url] = per_url_kept.get(url, 0) + 1

    if not all_chunks:
        print("No data extracted.")
        if failed_urls:
            print("Failed or empty URLs:", ", ".join(failed_urls))
        return

    n_pre_final = len(all_chunks)
    all_chunks, metadata, removed = _final_dedupe_globally(all_chunks, metadata)
    if removed:
        print(f"Final dedupe removed {removed} duplicate rows (keys matched earlier rows).")

    per_url_final = Counter(m["source"] for m in metadata)

    print(f"Extracted {len(all_chunks)} unique chunks (from {n_pre_final} collected). Per URL:")
    for u in URLS:
        n = per_url_final.get(u, 0)
        status = "OK" if n > 0 and u not in failed_urls else ("FETCH FAILED" if u in failed_urls else "no unique chunks")
        print(f"  {n:3d}  {status:16s}  {u}")
    if failed_urls:
        print("Fetch failures:", ", ".join(failed_urls))

    # Sanity: no exact duplicate keys in output
    keys = [_normalize_chunk_key(c) for c in all_chunks]
    if len(keys) != len(set(keys)):
        print("ERROR: Duplicate keys still present after dedupe; aborting write.")
        return

    print("Generating embeddings...")
    batch_size = 20
    all_embeddings: list[list[float]] = []

    for i in range(0, len(all_chunks), batch_size):
        print(f"Embedding batch {i // batch_size + 1}...")
        batch = all_chunks[i : i + batch_size]
        try:
            all_embeddings.extend(get_embeddings(batch))
        except Exception as e:
            print("Embedding error:", e)

    if not all_embeddings:
        print("ERROR: No embeddings generated.")
        return

    embedding_matrix = np.array(all_embeddings).astype("float32")
    if embedding_matrix.ndim != 2:
        print(f"ERROR: Invalid embedding matrix shape {embedding_matrix.shape}.")
        return

    embedding_dim = embedding_matrix.shape[1]
    print("Building FAISS index...")
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(embedding_matrix)
    faiss.write_index(index, FAISS_INDEX_PATH)

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    if index.ntotal != len(metadata):
        print(f"ERROR: FAISS ntotal ({index.ntotal}) != metadata ({len(metadata)}).")
    else:
        print(f"Ingestion complete. {index.ntotal} vectors; metadata has no exact duplicate keys.")


if __name__ == "__main__":
    ingest_data()
