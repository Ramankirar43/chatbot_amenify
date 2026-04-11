import os
import json
import numpy as np
import requests
import faiss
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# Configure local embeddings
print("Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

URLS = [
    "https://www.amenify.in/",
    "https://www.amenify.in/home-interior-designing-services",
    "https://www.amenify.in/office-interiors-furnishing",
    "https://www.amenify.in/modular-kitchen-design-services",
    "https://www.amenify.in/furniture-store",
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
            headers={"User-Agent": "Mozilla/5.0"}
        )
        response.raise_for_status()
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return ""

    soup = BeautifulSoup(response.content, 'html.parser')

    # Remove unwanted elements
    for element in soup(["nav", "footer", "script", "style", "header", "noscript"]):
        element.decompose()

    text = soup.get_text(separator=' ', strip=True)
    return text


def chunk_text(text: str, chunk_size: int = 1500, overlap: int = 200) -> list[str]:
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        # avoid cutting words
        if end < len(text) and text[end] != ' ':
            space_idx = text.rfind(' ', start, end)
            if space_idx != -1:
                end = space_idx

        chunk = text[start:end].strip()

        if len(chunk) > 50:
            chunks.append(chunk)

        # FIX: prevent infinite/slow loop
        start = max(end - overlap, start + 1)

    return chunks


def get_embeddings(texts: list[str]) -> list[list[float]]:
    embeddings = model.encode(texts, convert_to_tensor=False)
    return embeddings.tolist()


def ingest_data():
    os.makedirs(DATA_DIR, exist_ok=True)

    all_chunks = []
    metadata = []

    for url in URLS:
        page_text = scrape_text_from_url(url)
        if not page_text:
            continue

        chunks = chunk_text(page_text)

        # Keep more chunks (increased from 5 to 20 for richer data)
        chunks = chunks[:20]

        for idx, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            metadata.append({
                "source": url,
                "chunk_id": f"{url}_chunk_{idx}",
                "text": chunk
            })

    if not all_chunks:
        print("No data extracted.")
        return

    print(f"Extracted {len(all_chunks)} chunks total. Generating embeddings...")

    # FIX: smaller batch size
    batch_size = 20
    all_embeddings = []

    for i in range(0, len(all_chunks), batch_size):
        print(f"Embedding batch {i//batch_size + 1}...")
        batch = all_chunks[i:i+batch_size]

        try:
            embeddings = get_embeddings(batch)
            all_embeddings.extend(embeddings)
        except Exception as e:
            print("Embedding error:", e)
            continue

    if not all_embeddings:
        print("ERROR: No embeddings generated. Please check the embedding model.")
        return

    embedding_matrix = np.array(all_embeddings).astype("float32")
    
    if embedding_matrix.ndim != 2:
        print(f"ERROR: Invalid embedding matrix shape {embedding_matrix.shape}. Expected 2D array.")
        return
    
    embedding_dim = embedding_matrix.shape[1]

    print("Building FAISS index...")
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(embedding_matrix)

    faiss.write_index(index, FAISS_INDEX_PATH)

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("Ingestion complete. Vector DB saved.")


if __name__ == "__main__":
    ingest_data()