import os
import json
import numpy as np
import requests
import faiss
from bs4 import BeautifulSoup
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Configure OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
EMBEDDING_MODEL = "text-embedding-3-small"

URLS = [
    "https://www.amenify.in/",
    "https://www.amenify.in/home-interior-designing-services",
    "https://www.amenify.in/office-interiors-furnishing",
    "https://www.amenify.in/modular-kitchen-design-services",
    "https://www.amenify.in/blog"
]

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "amenify.index")
METADATA_PATH = os.path.join(DATA_DIR, "metadata.json")

def scrape_text_from_url(url: str) -> str:
    print(f"Scraping {url}...")
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return ""

    soup = BeautifulSoup(response.content, 'html.parser')

    # Remove unwanted elements
    for element in soup(["nav", "footer", "script", "style", "header", "noscript"]):
        element.decompose()

    # Extract text from remaining tags
    text = soup.get_text(separator=' ', strip=True)
    return text

def chunk_text(text: str, chunk_size: int = 1500, overlap: int = 200) -> list[str]:
    # Simple character-level chunking roughly equating to 200-400 tokens
    # Average ~4 characters per token
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        # Optional: adjust end to nearest space to avoid breaking words
        if end < len(text) and text[end] != ' ':
            space_idx = text.rfind(' ', start, end)
            if space_idx != -1:
                end = space_idx
        chunk = text[start:end].strip()
        if len(chunk) > 50: # Skip very small fragments
            chunks.append(chunk)
        start = end - overlap
    return chunks

def get_embeddings(texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(
        input=texts,
        model=EMBEDDING_MODEL
    )
    return [data.embedding for data in response.data]

def ingest_data():
    os.makedirs(DATA_DIR, exist_ok=True)
    
    all_chunks = []
    metadata = []

    for url in URLS:
        page_text = scrape_text_from_url(url)
        if not page_text:
            continue
        
        chunks = chunk_text(page_text)
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
    
    # Process embeddings in batches to avoid rate limits / large payloads
    batch_size = 100
    all_embeddings = []
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i+batch_size]
        embeddings = get_embeddings(batch)
        all_embeddings.extend(embeddings)

    embedding_matrix = np.array(all_embeddings).astype("float32")
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
