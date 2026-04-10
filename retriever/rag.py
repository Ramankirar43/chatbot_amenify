import os
import json
import faiss
from openai import AsyncOpenAI
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "amenify.index")
METADATA_PATH = os.path.join(DATA_DIR, "metadata.json")
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
SIMILARITY_THRESHOLD = 1.0 # This threshold depends on distance metric. FlatL2 distance. L2 distance lower is better. Assuming threshold around 1.0-1.5

# We need a synchronous client for simple embedding or reuse the async one
aclient = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

index = None
metadata = []

def load_vector_db():
    global index, metadata
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(METADATA_PATH):
        index = faiss.read_index(FAISS_INDEX_PATH)
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        print("Vector DB loaded securely.")
    else:
        print("WARNING: Vector DB files not found. Please run ingest.py first.")

async def get_query_embedding(query: str) -> list[float]:
    response = await aclient.embeddings.create(
        input=[query],
        model=EMBEDDING_MODEL
    )
    return response.data[0].embedding

async def retrieve_contexts(query: str, top_k: int = 3, threshold: float = 1.5):
    if index is None or not metadata:
        return []

    q_emb = await get_query_embedding(query)
    q_emb_matrix = np.array([q_emb]).astype("float32")

    # D is distances (L2), I is indices
    distances, indices = index.search(q_emb_matrix, top_k)
    
    contexts = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx != -1 and dist < threshold:
            contexts.append(metadata[idx])
    return contexts

async def generate_chat_stream(query: str, chat_history: list):
    """
    RAG Logic with Strict Anti-Hallucination
    Returns an async generator yielding chunks of response.
    """
    contexts = await retrieve_contexts(query)
    
    if not contexts:
        yield "I don't know."
        return
    
    # Build prompt
    context_text = "\n\n".join([f"Source: {ctx['source']}\nContent: {ctx['text']}" for ctx in contexts])
    
    system_prompt = f"""You are an expert customer support agent for Amenify (an interior design and furnishing company).
Your task is to answer user queries using ONLY the provided context below.
If the answer cannot be deduced from the provided context, you MUST answer EXACTLY with "I don't know."
Do NOT guess. Do NOT make up information.
When you answer, briefly mention the source URL if applicable.

<context>
{context_text}
</context>"""

    messages = [{"role": "system", "content": system_prompt}]
    
    # Append limited chat history (last 5 interactions)
    for msg in chat_history[-5:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
        
    messages.append({"role": "user", "content": query})

    stream = await aclient.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0.0,
        stream=True
    )

    sources = set([c['source'] for c in contexts])
    source_attribution = f"\n\n*(Sources: {', '.join(sources)})*"

    async for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
            
    yield source_attribution
