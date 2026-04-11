import os
import json
import faiss
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

load_dotenv()

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "amenify.index")
METADATA_PATH = os.path.join(DATA_DIR, "metadata.json")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "gemini-2.5-flash"  # Using latest available model
SIMILARITY_THRESHOLD = 1.0

# Configure Google Gemini
api_key = os.getenv("GOOGLE_GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

# Load embedding model
print("Loading embedding model...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

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

def list_available_models():
    """Debug: List available Gemini models"""
    try:
        models = genai.list_models()
        print("\n=== Available Gemini Models ===")
        for model in models:
            print(f"  - {model.name}")
        print("================================\n")
    except Exception as e:
        print(f"Could not list models: {e}")

def get_query_embedding(query: str) -> list[float]:
    embedding = embedding_model.encode(query, convert_to_tensor=False)
    return embedding.tolist()

def polish_response(contexts: list, query: str, max_lines: int = 5) -> str:
    """Create a polished 5-line summary from retrieved contexts."""
    if not contexts:
        return "I don't know."
    
    # Extract key sentences from best matching context
    best_context = contexts[0]['text'].strip()
    
    # Split into sentences and clean up
    sentences = []
    for sent in best_context.split('.'):
        sent = sent.strip()
        # Filter out very short or low-quality sentences
        if len(sent) > 20 and not sent.endswith(('View', 'More', 'Know')):
            sentences.append(sent)
    
    if not sentences:
        # Fallback: use first 300 chars as one summary line
        if len(best_context) > 200:
            return best_context[:200].strip() + "..."
        return best_context
    
    # Select up to 5 most relevant sentences
    selected_sentences = sentences[:max_lines]
    
    # Join sentences and create polished response
    polished_lines = []
    for sent in selected_sentences:
        # Clean up sentence
        sent = sent.strip()
        if sent and not sent.endswith('.'):
            sent += '.'
        if len(sent) > 5:  # Only add meaningful sentences
            polished_lines.append(sent)
    
    if not polished_lines:
        # Emergency fallback
        return best_context[:250] + "..." if len(best_context) > 250 else best_context
    
    # Ensure we have max 5 lines
    polished = "\n".join(polished_lines[:5])
    
    # Cap total length at 600 chars
    if len(polished) > 600:
        polished = polished[:600].rsplit('\n', 1)[0] + "."
    
    return polished

def retrieve_contexts(query: str, top_k: int = 3, threshold: float = 1.5):
    """Retrieve contexts with STRICT similarity threshold.
    Lower l2 distance = higher similarity.
    Threshold of 1.5 = only very relevant matches (strict mode).
    If distance > 1.5, query is considered outside knowledge base."""
    if index is None or not metadata:
        return []

    q_emb = get_query_embedding(query)
    q_emb_matrix = np.array([q_emb]).astype("float32")

    # D is distances (L2), I is indices
    distances, indices = index.search(q_emb_matrix, top_k)
    
    contexts = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx != -1 and dist < threshold:
            ctx = metadata[idx].copy()
            ctx["distance"] = float(dist)
            contexts.append(ctx)
            print(f"Found context: distance={dist:.4f}, source={metadata[idx]['source'][:50]}")
    
    if not contexts:
        print(f"No contexts found. Best match distance: {distances[0][0]:.4f} (threshold: {threshold})")
    
    return contexts

async def generate_chat_stream(query: str, chat_history: list):
    """
    RAG Logic with Strict Anti-Hallucination.
    Only answers from knowledge base; responds 'I don't know' if no good match found.
    """
    contexts = retrieve_contexts(query)
    
    if not contexts:
        print(f"Query '{query}' has no matching contexts")
        yield "I don't know."
        return
    
    # Build prompt
    context_text = "\n\n".join([f"Source: {ctx['source']}\nContent: {ctx['text']}" for ctx in contexts])
    
    system_prompt = f"""You are a concise customer support agent for Amenify.
Answer the user's query in EXACTLY 5 lines or less using ONLY the provided context.
Be direct and informative. Do NOT explain or apologize.
If you cannot answer from the context, respond with: "I don't know."

<context>
{context_text}
</context>

User Query: {query}

Remember: Answer in maximum 5 lines. Be concise and relevant only."""
    
    try:
        model = genai.GenerativeModel(LLM_MODEL)
        response = model.generate_content(
            system_prompt,
            stream=True,
            generation_config=genai.types.GenerationConfig(temperature=0.0)
        )
        
        sources = set([c['source'] for c in contexts])
        source_attribution = f"\n\n*(Sources: {', '.join(sources)})*"
        
        full_response = ""
        for chunk in response:
            if chunk.text:
                full_response += chunk.text
                yield chunk.text
        
        if full_response:
            yield source_attribution
    except Exception as e:
        print(f"Error generating response: {e}")
        # Fallback: Use polished response from context
        if contexts:
            polished = polish_response(contexts, query)
            yield polished
            
            source_list = sorted(set([c['source'] for c in contexts]))
            yield f"\n\n*(Sources: {', '.join(source_list)})*"
        else:
            yield "I don't know."
