from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from retriever.rag import load_vector_db, generate_chat_stream
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Amenify AI Support")

# Enable CORS for local dev if frontend served separately
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-Memory chat history
# Format: { "session_id": [{"role": "user", "content": "..."}] }
chat_sessions = {}

class ChatRequest(BaseModel):
    session_id: str
    message: str

@app.on_event("startup")
async def startup_event():
    load_vector_db()

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    if req.session_id not in chat_sessions:
        chat_sessions[req.session_id] = []
        
    user_message = req.message
    
    # Define an async generator to wrap RAG stream and update history
    async def stream_wrapper():
        full_response = ""
        # stream generator from RAG
        rag_stream = generate_chat_stream(user_message, chat_sessions[req.session_id])
        
        async for chunk in rag_stream:
            full_response += chunk
            yield chunk
            
        # Update history once stream is done
        chat_sessions[req.session_id].append({"role": "user", "content": user_message})
        chat_sessions[req.session_id].append({"role": "assistant", "content": full_response})

    return StreamingResponse(stream_wrapper(), media_type="text/plain")

app.mount("/", StaticFiles(directory=os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend"), html=True), name="frontend")
