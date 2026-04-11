import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from retriever.rag import generate_chat_stream

load_dotenv()

app = FastAPI(title="Amenify AI Support")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

chat_sessions: dict[str, list] = {}


class ChatRequest(BaseModel):
    session_id: str
    message: str


@app.get("/health")
async def health():
    """Lightweight probe for Render/orchestrators; does not load ML resources."""
    return {"status": "ok"}


@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    if req.session_id not in chat_sessions:
        chat_sessions[req.session_id] = []

    user_message = req.message

    async def stream_wrapper():
        full_response = ""
        rag_stream = generate_chat_stream(user_message, chat_sessions[req.session_id])

        async for chunk in rag_stream:
            full_response += chunk
            yield chunk

        chat_sessions[req.session_id].append({"role": "user", "content": user_message})
        chat_sessions[req.session_id].append({"role": "assistant", "content": full_response})

    return StreamingResponse(stream_wrapper(), media_type="text/plain")


_frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.isdir(_frontend_dir):
    app.mount("/", StaticFiles(directory=_frontend_dir, html=True), name="frontend")
