# Amenify AI Support Chatbot

An AI-powered customer support chatbot designed specifically for Amenify. It uses Retrieval-Augmented Generation (RAG) to ensure responses are strictly grounded in an ingested knowledge base, with a built-in strict anti-hallucination logic.

## System Architecture

- **Backend:** Python + FastAPI
- **LLM:** Google Gemini API (`gemini-1.5-flash`)
- **Embeddings:** Local (sentence-transformers, `all-MiniLM-L6-v2`)
- **Vector DB:** FAISS
- **Frontend:** HTML, CSS, Vanilla JS

## Local Setup

### 1. Install Dependencies

Make sure you have Python 3.9+ installed.

```bash
pip install -r requirements.txt
```

### 2. Configure Environment variables

Create a `.env` file in the root directory and add your Google Gemini API key:

```env
GOOGLE_GEMINI_API_KEY="your-google-gemini-api-key"
```

**Note:** Embeddings are generated locally using sentence-transformers, so no API key is needed for embeddings.

### 3. Data Ingestion

Run the ingestion script to scrape the provided Amenify URLs, create chunks, generate embeddings, and store them in the local `data/` directory.

```bash
python -m ingestion.ingest
```

### 4. Run the API Server

Start the FastAPI backend with Uvicorn. The server will mount the frontend files and serve them securely.

```bash
uvicorn api.main:app --reload
```

Then visit: [http://localhost:8000](http://localhost:8000)

## Deployment Steps

This application can be easily deployed to a cloud provider like Render, Railway, or AWS.

### Option 1: Render / Railway (Recommended)

1. **Prepare your repo**: Push this entire directory to a GitHub repository. Note that the `data/` folder contains your FAISS indices; while for small indices it can be checked into git, best practice is to rebuild it in the build step, or store it in S3. For simplicity, commit it if it's small, or add a build script that runs `python -m ingestion.ingest` before starting.
2. **Setup on Render/Railway**: Create a new Web Service using the GitHub repository.
3. **Environment setup**: Set `GOOGLE_GEMINI_API_KEY` in the environment variables menu.
4. **Start Command**: Set the start command to: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
5. **Deploy**: Render/Railway will install the dependencies from `requirements.txt` and start the server.

### Option 2: AWS Elastic Beanstalk (EB)

1. Initialize the EB app in standard Python runtime:
   ```bash
   eb init -p python-3.9 amenify-bot
   eb create amenify-bot-env
   ```
2. Make sure you set your env variables for EB:
   ```bash
   eb setenv GOOGLE_GEMINI_API_KEY=your-api-key
   ```
3. EB automatically runs Uvicorn on port 8000 or you can add a `Procfile`:
   `web: uvicorn api.main:app --host 0.0.0.0 --port 8000`

## Features Included

- **Modular Design**: Separated components into `/api`, `/ingestion`, `/retriever`, `/frontend`.
- **Streaming Response**: Real-time streaming chunks using SSE + Fetch streams.
- **Typing Indicator**: Clean frontend UX element during LLM computation.
- **Source Attribution**: Citations rendered explicitly upon streaming completion.
- **Session Memory**: In-memory sessions maintained across client disconnections mapped via standard UUIDs in LocalStorage.
- **Anti-Hallucination**: High similarity thresholds enforced, dropping to standard "I don't know" logic if no viable context is fetched.
