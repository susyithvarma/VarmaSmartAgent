from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException, Header, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
from agent.agent import AIAgent, data_manager

load_dotenv()

app = FastAPI(title="Varma's Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = AIAgent()

async def require_api_key(x_api_key: str = Header(None)):
    if not x_api_key or x_api_key != os.getenv("VARMA_AGENT_API_KEY"):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return x_api_key

async def require_admin_key(x_admin_key: str = Header(None)):
    if not x_admin_key or x_admin_key != os.getenv("VARMA_ADMIN_KEY"):
        raise HTTPException(status_code=401, detail="Invalid or missing admin key")
    return x_admin_key

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket, api_key: str = ""):
    if api_key != os.getenv("VARMA_AGENT_API_KEY"):
        await websocket.close(code=4001)
        return

    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            user_message = data.get("message", "")
            if not user_message:
                await websocket.send_text("[error] message required")
                continue

            async for token in agent.stream_reply(user_message):
                await websocket.send_text(token)
            await websocket.send_text("[DONE]")
    except WebSocketDisconnect:
        return
    except Exception as e:
        try:
            await websocket.send_text(f"[error] {e}")
        except:
            pass
        await websocket.close()

@app.post("/chat")
async def chat_http(body: dict, api_key: str = Depends(require_api_key)):
    message = body.get("message")
    if not message:
        raise HTTPException(status_code=400, detail="message required")
    resp = await agent.handle_message(message)
    return {"response": resp}

@app.post("/ingest-url")
async def ingest_url(body: dict, background: BackgroundTasks, admin_key: str = Depends(require_admin_key)):
    """
    Admin endpoint to ingest a URL into the vector store.
    Body: {"url": "...", "name": "optional name"}
    Runs ingestion in background and returns immediately with a job id.
    """
    url = body.get("url")
    name = body.get("name")
    if not url:
        raise HTTPException(status_code=400, detail="url required")
    # run ingestion in background
    background.add_task(data_manager.ingest_url, url, name)
    return {"status": "ingestion started", "url": url}

@app.get("/sources")
async def list_sources(admin_key: str = Depends(require_admin_key)):
    return {"sources": data_manager.list_sources(), "count": data_manager.vs.count()}

@app.post("/reindex")
async def reindex_all(admin_key: str = Depends(require_admin_key)):
    """
    Reindex all sources (naive clear and reingest).
    """
    count = data_manager.reindex_all()
    return {"status": "reindex complete", "sources_reindexed": count}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)