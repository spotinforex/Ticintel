import asyncio
import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from logic.pipeline import run_pipeline
from agent.ai import synthesis_agent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("TICINTEL API starting...")
    yield
    logger.info("TICINTEL API shutting down...")

app = FastAPI(
    title="TICINTEL API",
    description="Investigative intelligence pipeline",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InvestigateRequest(BaseModel):
    topic: str
    mode: str = "quick"  # quick | deep


class FollowUpRequest(BaseModel):
    question: str
    thread_id: str
    extractions: list[dict]
    contradiction: dict

@app.get("/health")
async def health():
    return {"status": "ok", "service": "ticintel"}

@app.post("/investigate")
async def investigate(request: InvestigateRequest):
    """
    Runs full pipeline — search → retrieve → extract → contradict → synthesize.
    Streams progress events via SSE so the frontend can show live steps.

    Event shape:
    { "step": str, "status": "in_progress" | "done" | "error", "data": any }

    Final event:
    { "step": "complete", "status": "done", "data": { full result } }
    """
    queue: asyncio.Queue = asyncio.Queue()

    async def progress(step: str, status: str, data=None):
        await queue.put({"step": step, "status": status, "data": data})

    async def run():
        try:
            # emit each stage as it starts and completes
            await progress("search", "in_progress")
            result = await run_pipeline(
                topic=request.topic,
                mode=request.mode,
                progress_callback=progress
            )
            await progress("complete", "done", result)
        except Exception as e:
            logger.error("Pipeline error: %s", e)
            await progress("error", "error", {"reason": str(e)})
        finally:
            await queue.put(None)  # sentinel

    async def event_stream():
        asyncio.create_task(run())
        while True:
            event = await queue.get()
            if event is None:
                break
            yield f"data: {json.dumps(event, default=str)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no", 
        }
    )


@app.post("/followup")
async def followup(request: FollowUpRequest):
    """
    Handles follow-up questions on an existing investigation.
    Passes thread_id so the agent recalls full context.
    Does NOT re-run the pipeline — just calls synthesis agent directly.
    """
    if not request.thread_id:
        raise HTTPException(status_code=400, detail="thread_id is required for follow-up")

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="question cannot be empty")

    logger.info("Follow-up received — thread: %s | question: %s", request.thread_id, request.question[:80])

    # inject follow-up question into the topic for synthesis context
    topic_with_followup = f"Follow-up question: {request.question}"

    result = await synthesis_agent(
        topic=topic_with_followup,
        extractions=request.extractions,
        contradiction=request.contradiction,
        thread_id=request.thread_id
    )

    if result is None:
        raise HTTPException(status_code=500, detail="Synthesis agent failed to respond")

    return {
        "status": "ok",
        "thread_id": request.thread_id,
        "answer": result
    }
