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
    queue: asyncio.Queue = asyncio.Queue()

    async def progress(step: str, status: str, data=None):
        await queue.put({"step": step, "status": status, "data": data})

    async def run():
        try:
            # remove the manual "search in_progress" emit here —
            # run_pipeline emits it internally
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
            await queue.put(None)

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
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/followup")
async def followup(request: FollowUpRequest):
    if not request.thread_id:
        raise HTTPException(status_code=400, detail="thread_id is required for follow-up")
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="question cannot be empty")

    logger.info("Follow-up received — thread: %s | question: %s", request.thread_id, request.question[:80])

    topic_with_followup = f"Follow-up question: {request.question}"

    result, response_thread_id = await synthesis_agent(
        topic=topic_with_followup,
        extractions=request.extractions,
        contradiction=request.contradiction,
        thread_id=None
    )

    if result is None:
        raise HTTPException(status_code=500, detail="Synthesis agent failed to respond")

    # Build a readable markdown answer from the structured brief
    parts = []
    if result.get("headline"):
        parts.append(f"## {result['headline']}")
    if result.get("situation_summary"):
        parts.append(result["situation_summary"])
    if result.get("key_conflicts"):
        parts.append("**Key Conflicts**")
        for c in result["key_conflicts"]:
            text = c if isinstance(c, str) else c.get("summary") or c.get("description") or str(c)
            parts.append(f"- {text}")
    if result.get("no_conflicts_note"):
        parts.append(result["no_conflicts_note"])
    if result.get("consensus"):
        parts.append("**Consensus**")
        for c in result["consensus"]:
            text = c if isinstance(c, str) else c.get("claim") or str(c)
            parts.append(f"- {text}")
    if result.get("open_questions"):
        parts.append("**Open Questions**")
        for q in result["open_questions"]:
            text = q if isinstance(q, str) else q.get("question") or str(q)
            parts.append(f"- {text}")

    answer_text = "\n\n".join(parts) if parts else "No answer could be generated."

    return {
        "status": "ok",
        "thread_id": response_thread_id,
        "answer": answer_text
    }
