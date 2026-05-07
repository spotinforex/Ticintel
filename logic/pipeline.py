import asyncio
import json
import re
import logging
from utils.url_retriever import retrieve_articles
from agent.ai import (
    search_agent,
    extraction_agent,
    contradiction_agent,
    synthesis_agent
)

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)


def parse_search_output(raw: str) -> dict:
    """
    Parses JSON from search agent output.
    Strips markdown fences if present.
    """
    cleaned = re.sub(r"```(?:json)?", "", raw).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse search agent output: {e}\nRaw: {raw[:200]}")


async def run_extraction_pipeline(viable: list) -> dict:
    extractions = []
    failed = []

    logger.info("Starting extraction for %d articles...", len(viable))

    for i, article in enumerate(viable):
        source = article.source if hasattr(article, "source") else article.get("source", "unknown")
        logger.info("Extracting [%d/%d] — %s", i + 1, len(viable), source)

        result, _ = await extraction_agent(article)

        if result is None:
            logger.warning("Extraction failed for [%s] — skipping", source)
            failed.append(source)
            continue

        quality = result.get("quality", "good")
        if quality == "noisy":
            logger.warning(
                "Skipping noisy article [%s] — note: %s",
                source, result.get("quality_note", "")
            )
            failed.append(source)
            continue

        extractions.append(result)

        if i < len(viable) - 1:
            await asyncio.sleep(0.3)

    total_claims = sum(e.get("claim_count", 0) for e in extractions)
    good_count = sum(1 for e in extractions if e.get("quality") == "good")
    thin_count = sum(1 for e in extractions if e.get("quality") == "thin")

    logger.info(
        "Extraction complete — %d/%d articles | %d claims | good: %d | thin: %d | failed: %d",
        len(extractions), len(viable), total_claims, good_count, thin_count, len(failed)
    )

    return {
        "extractions": extractions,
        "failed": failed,
        "total_claims": total_claims,
        "good_count": good_count,
        "thin_count": thin_count,
        "noisy_count": len(failed)
    }


async def run_contradiction_pipeline(extractions: list[dict]) -> dict:
    logger.info("Running contradiction agent on %d extractions...", len(extractions))

    result, _ = await contradiction_agent(extractions)

    if result is None:
        logger.error("Contradiction agent returned no output")
        return {
            "status": "failed",
            "conflict_count": 0,
            "has_conflicts": False,
            "conflicts": [],
            "consensus_claims": [],
            "coverage_gaps": [],
            "topic_summary": "",
            "raw": {}
        }

    has_conflicts = result.get("has_conflicts", False)
    conflict_count = result.get("conflict_count", 0)

    if not has_conflicts:
        logger.info("No conflicts detected across sources")
        return {
            "status": "no_conflicts",
            "conflict_count": 0,
            "has_conflicts": False,
            "conflicts": [],
            "consensus_claims": result.get("consensus_claims", []),
            "coverage_gaps": result.get("coverage_gaps", []),
            "topic_summary": result.get("topic_summary", ""),
            "raw": result
        }

    logger.info(
        "Contradiction pipeline complete — %d conflicts | %d consensus | %d gaps",
        conflict_count,
        len(result.get("consensus_claims", [])),
        len(result.get("coverage_gaps", []))
    )

    return {
        "status": "ok",
        "conflict_count": conflict_count,
        "has_conflicts": True,
        "conflicts": result.get("conflicts", []),
        "consensus_claims": result.get("consensus_claims", []),
        "coverage_gaps": result.get("coverage_gaps", []),
        "topic_summary": result.get("topic_summary", ""),
        "raw": result
    }


async def run_synthesis_pipeline(
    topic: str,
    extractions: list[dict],
    contradiction: dict
) -> dict:
    logger.info("Running synthesis agent...")

    result, _ = await synthesis_agent(topic, extractions, contradiction)

    if result is None:
        logger.error("Synthesis agent returned no output")
        return {
            "status": "failed",
            "brief": None,
            "headline": "",
            "situation_summary": "",
            "key_conflicts": [],
            "no_conflicts_note": "",
            "consensus": [],
            "open_questions": [],
            "sources": [],
            "meta": {}
        }

    logger.info("Synthesis complete — brief ready")

    return {
        "status": "ok",
        "brief": result,
        "headline": result.get("headline", ""),
        "situation_summary": result.get("situation_summary", ""),
        "key_conflicts": result.get("key_conflicts", []),
        "no_conflicts_note": result.get("no_conflicts_note", ""),
        "consensus": result.get("consensus", []),
        "open_questions": result.get("open_questions", []),
        "sources": result.get("sources", []),
        "meta": result.get("meta", {})
    }


async def search_and_retrieve(topic: str, mode: str = "quick") -> dict:
    logger.info("Pipeline starting — topic: '%s', mode: %s", topic, mode)
    logger.info("Step 1/2 — Running search agent...")

    raw_content, thread_id = await search_agent(topic, mode)

    if raw_content is None:
        return {"status": "error", "reason": "Search agent failed"}

    parsed = parse_search_output(raw_content)
    search_mode = parsed.get("mode")

    if search_mode == "clarification":
        logger.warning("Search agent returned clarification — reason: %s", parsed.get("reason"))
        return {
            "status": "clarification",
            "topic": topic,
            "mode": mode,
            "thread_id": thread_id,
            "reason": parsed.get("reason"),
            "searches_attempted": parsed.get("searches_attempted", []),
            "questions": parsed.get("questions", []),
            "partial_results": parsed.get("partial_results", [])
        }

    articles = parsed.get("articles", [])
    logger.info("Step 2/2 — Retrieving full text for %d articles...", len(articles))
    retrieval_results = await retrieve_articles(articles)
    viable = retrieval_results["viable"]

    logger.info(
        "Retrieval complete — viable: %d | failed: %d | skipped: %d",
        retrieval_results["stats"]["viable"],
        retrieval_results["stats"]["failed"],
        retrieval_results["stats"]["skipped"]
    )

    if len(viable) < 3:
        logger.warning("Insufficient viable articles (%d) — forcing clarification", len(viable))
        return {
            "status": "clarification",
            "topic": topic,
            "mode": mode,
            "thread_id": thread_id,
            "reason": f"Only {len(viable)} usable sources found — topic may be too localised or obscure",
            "searches_attempted": [],
            "questions": [
                {
                    "id": "q1",
                    "question": "Can you provide more context or alternative names for this topic?",
                    "why": "More specific details would help find relevant coverage"
                }
            ],
            "partial_results": [a.to_dict() for a in viable]
        }

    return {
        "status": "ok",
        "topic": topic,
        "mode": mode,
        "thread_id": thread_id,
        "viable": viable,
        "failed": retrieval_results["failed"],
        "skipped": retrieval_results["skipped"],
        "stats": retrieval_results["stats"]
    }


async def run_pipeline(
    topic: str,
    mode: str = "quick",
    progress_callback=None
) -> dict:

    async def emit(step: str, status: str, data=None):
        if progress_callback:
            await progress_callback(step, status, data)

    # Stage 1
    await emit("search", "in_progress")
    search_result = await search_and_retrieve(topic, mode)
    if search_result["status"] != "ok":
        return search_result
    await emit("search", "done", {"articles": search_result["stats"]})

    # Stage 2
    await emit("extraction", "in_progress")
    extraction_result = await run_extraction_pipeline(search_result["viable"])
    good_extractions = [
        e for e in extraction_result["extractions"]
        if e.get("quality") in ("good", "thin")
    ]
    if len(good_extractions) < 2:
        return {
            "status": "clarification",
            "topic": topic,
            "mode": mode,
            "thread_id": search_result["thread_id"],
            "reason": "Could not extract enough structured claims",
            "searches_attempted": [],
            "questions": [
                {
                    "id": "q1",
                    "question": "Can you provide additional sources or a more specific angle?",
                    "why": "Available articles did not contain enough extractable facts"
                }
            ],
            "partial_results": []
        }
    await emit("extraction", "done", {"total_claims": extraction_result["total_claims"]})

    # Stage 3
    await emit("contradiction", "in_progress")
    contradiction_result = await run_contradiction_pipeline(good_extractions)
    await emit("contradiction", "done", {
        "conflict_count": contradiction_result["conflict_count"]
    })

    # Stage 4
    await emit("synthesis", "in_progress")
    synthesis_result = await run_synthesis_pipeline(
        topic=topic,
        extractions=good_extractions,
        contradiction=contradiction_result
    )
    await emit("synthesis", "done")

    with open("output.json", "w") as f:
        f.write(json.dumps({
            "topic": topic,
            "brief": synthesis_result.get("brief"),
            "extractions": extraction_result["extractions"],
            "contradiction": contradiction_result
        }, indent=2))

    return {
        "status": "ok",
        "topic": topic,
        "mode": mode,
        "thread_id": search_result["thread_id"],
        "brief": synthesis_result.get("brief"),
        "extractions": extraction_result["extractions"],
        "contradiction": contradiction_result,
        "extraction_stats": {
            "total_claims": extraction_result["total_claims"],
            "good_count": extraction_result["good_count"],
            "thin_count": extraction_result["thin_count"],
            "failed": extraction_result["failed"]
        },
        "retrieval_stats": search_result["stats"]
    }
