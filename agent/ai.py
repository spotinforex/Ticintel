import json
import re
import logging
import httpx
from backboard import BackboardClient
from dotenv import load_dotenv
from utils.retry import retry
import os

load_dotenv()

logger = logging.getLogger(__name__)

api_key = os.getenv("api_key")
search_assistant_id = os.getenv("search_assistant_id")
extract_assistant_id = os.getenv("extract_assistant_id")
contradict_assistant_id = os.getenv("contradict_assistant_id")
synthesis_assistant_id = os.getenv("synthesis_assistant_id")

client = BackboardClient(api_key=api_key, timeout=120)


def _build_message_kwargs(**kwargs) -> dict:
    """Strip None values so optional params aren't sent."""
    return {k: v for k, v in kwargs.items() if v is not None}


@retry(max_attempts=3, delay=0.5, backoff=0.5, exceptions=(httpx.HTTPError, httpx.ConnectError))
async def search_agent(topic: str, mode: str = "quick", thread_id: str | None = None) -> tuple[str | None, str | None]:
    """
    Calls search agent.
    Returns (content, thread_id) or (None, None) on failure.
    """
    try:
        prompt = f"Topic: {topic}\nMode: {mode}"
        kwargs = _build_message_kwargs(
            llm_provider="openai",
            model_name="gpt-4.1-mini",
            assistant_id=search_assistant_id,
            web_search="Auto",
            thread_id=thread_id  # passed only if not None
        )
        response = await client.send_message(prompt, **kwargs)
        thread_id = response.messages[0].get("thread_id")
        content = response.messages[0].get("content")
        logger.info("Search agent complete — thread: %s", thread_id)
        return content, thread_id
    except Exception as e:
        logger.error("Search agent error: %s", e)
        return None, None


@retry(max_attempts=3, delay=0.5, backoff=0.5, exceptions=(httpx.HTTPError, httpx.ConnectError))
async def extraction_agent(article, thread_id: str | None = None) -> tuple[dict | None, str | None]:
    """
    Calls extraction agent for a single article.
    Accepts Article dataclass or dict.
    Returns (parsed claims dict, thread_id) or (None, None) on failure.
    """
    try:
        if hasattr(article, "__dict__"):
            data = {
                "title": article.title,
                "url": article.url,
                "source": article.source,
                "date": article.date,
                "full_text": article.full_text
            }
        else:
            data = {
                "title": article.get("title", ""),
                "url": article.get("url", ""),
                "source": article.get("source", ""),
                "date": article.get("date", ""),
                "full_text": article.get("full_text", "")
            }

        kwargs = _build_message_kwargs(
            llm_provider="openai",
            model_name="gpt-4.1-mini",
            assistant_id=extract_assistant_id,
            thread_id=thread_id
        )
        response = await client.send_message(json.dumps(data), **kwargs)
        thread_id = response.messages[0].get("thread_id")
        content = response.messages[0].get("content")
        cleaned = re.sub(r"```(?:json)?", "", content).strip()
        parsed = json.loads(cleaned)

        logger.info(
            "Extracted %d claims from [%s] — quality: %s",
            parsed.get("claim_count", 0),
            data["source"],
            parsed.get("quality", "unknown")
        )
        return parsed

    except json.JSONDecodeError as e:
        logger.error("Failed to parse extraction output: %s", e)
        return None
    except Exception as e:
        logger.error("Extraction agent error: %s", e)
        return None


@retry(max_attempts=3, delay=0.5, backoff=0.5, exceptions=(httpx.HTTPError, httpx.ConnectError))
async def contradiction_agent(extractions: list[dict], thread_id: str | None = None) -> tuple[dict | None, str | None]:
    """
    Calls contradiction agent with all extracted claims.
    Returns (parsed conflict analysis, thread_id) or (None, None) on failure.
    """
    try:
        kwargs = _build_message_kwargs(
            llm_provider="openai",
            model_name="gpt-4.1",
            assistant_id=contradict_assistant_id,
            thread_id=thread_id
        )
        response = await client.send_message(json.dumps(extractions), **kwargs)
        thread_id = response.messages[0].get("thread_id")
        content = response.messages[0].get("content")
        cleaned = re.sub(r"```(?:json)?", "", content).strip()
        parsed = json.loads(cleaned)

        logger.info(
            "Contradiction agent complete — conflicts: %d | has_conflicts: %s",
            parsed.get("conflict_count", 0),
            parsed.get("has_conflicts", False)
        )
        return parsed

    except json.JSONDecodeError as e:
        logger.error("Failed to parse contradiction output: %s", e)
        return None
    except Exception as e:
        logger.error("Contradiction agent error: %s", e)
        return None


@retry(max_attempts=3, delay=0.5, backoff=0.5, exceptions=(httpx.HTTPError, httpx.ConnectError))
async def synthesis_agent(
    topic: str,
    extractions: list[dict],
    contradiction: dict,
    thread_id: str | None = None
) -> tuple[dict | None, str | None]:
    """
    Calls synthesis agent with full pipeline output.
    Returns (parsed investigative brief, thread_id) or (None, None) on failure.
    """
    try:
        prompt = json.dumps({
            "topic": topic,
            "extractions": extractions,
            "contradiction": contradiction
        })

        kwargs = _build_message_kwargs(
            llm_provider="openai",
            model_name="gpt-4.1",
            assistant_id=synthesis_assistant_id,
            thread_id=thread_id
        )
        response = await client.send_message(prompt, **kwargs)
        thread_id = response.messages[0].get("thread_id")
        content = response.messages[0].get("content")
        cleaned = re.sub(r"```(?:json)?", "", content).strip()
        parsed = json.loads(cleaned)

        logger.info("Synthesis agent complete — headline: %s", parsed.get("headline", ""))
        return parsed

    except json.JSONDecodeError as e:
        logger.error("Failed to parse synthesis output: %s", e)
        return None
    except Exception as e:
        logger.error("Synthesis agent error: %s", e)
        return None
