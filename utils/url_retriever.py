import asyncio
import json
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Optional
import logging 

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

import httpx
from bs4 import BeautifulSoup

FETCH_TIMEOUT = 12.0
MIN_TEXT_LENGTH = 400
MAX_TEXT_LENGTH = 6000
REQUEST_DELAY = 0.6
MAX_RETRIES = 2

JUNK_TAGS = [
    "nav", "footer", "header", "script", "style",
    "aside", "form", "button", "iframe", "noscript",
    "figure", "figcaption", "advertisement"
]

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}

SKIP_URL_PATTERNS = [
    r"accounts\.google\.com",
    r"login\.",
    r"signin\.",
    r"subscribe\.",
    r"\.pdf$",
    r"youtube\.com",
    r"twitter\.com",
    r"x\.com",
    r"facebook\.com",
    r"instagram\.com",
    r"/liveblog/",       
    r"/live-news/",    
    r"nytimes\.com"    # paywall restricted 
]

@dataclass
class Article:
    title: str
    url: str
    source: str
    date: str
    full_text: str
    char_count: int = 0
    fetch_status: str = "ok"
    fetch_error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def viable(self) -> bool:
        return self.fetch_status == "ok" and self.char_count >= MIN_TEXT_LENGTH


def should_skip_url(url: str) -> bool:
    for pattern in SKIP_URL_PATTERNS:
        if re.search(pattern, url, re.IGNORECASE):
            return True
    return False


def is_likely_paywall(html: str) -> bool:
    signals = [
        "subscribe to continue",
        "sign in to read",
        "create a free account",
        "this content is for subscribers",
        "paywall",
        "premium content",
    ]
    lower = html.lower()
    return any(s in lower for s in signals)


def extract_text(html: str) -> str:
    try:
        soup = BeautifulSoup(html, "html.parser")

        for tag in soup(JUNK_TAGS):
            tag.decompose()

        article_tag = soup.find("article")
        if article_tag:
            paragraphs = article_tag.find_all("p")
        else:
            paragraphs = soup.find_all("p")

        text = " ".join(p.get_text(separator=" ") for p in paragraphs)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:MAX_TEXT_LENGTH]

    except Exception:
        return ""


async def fetch_url(client: httpx.AsyncClient, url: str) -> tuple[str, str]:
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = await client.get(
                url,
                headers=BROWSER_HEADERS,
                timeout=FETCH_TIMEOUT,
                follow_redirects=True,
            )
            response.raise_for_status()
            response.encoding = response.charset_encoding or "utf-8"
            return response.text, ""

        except httpx.TimeoutException:
            if attempt < MAX_RETRIES:
                logger.warning("Timeout on attempt %d/%d for %s — retrying", attempt + 1, MAX_RETRIES + 1, url[:60])
                await asyncio.sleep(1.0)
                continue
            return "", "timeout"

        except httpx.HTTPStatusError as e:
            return "", f"http_{e.response.status_code}"

        except httpx.RequestError as e:
            return "", f"request_error: {str(e)[:60]}"

    return "", "max_retries_exceeded"


async def process_article(
    client: httpx.AsyncClient,
    raw: dict,
    index: int,
    total: int,
) -> Article:
    url = raw.get("url", "").strip()
    title = raw.get("title", "Untitled")
    source = raw.get("source", "Unknown")
    date = raw.get("date", "")

    logger.info("[%d/%d] %s — %s", index + 1, total, source, url[:60])

    if should_skip_url(url):
        logger.warning("[%d/%d] Skipped — matched bad URL pattern: %s", index + 1, total, url[:60])
        return Article(
            title=title, url=url, source=source, date=date,
            full_text="", fetch_status="skipped",
            fetch_error="matched skip pattern"
        )

    html, error = await fetch_url(client, url)

    if error:
        logger.error("[%d/%d] Fetch failed (%s): %s", index + 1, total, error, url[:60])
        return Article(
            title=title, url=url, source=source, date=date,
            full_text="", fetch_status="failed", fetch_error=error
        )

    if is_likely_paywall(html):
        logger.warning("[%d/%d] Paywall detected: %s", index + 1, total, url[:60])
        return Article(
            title=title, url=url, source=source, date=date,
            full_text="", fetch_status="failed", fetch_error="paywall"
        )

    text = extract_text(html)

    if len(text) < MIN_TEXT_LENGTH:
        logger.warning("[%d/%d] Text too short (%d chars): %s", index + 1, total, len(text), url[:60])
        return Article(
            title=title, url=url, source=source, date=date,
            full_text=text, char_count=len(text),
            fetch_status="too_short"
        )

    logger.info("[%d/%d] OK — %d chars extracted from %s", index + 1, total, len(text), source)
    return Article(
        title=title, url=url, source=source, date=date,
        full_text=text, char_count=len(text),
        fetch_status="ok"
    )


async def retrieve_articles(llm_output: list[dict]) -> dict:
    """
    Main entry point.

    Args:
        llm_output: list of dicts from search agent, each with:
                    {title, url, source, date}

    Returns:
        {
            "viable": [Article, ...],      # ready for extraction agent
            "failed": [Article, ...],      # fetch failed or too short
            "skipped": [Article, ...],     # bad URL patterns
            "stats": { total, viable, failed, skipped, success_rate }
        }
    """
    total = len(llm_output)
    logger.info("Starting retrieval for %d articles", total)

    results: list[Article] = []

    async with httpx.AsyncClient() as client:
        for i, raw in enumerate(llm_output):
            article = await process_article(client, raw, i, total)
            results.append(article)

            if i < total - 1:
                await asyncio.sleep(REQUEST_DELAY)

    viable = [a for a in results if a.viable]
    failed = [a for a in results if a.fetch_status == "failed"]
    too_short = [a for a in results if a.fetch_status == "too_short"]
    skipped = [a for a in results if a.fetch_status == "skipped"]

    success_rate = round(len(viable) / total * 100) if total > 0 else 0

    logger.info(
        "Retrieval complete — viable: %d/%d | failed: %d | too_short: %d | skipped: %d | success_rate: %d%%",
        len(viable), total, len(failed), len(too_short), len(skipped), success_rate
    )

    return {
        "viable": viable,
        "failed": failed + too_short,
        "skipped": skipped,
        "stats": {
            "total": total,
            "viable": len(viable),
            "failed": len(failed),
            "too_short": len(too_short),
            "skipped": len(skipped),
            "success_rate": success_rate,
        }
    }


def parse_llm_search_output(raw_response: str) -> list[dict]:
    """
    Safely parses JSON array from LLM search agent response.
    Handles cases where LLM wraps output in markdown code fences.
    """
    cleaned = re.sub(r"```(?:json)?", "", raw_response).strip()

    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return data
        raise ValueError("Expected a JSON array")
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse LLM output as JSON: {e}\nRaw: {raw_response[:200]}")


async def _test():
    sample_llm_output = [
        {
            "title": "Nigeria inflation rate hits record high",
            "url": "https://punchng.com/breaking-bauchi-gov-loyalists-dump-pdp-for-apm/",
            "source": "Premium Times",
            "date": "2026-05-01"
        },
        {
            "title": "CBN holds interest rate steady",
            "url": "https://punchng.com/gunmen-attack-police-base-in-kwara-kill-three/",
            "source": "Business Day",
            "date": "2026-05-01"
        },
        {
            "title": "Bad URL test",
            "url": "https://youtube.com/watch?v=xyz",
            "source": "YouTube",
            "date": "2026-05-01"
        },
    ]

    results = await retrieve_articles(sample_llm_output)

    logging.info(results)

