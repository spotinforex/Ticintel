# TicIntel

TicIntel (Truth In Coverage Intelligence) is an investigative intelligence agent built to explore a topic, gather evidence from multiple sources, identify contradictions, and synthesize a clear investigative brief.

It is designed for situations where coverage differs across outlets, claims need verification, and a concise summary is required quickly.

## What TicIntel does

TicIntel combines a lightweight retrieval and analysis pipeline with AI-powered agents to:

- search for relevant coverage on a topic
- retrieve and evaluate source material
- extract claims from articles
- detect contradictions and consensus points
- generate a synthesized summary with open questions and conflict highlights
- expose the workflow through a simple API for streaming progress and follow-up analysis

## Core workflow

1. Search agent identifies relevant sources for a topic.
2. Retrieval layer collects article content and filters for usable material.
3. Extraction agent turns source text into structured claims.
4. Contradiction agent compares claims across sources.
5. Synthesis agent produces a concise investigative brief.

## Project structure

- main.py — FastAPI application and API endpoints
- logic/pipeline.py — orchestration of the investigation pipeline
- agent/ — agent integrations for search, extraction, contradiction detection, and synthesis
- utils/ — helpers for retries and article retrieval

## Requirements

- Python 3.13+
- Dependencies listed in pyproject.toml

## Installation

```bash
pip install -e .
```

If you prefer to install dependencies manually:

```bash
pip install backboard-sdk beautifulsoup4 fastapi httpx python-dotenv uvicorn
```

## Environment variables

Create a .env file in the project root with the required values:

```env
api_key=your_backboard_api_key
search_assistant_id=your_search_assistant_id
extract_assistant_id=your_extraction_assistant_id
contradict_assistant_id=your_contradiction_assistant_id
synthesis_assistant_id=your_synthesis_assistant_id
```

## Running the application

Start the API server:

```bash
uvicorn main:app --reload
```

The app will be available at:

- http://127.0.0.1:8000/docs for Swagger UI
- http://127.0.0.1:8000/health for a basic health check

## API endpoints

### Health check

```bash
curl http://127.0.0.1:8000/health
```

### Investigate a topic

```bash
curl -X POST http://127.0.0.1:8000/investigate \
  -H "Content-Type: application/json" \
  -d '{"topic":"AI safety regulation","mode":"quick"}'
```

This endpoint streams progress events using Server-Sent Events (SSE) and returns a final synthesized result.

### Follow-up analysis

```bash
curl -X POST http://127.0.0.1:8000/followup \
  -H "Content-Type: application/json" \
  -d '{
    "thread_id":"example-thread",
    "question":"What are the main unresolved questions?",
    "extractions": [],
    "contradiction": {}
  }'
```

## Links

Video Link: [TicIntel Video](https://youtu.be/jRyxCSk0y9w?si=KdRwnKYvAM1B8T-L)
Live App Link: [App](https://ticintel-frontend.vercel.app/)

## Example use case

Use TicIntel when you want to quickly understand whether a claim is supported, disputed, or under-covered across the available media landscape.

Examples include:

- public affairs investigations
- claim verification workflows
- media monitoring and contradiction mapping
- policy and issue analysis

## License

This project is licensed under the Apache License 2.0.
