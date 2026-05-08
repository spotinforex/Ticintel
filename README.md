# TICINTEL

**Truth In Coverage Intelligence**

TICINTEL is an investigative search tool that goes beyond summarizing what sources say — it maps where they disagree. Given a topic, it retrieves full article text from multiple outlets, extracts structured claims per source, detects genuine contradictions across them, and produces an investigative brief with attribution, conflict analysis, and actionable next steps.

---

## What Makes It Different

Normal AI search tools resolve sources into one blended answer. TICINTEL surfaces conflict. Every claim is pinned to the outlet or person who made it. Where sources contradict each other — on figures, events, or policy outcomes — the tool flags the conflict, explains why it matters, and tells an investigator what to do next.

---

## Pipeline

```
User inputs topic
      ↓
Search Agent        — finds real article URLs via web search
      ↓
URL Retriever       — fetches and cleans full article text (no LLM)
      ↓
Extraction Agent    — pulls structured claims per article, with attribution
      ↓
Contradiction Agent — cross-references all claims, detects conflicts
      ↓
Synthesis Agent     — writes the investigative brief
```

Each stage runs sequentially and streams progress to the frontend via SSE.

---

## Tech Stack

| Layer | Tech |
|---|---|
| Backend | Python 3.11+, FastAPI |
| AI Orchestration | Backboard API |
| Search + Mini tasks | GPT-4.1 mini |
| Reasoning tasks | GPT-4.1 |
| HTTP / Scraping | httpx, BeautifulSoup4 |
| Package manager | uv |
| Hosting | Render |

---

## Assistants

Four assistants are created in the Backboard dashboard, each with its own system prompt:

| Assistant | Model | Role |
|---|---|---|
| Search Agent | GPT-4.1 mini | Finds real article URLs via web search |
| Extraction Agent | GPT-4.1 mini | Extracts structured claims per article |
| Contradiction Agent | GPT-4.1 | Detects conflicts across all claims |
| Synthesis Agent | GPT-4.1 | Writes the investigative brief |

Mini is used for search and extraction — mechanical tasks with clear input/output contracts. GPT-4.1 is used for contradiction detection and synthesis — tasks requiring multi-document reasoning and writing quality.

---

## Model Routing

| Task | Model | Reason |
|---|---|---|
| Search | GPT-4.1 mini | Returns structured JSON, no deep reasoning needed |
| Extraction | GPT-4.1 mini | Structured extraction, straightforward per-article task |
| Contradiction | GPT-4.1 | Holds 40-60 claims in context, requires precise reasoning |
| Synthesis | GPT-4.1 | Output quality matters, leads the user experience |

---

## Built With

- [Backboard](https://backboard.io) — multi-agent orchestration, routing, memory, web search
---

## Hackathon

Built for **Backboard Challenges — Spring 2026**.
