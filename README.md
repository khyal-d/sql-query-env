---
title: SQL Query Writing Environment
sdk: docker
tags:
  - openenv
  - sql
  - text-to-sql
  - reinforcement-learning
---

# SQL Query Writing Environment

An OpenEnv RL environment where an AI agent learns to write SQL queries from natural-language questions.

---

## Motivation

Text-to-SQL is one of the most practically demanded AI capabilities — every data team needs it. Existing benchmarks (Spider, BIRD) test SQL accuracy in a static one-shot setting. This environment makes it interactive and trainable:

- The agent **observes** the database schema and a natural-language question
- The agent **acts** by submitting a SQL query
- The environment **executes** the query on a real SQLite database and returns a graded reward
- The agent can **iterate** across multiple attempts, seeing feedback each time

This enables RL training loops where a model learns not just to answer SQL questions once, but to self-correct based on execution feedback — a capability that mirrors how real data engineers work.

---

## Environment Description

### Database

An e-commerce SQLite database with four tables:

| Table | Rows | Description |
|-------|------|-------------|
| `customers` | 15 | Name, city, country, signup date |
| `products` | 12 | Name, category (`Electronics / Clothing / Books / Home`), price |
| `orders` | 21 | Customer, date, status (`completed / pending / cancelled`) |
| `order_items` | 36 | Order line items with quantity and unit price |

### Action Space

```python
class SQLAction(Action):
    query: str   # A SQLite-compatible SQL query string
```

### Observation Space

```python
class SQLObservation(Observation):
    done: bool
    reward: Optional[float]          # Jaccard score 0.0 – 1.0
    task_id: int                     # 1, 2, or 3
    difficulty: str                  # "easy" | "medium" | "hard"
    question: str                    # Natural-language question
    schema_description: str          # Full schema + sample rows
    query_result: Optional[List[dict]]  # First 10 rows of the agent's result
    columns: Optional[List[str]]     # Column names of the result
    error_message: Optional[str]     # SQL execution error (if any)
    attempts_remaining: int
    feedback: str                    # Human-readable scoring feedback
```

### State

```python
class SQLState(State):
    episode_id: Optional[str]
    step_count: int
    task_id: int
    difficulty: str
    max_attempts: int                # 5 per episode
    best_score: float                # Best score achieved this episode
```

---

## Tasks

### Task 1 — Easy

**Question:** List the name and city of every customer from France, ordered alphabetically by name.

**Why easy:** Single table, simple `WHERE` + `ORDER BY`. No joins required.

**Expected difficulty for frontier models:** 0.95–1.0

---

### Task 2 — Medium

**Question:** What is the total revenue for each product category from completed orders? Return the category and total_revenue (sum of quantity × unit_price), ordered by total_revenue descending.

**Why medium:** Requires 3-table JOIN (`order_items` → `products`, `orders`), `WHERE status = 'completed'`, `GROUP BY`, `SUM`, `ORDER BY`. Must filter out pending orders.

**Expected difficulty for frontier models:** 0.7–0.9

---

### Task 3 — Hard

**Question:** Find customers who placed at least one completed order in January 2024 AND at least one completed order in February 2024. For each such customer return their name and total_spending (sum of quantity × unit_price across both months), ordered by total_spending descending.

**Why hard:** Requires identifying the intersection of customers across two month windows. Clean solutions use `INTERSECT` or a correlated subquery. Must join 3 tables and aggregate correctly. Agents often return the wrong set (e.g., all Jan+Feb customers without enforcing the "both months" constraint).

**Expected difficulty for frontier models:** 0.4–0.7

---

## Reward Function

Each `step()` returns a **Jaccard similarity score** between the agent's result set and the expected result set:

```
score = |actual ∩ expected| / |actual ∪ expected|
```

Row comparison is value-based (column order and row order are ignored). Float values are rounded to 2 decimal places before comparison.

| Score | Meaning |
|-------|---------|
| 1.0 | Exact match — perfect answer |
| 0.75–0.99 | Very close — minor row-level error |
| 0.40–0.74 | Partial — wrong filter or missing join |
| 0.01–0.39 | Weak — some values match by coincidence |
| 0.0 | No match or SQL error |

The agent receives this signal **on every step**, enabling RL algorithms to learn from intermediate feedback rather than sparse end-of-episode signals. Episode ends at `score == 1.0` (success) or `attempts_remaining == 0`.

---

## API

### Standard OpenEnv endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/reset` | Start new episode. Body: `{"task_id": 1}` |
| `POST` | `/step` | Submit query. Body: `{"query": "SELECT ..."}` |
| `GET` | `/state` | Current episode metadata |
| `GET` | `/health` | Liveness probe |
| `WS` | `/ws` | WebSocket (primary training interface) |

### Additional endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/tasks` | All task definitions + action schema |
| `POST` | `/grader` | Stateless grader — score any query against a task |
| `POST` | `/baseline` | Run OpenAI model against all 3 tasks |

#### `POST /grader`

```json
{ "task_id": 2, "query": "SELECT p.category, SUM(oi.quantity * oi.unit_price) AS total_revenue FROM order_items oi JOIN products p ON oi.product_id = p.id JOIN orders o ON oi.order_id = o.id WHERE o.status = 'completed' GROUP BY p.category ORDER BY total_revenue DESC" }
```

Returns: `{ "score": 1.0, "rows_returned": 4, "rows_expected": 4, ... }`

#### `GET /tasks`

Returns task list with `action_schema` (JSON Schema for `SQLAction`).

#### `POST /baseline?model=gpt-4o-mini`

Requires `OPENAI_API_KEY` env var. Returns per-task scores and average.

---

## Baseline Scores

Run with `inference.py` using GPT-4o-mini (zero-shot, up to 5 attempts with feedback):

| Task | Difficulty | Score |
|------|-----------|-------|
| 1 | easy | ~1.00 |
| 2 | medium | ~0.85 |
| 3 | hard | ~0.55 |
| **Average** | | **~0.80** |

*Scores vary slightly due to model sampling. Re-run with `temperature=0.0` for reproducibility.*

---

## Setup & Usage

### Run locally

```bash
git clone https://huggingface.co/spaces/khyaal-d/sql-query-env
cd sql-query-env

pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Open http://localhost:8000/web for the browser UI or http://localhost:8000/docs for Swagger.

### Run with Docker

```bash
docker build -t sql-env .
docker run -p 8000:8000 -e OPENAI_API_KEY=sk-... sql-env
```

### Run inference (baseline)

```bash
# Required env vars
API_BASE_URL=https://api.openai.com/v1 \
MODEL_NAME=gpt-4o-mini \
HF_TOKEN=sk-... \
python inference.py

# Against deployed HF Space (set SERVER_URL to point at the environment server)
API_BASE_URL=https://api.openai.com/v1 \
MODEL_NAME=gpt-4o-mini \
HF_TOKEN=sk-... \
SERVER_URL=https://your-space.hf.space \
python inference.py
```

### Use as a training environment

```python
from client import SQLEnv, SQLAction

with SQLEnv(base_url="https://your-space.hf.space").sync() as env:
    # Task 1: easy
    result = env.reset(task_id=1)
    print(result.observation.question)

    result = env.step(SQLAction(query="SELECT name, city FROM customers WHERE country = 'France' ORDER BY name"))
    print(f"Score: {result.reward}")   # 1.0 if correct
    print(result.observation.feedback)
```

### Validate spec compliance

```bash
openenv validate
```

---

## Project Structure

```
sql-env/
├── models.py        # Pydantic Action / Observation / State types
├── tasks.py         # DB schema, seed data, task definitions, grader logic
├── environment.py   # SQLEnvironment class (reset / step / state)
├── app.py           # FastAPI app + /tasks /grader /baseline endpoints
├── client.py        # EnvClient subclass for typed Python access
├── inference.py     # Baseline inference script (required by spec)
├── baseline.py      # Alias / legacy entry point
├── Dockerfile       # Container definition
├── requirements.txt
├── openenv.yaml     # Environment manifest
└── README.md
```
