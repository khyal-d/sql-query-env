"""
FastAPI application for the SQL Query Writing Environment.

Standard OpenEnv endpoints (created by create_fastapi_app):
  POST /reset          — start new episode
  POST /step           — submit a SQL query
  GET  /state          — episode metadata
  GET  /health         — liveness probe
  GET  /web            — browser UI
  GET  /docs           — Swagger UI
  WS   /ws             — WebSocket (primary interface)

Additional required endpoints:
  GET  /tasks          — list all tasks + action schema
  POST /grader         — score a single query against a task (stateless)
  POST /baseline       — run OpenAI model against all 3 tasks, return scores
"""

import os
import json
import logging
from typing import Optional

from fastapi import HTTPException
from pydantic import BaseModel

from openenv.core.env_server import create_fastapi_app
from models import SQLAction, SQLObservation
from environment import SQLEnvironment
from tasks import TASKS, create_database, execute_query, compute_score, SCHEMA_DESCRIPTION

logger = logging.getLogger(__name__)

# ── Core OpenEnv app ──────────────────────────────────────────────────────────
app = create_fastapi_app(SQLEnvironment, SQLAction, SQLObservation)


# ── /tasks ───────────────────────────────────────────────────────────────────

@app.get("/tasks")
async def list_tasks():
    """Return all task definitions and the action schema."""
    return {
        "tasks": [
            {
                "task_id":      t.task_id,
                "difficulty":   t.difficulty,
                "question":     t.question,
                "max_attempts": t.max_attempts,
                "action_schema": SQLAction.model_json_schema(),
            }
            for t in TASKS.values()
        ]
    }


# ── /grader ──────────────────────────────────────────────────────────────────

class GraderRequest(BaseModel):
    task_id: int = 1
    query: str


@app.post("/grader")
async def grader(req: GraderRequest):
    """
    Stateless grader: execute *query* against task *task_id* and return the score.
    Creates a fresh database for every call — fully deterministic.
    """
    if req.task_id not in TASKS:
        raise HTTPException(status_code=400, detail=f"task_id must be 1, 2, or 3.")

    task = TASKS[req.task_id]
    conn = create_database()

    expected_rows, err = execute_query(conn, task.expected_sql)
    if err:
        raise HTTPException(status_code=500, detail=f"Internal error: {err}")

    actual_rows, query_err = execute_query(conn, req.query)
    score = compute_score(actual_rows, expected_rows) if not query_err else 0.0

    return {
        "task_id":     req.task_id,
        "difficulty":  task.difficulty,
        "score":       score,
        "rows_returned":  len(actual_rows),
        "rows_expected":  len(expected_rows),
        "error":       query_err,
        "result_preview": actual_rows[:5] if actual_rows else [],
    }


# ── /baseline ────────────────────────────────────────────────────────────────

@app.post("/baseline")
async def run_baseline(model: str = "gpt-4o-mini"):
    """
    Run an LLM baseline against all 3 tasks using the OpenAI API.
    Requires OPENAI_API_KEY environment variable.

    Query param:
        model — OpenAI model to use (default: gpt-4o-mini)

    Returns per-task scores and an aggregate average.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY environment variable is not set.",
        )

    try:
        from openai import OpenAI
    except ImportError:
        raise HTTPException(status_code=503, detail="openai package not installed.")

    client = OpenAI(api_key=api_key)
    results = {}

    for task_id, task in TASKS.items():
        env = SQLEnvironment()
        obs = env.reset(task_id=task_id)
        best_score = 0.0
        best_query = ""

        for attempt in range(task.max_attempts):
            # Build prompt — include feedback from previous attempt if available
            previous_feedback = ""
            if attempt > 0:
                previous_feedback = f"\n\nPrevious attempt feedback: {obs.feedback}"

            prompt = (
                f"You are an expert SQL writer. Write a single SQLite-compatible SQL query "
                f"that answers the question below.\n\n"
                f"{SCHEMA_DESCRIPTION}\n\n"
                f"Question: {task.question}\n\n"
                f"Write ONLY the SQL query with no explanation or markdown."
                f"{previous_feedback}"
            )

            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=400,
                    temperature=0.0,
                )
                raw = response.choices[0].message.content.strip()
            except Exception as exc:
                results[f"task_{task_id}"] = {
                    "task_id":    task_id,
                    "difficulty": task.difficulty,
                    "score":      0.0,
                    "error":      str(exc),
                }
                break

            # Strip markdown fences if the model included them
            query = _strip_markdown(raw)

            obs = env.step(SQLAction(query=query))
            score = obs.reward or 0.0
            if score > best_score:
                best_score = score
                best_query = query

            if obs.done:
                break

        results[f"task_{task_id}"] = {
            "task_id":    task_id,
            "difficulty": task.difficulty,
            "score":      best_score,
            "best_query": best_query,
        }

    avg_score = sum(r["score"] for r in results.values()) / len(results)

    return {
        "model":         model,
        "results":       results,
        "average_score": round(avg_score, 4),
    }


# ── helpers ──────────────────────────────────────────────────────────────────

def _strip_markdown(text: str) -> str:
    """Remove ```sql ... ``` or ``` ... ``` fences if present."""
    text = text.strip()
    if "```sql" in text:
        text = text.split("```sql", 1)[1].split("```", 1)[0]
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0]
    return text.strip()


def main() -> None:
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
