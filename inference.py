#!/usr/bin/env python3
"""
Baseline inference script for the SQL Query Writing Environment.

Uses the OpenAI-compatible client to solve all 3 tasks.
Each task allows up to 5 attempts; the agent sees feedback from prior attempts.

Required environment variables:
    HF_TOKEN       — API key for the LLM

Optional environment variables (have defaults):
    API_BASE_URL   — LLM API endpoint (default: https://router.huggingface.co/v1)
    MODEL_NAME     — model identifier (default: Qwen/Qwen2.5-72B-Instruct)
    SERVER_URL     — SQL environment server URL (default: http://localhost:8000)
"""

import os
import sys
import json
import requests

# ---------------------------------------------------------------------------
# Config — all LLM credentials read from env vars per spec
# ---------------------------------------------------------------------------
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY      = os.environ.get("API_KEY") or os.environ.get("HF_TOKEN", "")
SERVER_URL   = os.environ.get("SERVER_URL", "https://khyaal-d-sql-query-env.hf.space")

TEMPERATURE = 0.0
MAX_TOKENS  = 400


def strip_markdown(text: str) -> str:
    text = text.strip()
    if "```sql" in text:
        text = text.split("```sql", 1)[1].split("```", 1)[0]
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0]
    return text.strip()


def call_llm(client, prompt: str) -> str:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
    )
    return response.choices[0].message.content.strip()


def run_task(client, task: dict) -> dict:
    task_id      = task["task_id"]
    difficulty   = task["difficulty"]
    question     = task["question"]
    max_attempts = task["max_attempts"]

    print(f"[START] task={task_id} difficulty={difficulty}")
    print(f"Q: {question}")

    best_score = 0.0
    best_query = ""
    last_feedback = ""

    # Reset environment for this task
    try:
        resp = requests.post(f"{SERVER_URL}/reset", json={"task_id": task_id}, timeout=30)
        resp.raise_for_status()
        obs = resp.json()
    except Exception as exc:
        print(f"  ERROR: Failed to reset task {task_id}: {exc}")
        print(f"[END] task={task_id} score={best_score:.3f}")
        return {"task_id": task_id, "difficulty": difficulty, "score": best_score, "best_query": best_query}

    obs_data           = obs.get("observation") or obs
    schema_description = obs_data.get("schema_description", "")
    episode_id         = obs_data.get("episode_id")

    for attempt in range(max_attempts):
        previous = ""
        if attempt > 0 and last_feedback:
            previous = f"\n\nFeedback from your last attempt: {last_feedback}"

        prompt = (
            f"You are an expert SQLite SQL writer.\n\n"
            f"{schema_description}\n\n"
            f"Question: {question}\n\n"
            f"Write ONLY the SQL query — no explanation, no markdown fences."
            f"{previous}"
        )

        try:
            raw_query = call_llm(client, prompt)
        except Exception as exc:
            print(f"  [attempt {attempt+1}] LLM error: {exc}")
            break

        query = strip_markdown(raw_query)
        print(f"[STEP] task={task_id} attempt={attempt+1} query={query[:100]}")

        try:
            step_resp = requests.post(
                f"{SERVER_URL}/step",
                json={"action": {"query": query}, "episode_id": episode_id},
                timeout=30,
            )
            step_resp.raise_for_status()
            step_obs = step_resp.json()
        except Exception as exc:
            print(f"  [attempt {attempt+1}] Server error: {exc}")
            break

        reward        = step_obs.get("reward")
        score         = float(reward) if reward is not None else 0.0
        obs_inner     = step_obs.get("observation") or {}
        last_feedback = obs_inner.get("feedback", "")

        print(f"[STEP] task={task_id} attempt={attempt+1} score={score:.3f} feedback={last_feedback[:80]}")

        if score > best_score:
            best_score = score
            best_query = query

        if step_obs.get("done") or obs_inner.get("done"):
            break

    print(f"[END] task={task_id} score={best_score:.3f}")
    return {
        "task_id":    task_id,
        "difficulty": difficulty,
        "score":      best_score,
        "best_query": best_query,
    }


def main():
    print(f"SQL Query Writing Environment — Baseline ({MODEL_NAME})")
    print(f"LLM API: {API_BASE_URL}")
    print(f"Server:  {SERVER_URL}")

    # Set up OpenAI client
    try:
        from openai import OpenAI
        client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    except Exception as exc:
        print(f"WARNING: Could not initialise LLM client: {exc}")
        client = None

    # Verify server is reachable
    try:
        health = requests.get(f"{SERVER_URL}/health", timeout=60)
        health.raise_for_status()
        print(f"Server health: OK")
    except Exception as exc:
        print(f"WARNING: Server health check failed: {exc}")

    # Fetch task list
    tasks = []
    try:
        tasks_resp = requests.get(f"{SERVER_URL}/tasks", timeout=30)
        tasks_resp.raise_for_status()
        tasks = tasks_resp.json()["tasks"]
    except Exception as exc:
        print(f"WARNING: Could not fetch tasks: {exc}")
        # Fallback: hardcoded task list so script always produces output
        tasks = [
            {"task_id": 1, "difficulty": "easy",   "question": "List the name and city of every customer from France, ordered alphabetically by name.", "max_attempts": 5},
            {"task_id": 2, "difficulty": "medium",  "question": "What is the total revenue for each product category from completed orders? Return the category and total_revenue (sum of quantity x unit_price), ordered by total_revenue descending.", "max_attempts": 5},
            {"task_id": 3, "difficulty": "hard",    "question": "Find customers who placed at least one completed order in January 2024 AND at least one completed order in February 2024. For each such customer return their name and total_spending (sum of quantity x unit_price across both months), ordered by total_spending descending.", "max_attempts": 5},
        ]

    results = {}
    for task in tasks:
        if client is not None:
            result = run_task(client, task)
        else:
            result = {"task_id": task["task_id"], "difficulty": task["difficulty"], "score": 0.0, "best_query": ""}
        results[f"task_{result['task_id']}"] = result

    avg = sum(r["score"] for r in results.values()) / len(results) if results else 0.0

    print(f"\n{'='*60}")
    print("BASELINE RESULTS")
    print(f"{'='*60}")
    print(f"{'Task':<8} {'Difficulty':<10} {'Score'}")
    print(f"{'-'*30}")
    for r in results.values():
        print(f"  {r['task_id']:<6} {r['difficulty']:<10} {r['score']:.3f}")
    print(f"{'-'*30}")
    print(f"  {'Average':<16} {avg:.3f}")
    print(f"{'='*60}")

    output = {
        "model":         MODEL_NAME,
        "api_base_url":  API_BASE_URL,
        "server_url":    SERVER_URL,
        "results":       results,
        "average_score": round(avg, 4),
    }
    print(f"\nJSON summary:\n{json.dumps(output, indent=2)}")


if __name__ == "__main__":
    main()
