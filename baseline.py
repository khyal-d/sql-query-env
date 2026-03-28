#!/usr/bin/env python3
"""
Baseline inference script for the SQL Query Writing Environment.

Uses the OpenAI API (gpt-4o-mini by default) to solve all 3 tasks.
Each task allows up to 5 attempts; the agent sees feedback from prior attempts.

Environment variables:
    OPENAI_API_KEY   — required
    BASE_URL         — server base URL (default: http://localhost:8000)
    BASELINE_MODEL   — OpenAI model ID (default: gpt-4o-mini)

Usage:
    # Against local server (start with: uvicorn app:app --port 8000)
    OPENAI_API_KEY=sk-... python baseline.py

    # Against deployed HF Space
    OPENAI_API_KEY=sk-... BASE_URL=https://your-space.hf.space python baseline.py
"""

import os
import sys
import json
import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_URL   = os.environ.get("BASE_URL", "http://localhost:8000")
API_KEY    = os.environ.get("OPENAI_API_KEY")
MODEL      = os.environ.get("BASELINE_MODEL", "gpt-4o-mini")
MAX_RETRIES = 3


def strip_markdown(text: str) -> str:
    text = text.strip()
    if "```sql" in text:
        text = text.split("```sql", 1)[1].split("```", 1)[0]
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0]
    return text.strip()


def call_openai(client, prompt: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()


def run_task(client, task: dict) -> dict:
    task_id    = task["task_id"]
    difficulty = task["difficulty"]
    question   = task["question"]
    max_attempts = task["max_attempts"]

    print(f"\n{'='*60}")
    print(f"Task {task_id} [{difficulty.upper()}]")
    print(f"Q: {question}")
    print(f"{'='*60}")

    # Reset environment for this task
    resp = requests.post(f"{BASE_URL}/reset", json={"task_id": task_id}, timeout=30)
    resp.raise_for_status()
    obs = resp.json()

    schema_description = obs.get("schema_description", "")
    best_score = 0.0
    best_query = ""
    last_feedback = ""

    for attempt in range(max_attempts):
        # Build prompt
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
            raw_query = call_openai(client, prompt)
        except Exception as exc:
            print(f"  [attempt {attempt+1}] OpenAI error: {exc}")
            break

        query = strip_markdown(raw_query)
        print(f"\n  [attempt {attempt+1}] Query: {query[:100]}{'...' if len(query)>100 else ''}")

        # Submit to environment
        try:
            step_resp = requests.post(
                f"{BASE_URL}/step",
                json={"query": query},
                timeout=30,
            )
            step_resp.raise_for_status()
            step_obs = step_resp.json()
        except Exception as exc:
            print(f"  [attempt {attempt+1}] Server error: {exc}")
            break

        score = step_obs.get("reward") or 0.0
        last_feedback = step_obs.get("feedback", "")

        print(f"  [attempt {attempt+1}] Score: {score:.3f} | {last_feedback[:80]}")

        if score > best_score:
            best_score = score
            best_query = query

        if step_obs.get("done"):
            break

    return {
        "task_id":    task_id,
        "difficulty": difficulty,
        "score":      best_score,
        "best_query": best_query,
    }


def main():
    if not API_KEY:
        print("ERROR: OPENAI_API_KEY environment variable is not set.")
        sys.exit(1)

    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: openai package not installed. Run: pip install openai")
        sys.exit(1)

    client = OpenAI(api_key=API_KEY)

    # Verify server is reachable
    try:
        health = requests.get(f"{BASE_URL}/health", timeout=10)
        health.raise_for_status()
    except Exception as exc:
        print(f"ERROR: Cannot reach server at {BASE_URL}: {exc}")
        sys.exit(1)

    # Fetch task list
    tasks_resp = requests.get(f"{BASE_URL}/tasks", timeout=10)
    tasks_resp.raise_for_status()
    tasks = tasks_resp.json()["tasks"]

    print(f"\nSQL Query Writing Environment — Baseline ({MODEL})")
    print(f"Server: {BASE_URL}")

    results = {}
    for task in tasks:
        result = run_task(client, task)
        results[f"task_{result['task_id']}"] = result

    # ── Summary ──────────────────────────────────────────────────────────
    avg = sum(r["score"] for r in results.values()) / len(results)

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

    # Machine-readable output for CI / automated evaluation
    output = {
        "model":         MODEL,
        "base_url":      BASE_URL,
        "results":       results,
        "average_score": round(avg, 4),
    }
    print(f"\nJSON summary:\n{json.dumps(output, indent=2)}")
    return output


if __name__ == "__main__":
    main()
