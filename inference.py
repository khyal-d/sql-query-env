#!/usr/bin/env python3
"""
Baseline inference script for the SQL Query Writing Environment.

Uses the OpenAI-compatible client to solve all 3 tasks.
Each task allows up to 5 attempts; the agent sees feedback from prior attempts.

Required environment variables:
    API_BASE_URL   — LLM API endpoint (e.g. https://api.openai.com/v1)
    MODEL_NAME     — model identifier (e.g. gpt-4o-mini)
    HF_TOKEN       — API key for the LLM

Optional environment variables:
    SERVER_URL     — SQL environment server URL (default: http://localhost:8000)

Usage:
    # Against local server (start with: uvicorn app:app --port 8000)
    API_BASE_URL=https://api.openai.com/v1 MODEL_NAME=gpt-4o-mini HF_TOKEN=sk-... python inference.py

    # Against deployed HF Space
    API_BASE_URL=https://api.openai.com/v1 MODEL_NAME=gpt-4o-mini HF_TOKEN=sk-... SERVER_URL=https://your-space.hf.space python inference.py
"""

import os
import sys
import json
import requests

# ---------------------------------------------------------------------------
# Config — all LLM credentials read from required env vars per spec
# ---------------------------------------------------------------------------
API_BASE_URL = os.environ.get("API_BASE_URL")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN")
SERVER_URL   = os.environ.get("SERVER_URL", "http://localhost:8000")

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

    print(f"\n{'='*60}")
    print(f"Task {task_id} [{difficulty.upper()}]")
    print(f"Q: {question}")
    print(f"{'='*60}")

    # Reset environment for this task
    try:
        resp = requests.post(f"{SERVER_URL}/reset", json={"task_id": task_id}, timeout=30)
        resp.raise_for_status()
        obs = resp.json()
    except Exception as exc:
        print(f"  ERROR: Failed to reset task {task_id}: {exc}")
        return {"task_id": task_id, "difficulty": difficulty, "score": 0.0, "best_query": ""}

    obs_data       = obs.get("observation", obs)  # nested under "observation"
    schema_description = obs_data.get("schema_description", "")
    episode_id    = obs_data.get("episode_id")
    best_score    = 0.0
    best_query    = ""
    last_feedback = ""

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
        print(f"\n  [attempt {attempt+1}] Query: {query[:100]}{'...' if len(query) > 100 else ''}")

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

        score         = step_obs.get("reward") or 0.0
        last_feedback = step_obs.get("observation", {}).get("feedback", "")

        print(f"  [attempt {attempt+1}] Score: {score:.3f} | {last_feedback[:80]}")

        if score > best_score:
            best_score = score
            best_query = query

        if step_obs.get("done") or step_obs.get("observation", {}).get("done"):
            break

    return {
        "task_id":    task_id,
        "difficulty": difficulty,
        "score":      best_score,
        "best_query": best_query,
    }


def main():
    # Validate required env vars
    missing = [v for v in ("API_BASE_URL", "HF_TOKEN") if not os.environ.get(v)]
    if missing:
        print(f"ERROR: Missing required environment variable(s): {', '.join(missing)}")
        print("Required: API_BASE_URL, MODEL_NAME, HF_TOKEN")
        sys.exit(1)

    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: openai package not installed. Run: pip install openai")
        sys.exit(1)

    client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

    # Verify environment server is reachable (generous timeout for HF Space cold start)
    try:
        health = requests.get(f"{SERVER_URL}/health", timeout=60)
        health.raise_for_status()
    except Exception as exc:
        print(f"ERROR: Cannot reach environment server at {SERVER_URL}: {exc}")
        sys.exit(1)

    # Fetch task list from server
    try:
        tasks_resp = requests.get(f"{SERVER_URL}/tasks", timeout=30)
        tasks_resp.raise_for_status()
        tasks = tasks_resp.json()["tasks"]
    except Exception as exc:
        print(f"ERROR: Failed to fetch tasks from {SERVER_URL}/tasks: {exc}")
        sys.exit(1)

    print(f"\nSQL Query Writing Environment — Baseline ({MODEL_NAME})")
    print(f"LLM API: {API_BASE_URL}")
    print(f"Server:  {SERVER_URL}")

    results = {}
    for task in tasks:
        result = run_task(client, task)
        results[f"task_{result['task_id']}"] = result

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

    output = {
        "model":         MODEL_NAME,
        "api_base_url":  API_BASE_URL,
        "server_url":    SERVER_URL,
        "results":       results,
        "average_score": round(avg, 4),
    }
    print(f"\nJSON summary:\n{json.dumps(output, indent=2)}")
    return output


if __name__ == "__main__":
    main()
