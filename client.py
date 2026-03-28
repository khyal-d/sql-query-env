"""
SQLEnv — typed Python client for the SQL Query Writing Environment.

Usage:
    from client import SQLEnv, SQLAction

    with SQLEnv(base_url="https://your-space.hf.space").sync() as env:
        result = env.reset(task_id=1)
        print(result.observation.question)

        result = env.step(SQLAction(query="SELECT name, city FROM customers WHERE country='France' ORDER BY name"))
        print(result.reward, result.observation.feedback)
"""

from typing import Any

from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult

from models import SQLAction, SQLObservation, SQLState


class SQLEnv(EnvClient[SQLAction, SQLObservation, SQLState]):
    """WebSocket client for the SQL Query Writing Environment."""

    # ── wire → action ────────────────────────────────────────────────────
    def _step_payload(self, action: SQLAction) -> dict:
        return {"query": action.query}

    # ── wire → StepResult ────────────────────────────────────────────────
    def _parse_result(self, payload: dict) -> StepResult:
        obs_data = payload.get("observation", {})
        return StepResult(
            observation=SQLObservation(
                done=payload.get("done", False),
                reward=payload.get("reward"),
                task_id=obs_data.get("task_id", 0),
                difficulty=obs_data.get("difficulty", ""),
                question=obs_data.get("question", ""),
                schema_description=obs_data.get("schema_description", ""),
                query_result=obs_data.get("query_result"),
                columns=obs_data.get("columns"),
                error_message=obs_data.get("error_message"),
                attempts_remaining=obs_data.get("attempts_remaining", 0),
                feedback=obs_data.get("feedback", ""),
            ),
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    # ── wire → State ─────────────────────────────────────────────────────
    def _parse_state(self, payload: dict) -> SQLState:
        return SQLState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_id=payload.get("task_id", 0),
            difficulty=payload.get("difficulty", ""),
            max_attempts=payload.get("max_attempts", 5),
            best_score=payload.get("best_score", 0.0),
        )
