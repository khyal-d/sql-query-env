"""
SQLEnvironment — OpenEnv environment for the SQL Query Writing task.

The HTTP server creates a new environment instance per request, so session
state is stored at the class level in _sessions, keyed by episode_id.
The client must pass episode_id in every step() request body.

Each episode:
  1. reset(task_id=N) creates a session, returns episode_id + schema + question.
  2. The agent calls step(action, episode_id=...) up to 5 times.
  3. Each step executes the query on an in-memory SQLite DB and returns:
       - the query result (first 10 rows)
       - a Jaccard-based reward signal (0.0 – 1.0)
       - human-readable feedback
  4. Episode ends when the agent achieves a perfect score (1.0) or exhausts attempts.
"""

import uuid
import threading
from typing import Optional

from openenv.core.env_server import Environment

from tasks import (
    TASKS,
    SCHEMA_DESCRIPTION,
    create_database,
    execute_query,
    compute_score,
)
from models import SQLAction, SQLObservation, SQLState


class SQLEnvironment(Environment[SQLAction, SQLObservation, SQLState]):
    """OpenEnv environment: SQL Query Writing."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    # Class-level session store — survives across per-request instances
    _sessions: dict = {}
    _lock = threading.Lock()

    # ------------------------------------------------------------------
    def __init__(self) -> None:
        super().__init__()
        self._current_episode_id: Optional[str] = None

    # ------------------------------------------------------------------
    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: int = 1,
        **kwargs,
    ) -> SQLObservation:
        """Start a new episode for the given task_id (1, 2, or 3)."""
        if task_id not in TASKS:
            task_id = 1

        episode_id = episode_id or str(uuid.uuid4())
        task = TASKS[task_id]
        conn = create_database()

        expected_result, err = execute_query(conn, task.expected_sql)
        if err:
            raise RuntimeError(f"Expected SQL failed: {err}")

        state = SQLState(
            episode_id=episode_id,
            step_count=0,
            task_id=task_id,
            difficulty=task.difficulty,
            max_attempts=task.max_attempts,
            best_score=0.0,
        )

        with self._lock:
            SQLEnvironment._sessions[episode_id] = {
                "task":     task,
                "conn":     conn,
                "expected": expected_result,
                "state":    state,
            }

        self._current_episode_id = episode_id

        return SQLObservation(
            done=False,
            reward=None,
            episode_id=episode_id,
            task_id=task_id,
            difficulty=task.difficulty,
            question=task.question,
            schema_description=SCHEMA_DESCRIPTION,
            attempts_remaining=task.max_attempts,
            feedback=(
                "Write a SQL query that answers the question above. "
                "Your query will be executed against the database and scored."
            ),
        )

    # ------------------------------------------------------------------
    def step(
        self,
        action: SQLAction,
        timeout_s: Optional[float] = None,
        episode_id: Optional[str] = None,
        **kwargs,
    ) -> SQLObservation:
        """Execute the agent's SQL query and return scored observation."""
        eid = episode_id or self._current_episode_id

        with self._lock:
            session = SQLEnvironment._sessions.get(eid)

        if session is None:
            raise RuntimeError(
                f"No active session for episode_id={eid!r}. Call reset() first."
            )

        task     = session["task"]
        conn     = session["conn"]
        expected = session["expected"]
        state    = session["state"]

        state.step_count += 1
        attempts_remaining = max(0, task.max_attempts - state.step_count)

        # ── Execute the agent's query ──────────────────────────────────
        actual_rows, error = execute_query(conn, action.query)

        if error:
            done = attempts_remaining == 0
            if done:
                with self._lock:
                    SQLEnvironment._sessions.pop(eid, None)
            return SQLObservation(
                done=done,
                reward=0.0,
                episode_id=eid,
                task_id=state.task_id,
                difficulty=task.difficulty,
                question=task.question,
                schema_description=SCHEMA_DESCRIPTION,
                error_message=error,
                attempts_remaining=attempts_remaining,
                feedback=f"SQL Error: {error}",
            )

        # ── Score result set ───────────────────────────────────────────
        score = compute_score(actual_rows, expected)
        state.best_score = max(state.best_score, score)

        done = (score == 1.0) or (attempts_remaining == 0)

        # ── Build feedback message ─────────────────────────────────────
        expected_count = len(expected)
        actual_count   = len(actual_rows)

        if score == 1.0:
            feedback = f"Correct! Your query returns exactly the expected {expected_count} row(s)."
        elif score >= 0.75:
            feedback = (
                f"Very close ({score:.0%} match). "
                f"You returned {actual_count} row(s), expected {expected_count}. "
                "Check for missing or extra rows."
            )
        elif score >= 0.4:
            feedback = (
                f"Partial match ({score:.0%}). "
                f"You returned {actual_count} row(s), expected {expected_count}. "
                "Review your JOIN conditions or WHERE filters."
            )
        elif score > 0.0:
            feedback = (
                f"Weak match ({score:.0%}). "
                f"You returned {actual_count} row(s), expected {expected_count}. "
                "Rethink your query approach."
            )
        else:
            if actual_count == 0:
                feedback = "No rows returned. Check table names, column names, and filter conditions."
            else:
                feedback = (
                    f"No matching rows ({actual_count} returned, 0 correct). "
                    "Check your filter values and join keys."
                )

        if done and score < 1.0:
            feedback += f" Episode ended. Best score this episode: {state.best_score:.2f}."

        if done:
            with self._lock:
                SQLEnvironment._sessions.pop(eid, None)

        return SQLObservation(
            done=done,
            reward=score,
            episode_id=eid,
            task_id=state.task_id,
            difficulty=task.difficulty,
            question=task.question,
            schema_description=SCHEMA_DESCRIPTION,
            query_result=actual_rows[:10],
            columns=list(actual_rows[0].keys()) if actual_rows else None,
            attempts_remaining=attempts_remaining,
            feedback=feedback,
        )

    # ------------------------------------------------------------------
    @property
    def state(self) -> SQLState:
        eid = self._current_episode_id
        with self._lock:
            session = SQLEnvironment._sessions.get(eid)
        if session:
            return session["state"]
        return SQLState()

    def close(self) -> None:
        pass  # Sessions are managed at class level; cleaned up when done=True
