from typing import List, Optional, Dict, Any
from openenv.core.env_server import Action, Observation, State


class SQLAction(Action):
    """Submit a SQL query to answer the question."""
    query: str


class SQLObservation(Observation):
    """What the agent sees after each step.

    Inherited from Observation:
        done: bool
        reward: Optional[float]
    """
    episode_id: Optional[str] = None              # Pass this back in every step() request
    task_id: int
    difficulty: str                              # "easy" | "medium" | "hard"
    question: str                                # Natural language question to answer
    schema_description: str                      # Full schema + sample data
    query_result: Optional[List[Dict[str, Any]]] = None  # First 10 rows of agent's result
    columns: Optional[List[str]] = None          # Column names of the result
    error_message: Optional[str] = None          # SQL execution error (if any)
    attempts_remaining: int = 0
    feedback: str = ""                           # Human-readable scoring feedback


class SQLState(State):
    """Episode metadata.

    Inherited from State:
        episode_id: Optional[str]
        step_count: int
    """
    task_id: int = 0
    difficulty: str = ""
    max_attempts: int = 5
    best_score: float = 0.0
