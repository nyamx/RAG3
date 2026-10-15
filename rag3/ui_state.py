from __future__ import annotations

from enum import Enum


class JobState(str, Enum):
    IDLE = "IDLE"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    PAUSING = "PAUSING"
    PAUSED = "PAUSED"
    RESUMING = "RESUMING"
    CANCELLING = "CANCELLING"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"
    COMPLETE = "COMPLETE"
    RESETTING = "RESETTING"


ACTION_POLICY: dict[JobState, frozenset[str]] = {
    JobState.IDLE: frozenset({"run", "run_all", "delete", "open_source", "open_artifacts"}),
    JobState.QUEUED: frozenset({"run", "run_all", "delete", "open_source", "open_artifacts"}),
    JobState.RUNNING: frozenset({"pause", "cancel", "open_source", "open_artifacts"}),
    JobState.PAUSING: frozenset({"cancel", "open_source", "open_artifacts"}),
    JobState.PAUSED: frozenset({"resume", "restart", "cancel", "delete", "open_source", "open_artifacts"}),
    JobState.RESUMING: frozenset({"pause", "cancel", "open_source", "open_artifacts"}),
    JobState.CANCELLING: frozenset({"open_source", "open_artifacts"}),
    JobState.CANCELLED: frozenset({"resume", "restart", "retry", "delete", "open_source", "open_artifacts"}),
    JobState.FAILED: frozenset({"retry", "restart", "delete", "open_source", "open_artifacts"}),
    JobState.COMPLETE: frozenset({"restart", "delete", "open_source", "open_artifacts"}),
    JobState.RESETTING: frozenset({"open_source", "open_artifacts"}),
}


def allowed_actions(state: JobState | str) -> frozenset[str]:
    return ACTION_POLICY[JobState(state)]
