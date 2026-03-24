"""
Hysteresis-based goal tracking to prevent "see target but miss it" failures.

Uses dual thresholds and multi-round verification before committing to
goal-reached / goal-lost state transitions.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GoalManagerConfig:
    enter_threshold: float = 0.7   # confidence required to enter REACHED state
    exit_threshold: float = 0.4    # confidence must fall below this to exit REACHED state
    verify_rounds: int = 3         # sliding window size for multi-round votes
    verify_k: int = 2              # number of positive votes needed within window


class GoalManager:
    """
    Tracks whether a navigation goal has been reached using hysteresis and
    multi-round verification to reduce spurious state transitions.

    State machine:
      SEEKING  --[k of N frames >= enter_threshold]--> REACHED
      REACHED  --[any frame < exit_threshold]---------> VERIFYING_EXIT
      VERIFYING_EXIT --[k of N frames < exit_threshold]--> SEEKING
      VERIFYING_EXIT --[any frame >= exit_threshold]---> REACHED
    """

    SEEKING = "SEEKING"
    REACHED = "REACHED"
    VERIFYING_EXIT = "VERIFYING_EXIT"

    def __init__(self, config: Optional[GoalManagerConfig] = None):
        self.config = config or GoalManagerConfig()
        self._state = self.SEEKING
        self._window: deque = deque(maxlen=self.config.verify_rounds)

    @property
    def state(self) -> str:
        return self._state

    def is_reached(self) -> bool:
        return self._state == self.REACHED

    def update(self, confidence: float) -> str:
        """
        Feed a new confidence score and return the updated state.

        Parameters
        ----------
        confidence : float
            Model confidence that the goal is currently visible/reached (0–1).

        Returns
        -------
        str
            Current state after processing this frame.
        """
        self._window.append(confidence)
        cfg = self.config

        if self._state == self.SEEKING:
            positive = sum(1 for c in self._window if c >= cfg.enter_threshold)
            if len(self._window) >= cfg.verify_rounds and positive >= cfg.verify_k:
                self._state = self.REACHED

        elif self._state == self.REACHED:
            if confidence < cfg.exit_threshold:
                self._state = self.VERIFYING_EXIT

        elif self._state == self.VERIFYING_EXIT:
            if confidence >= cfg.enter_threshold:
                # Snap back — likely a momentary occlusion
                self._state = self.REACHED
            else:
                negative = sum(1 for c in self._window if c < cfg.exit_threshold)
                if negative >= cfg.verify_k:
                    self._state = self.SEEKING

        return self._state

    def reset(self):
        """Reset state and history (e.g., after a new goal is set)."""
        self._state = self.SEEKING
        self._window.clear()
