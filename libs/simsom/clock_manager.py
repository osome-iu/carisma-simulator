import numpy as np
from typing import Dict, List, Optional, Any
from collections import deque


class ClockManager:
    def __init__(self, seed: Optional[int] = None, circadian: bool = True, debug: bool = True):
        self.rng = np.random.default_rng(seed)
        self.clock = 0.0
        self.timestamps_today = deque([self.clock])  # More efficient than list for pop(0)
        self.circadian = circadian
        self.day = 0
        self.debug = debug
        self.sync_queue = deque()  # Efficient FIFO/LIFO operations
        
        # Tracking
        self.extra_used_today = 0
        self.last_day_report: Optional[Dict[str, Any]] = None
        
        # Cached circadian distribution
        self._circadian_cache: Optional[tuple] = None

    def _circadian_pdf(self, t: np.ndarray) -> np.ndarray:
        """
        Circadian activity probability density function (PDF).
        Two peaks: one in the morning (~0.3) and one in the evening (~0.7), plus a baseline.
        """
        morning_peak = 0.6 * np.exp(-0.5 * ((t - 0.3) / 0.1) ** 2)
        evening_peak = 0.9 * np.exp(-0.5 * ((t - 0.7) / 0.15) ** 2)
        baseline = 0.2
        return morning_peak + evening_peak + baseline

    def _get_circadian_distribution(self, force_refresh: bool = False) -> tuple:
        """
        Return a cached circadian distribution or compute it if not available.
        Returns (t_grid, cdf) for interpolation.
        """
        if self._circadian_cache is None or force_refresh:
            n_samples = 10_000
            t_grid = np.linspace(0, 1, n_samples)
            pdf = self._circadian_pdf(t_grid)
            cdf = np.cumsum(pdf)
            cdf /= cdf[-1]  # Normalize to [0, 1]
            self._circadian_cache = (t_grid, cdf)
        
        return self._circadian_cache

    def start_new_day(self, actions_per_user_vector: np.ndarray) -> None:
        """
        Initialize a new day with the expected number of actions per user.
        Handles synchronization between pending requests.
        """

        # If there are still timestamps left from the current day, enqueue the request
        if len(self.timestamps_today) > 0:
            self.sync_queue.append(actions_per_user_vector.copy())
            if self.debug:
                print(
                    f"[ClockManager] New day request queued. "
                    f"Requests in queue: {len(self.sync_queue)}. "
                    f"Timestamps remaining: {len(self.timestamps_today)}."
                )
            return
        
        # If this is the very first day, start immediately
        if self.clock == 0.0:
            self._start_new_day_internal(actions_per_user_vector)
        # Otherwise, process the next queued request if available
        elif self.sync_queue:
            next_actions = self.sync_queue.popleft()
            self._start_new_day_internal(next_actions)
        else:
            # Edge case: no requests queued but the current day has finished
            if self.debug:
                print(f"[ClockManager] No requests queued for day {self.day}")

    def _start_new_day_internal(self, actions_per_user_vector: np.ndarray) -> None:
        """Internal logic to initialize a new day."""
        # Print report for the previous day
        if self.last_day_report and self.debug:
            print(f"[ClockManager] Day {self.day-1} completed: {self.last_day_report}")
        
        # Reset counters
        self.extra_used_today = 0
        total_actions = int(np.sum(actions_per_user_vector))
        
        if total_actions <= 0:
            self.timestamps_today = deque()
            self.last_day_report = {
                "day": self.day,
                "actions_expected": 0,
                "timestamps_generated": 0,
                "extra_used": 0,
            }
            if self.debug:
                print(f"[ClockManager] Day {self.day}: no actions expected.")
            self.day += 1
            return
        
        # Generate timestamps
        if self.circadian:
            t_grid, cdf = self._get_circadian_distribution()
            u = self.rng.random(total_actions)
            fractions = np.interp(u, cdf, t_grid)
        else:
            fractions = self.rng.random(total_actions)
        
        # Sort and convert to absolute timestamps
        fractions.sort()
        self.timestamps_today = deque(self.day + fractions)
        
        self.last_day_report = {
            "day": self.day,
            "actions_expected": total_actions,
            "timestamps_generated": total_actions,
            "extra_used": 0,
        }
        
        if self.debug:
            print(
                f"[ClockManager] Day {self.day}: generated {total_actions} timestamps "
                f"(range: {self.timestamps_today[0]:.4f} - {self.timestamps_today[-1]:.4f})"
            )
        
        self.day += 1

    def next_timestamp(self) -> float:
        """
        Return the next timestamp in chronological order.
        Handles fallbacks and queued requests.
        """

        # Case 1: No timestamps available and no queued requests
        if not self.timestamps_today and not self.sync_queue:
            # Fallback: generate extra timestamps at the very end of the day
            self.extra_used_today += 1
            ts = (self.day - 1) + 0.999999 + (self.extra_used_today * 1e-6)
            
            if self.last_day_report:
                self.last_day_report["extra_used"] = self.extra_used_today
            
            if self.debug:
                print(
                    f"[ClockManager] Extra timestamp generated: {ts:.6f} "
                    f"(#{self.extra_used_today})"
                )

            return ts
            
        # Case 2: No timestamps left but a request is queued
        if not self.timestamps_today and self.sync_queue:
            next_actions = self.sync_queue.popleft()
            self._start_new_day_internal(next_actions)
            # Return the first timestamp of the new day (or recurse if empty)
            if not self.timestamps_today:
                return self.next_timestamp()

        # Case 3: Normal case, timestamps available
        if self.timestamps_today:
            self.clock = self.timestamps_today.popleft()
            return self.clock
        
        # Edge case: inconsistent internal state
        raise RuntimeError("ClockManager: inconsistent state!")

    def get_current_time(self) -> float:
        """Return the current timestamp without advancing the clock."""
        return self.clock

    def get_current_day(self) -> int:
        """Return the current day index (0-based)."""
        return max(0, self.day - 1)

    def get_remaining_timestamps(self) -> int:
        """Return the number of remaining timestamps for the current day."""
        return len(self.timestamps_today)

    def get_queue_length(self) -> int:
        """Return the number of pending day-start requests in the queue."""
        return len(self.sync_queue)

    def get_stats(self) -> Dict[str, Any]:
        """Return detailed ClockManager statistics."""
        return {
            "current_time": self.clock,
            "current_day": self.get_current_day(),
            "timestamps_remaining": self.get_remaining_timestamps(),
            "queue_length": self.get_queue_length(),
            "extra_used_today": self.extra_used_today,
            "last_day_report": self.last_day_report,
            "circadian_enabled": self.circadian,
        }

    def reset(self, seed: Optional[int] = None) -> None:
        """Completely reset the ClockManager to its initial state."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        
        self.clock = 0.0
        self.timestamps_today.clear()
        self.day = 0
        self.sync_queue.clear()
        self.extra_used_today = 0
        self.last_day_report = None
        self._circadian_cache = None

    def __repr__(self) -> str:
        return (f"ClockManager(day={self.get_current_day()}, "
                f"time={self.clock:.4f}, "
                f"remaining={self.get_remaining_timestamps()}, "
                f"queued={self.get_queue_length()})")
