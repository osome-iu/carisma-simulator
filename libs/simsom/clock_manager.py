import numpy as np

def predict_average_delta(n_users, puda, return_unit='days'):
    """
    Predict average inter-event time for a social system (Poisson assumption).
    Δt = 1 / (n_users × puda), expressed in requested unit.
    """
    if n_users <= 0:
        raise ValueError("Number of users must be positive")
    if puda <= 0:
        raise ValueError("PUDA must be positive")
    if return_unit not in ['seconds', 'minutes', 'hours', 'days']:
        raise ValueError("Invalid return_unit")

    lambda_total = n_users * puda  # total events per day
    delta_days = 1.0 / lambda_total

    factors = {
        'days': 1.0,
        'hours': 24.0,
        'minutes': 1440.0,
        'seconds': 86400.0
    }
    return delta_days * factors[return_unit]


class ClockManager:
    """
    Adaptive stochastic clock for social simulations.

    Features:
    - Automatically scales deltas from n_users and PUDA.
    - Log-normal variability around expected mean.
    - Circadian rhythm modulation (normalized to preserve global mean).
    - Optional spikes, configurable as 'burst' (shorter deltas) or 'delay' (longer deltas).
    """

    def __init__(
        self,
        n_users: int,
        puda: float,
        log_mean: float = -8.5,
        log_sigma: float = 1.0,
        circadian: bool = True,
        spike_prob: float = 0.0005,
        spike_magnitude: tuple = (5, 20),
        spike_duration: int = 3,
        spike_mode: str = "delay"  # "delay" (slower) or "burst" (faster)
    ):
        self.current_time = 0.0
        self.n_users = n_users
        self.puda = puda
        self.log_mean = log_mean
        self.log_sigma = log_sigma
        self.circadian = circadian
        self.spike_prob = spike_prob
        self.spike_magnitude = spike_magnitude
        self.spike_duration = spike_duration
        self.spike_mode = spike_mode.lower()
        self._spike_remaining = 0

        # expected delta in days
        self.target_delta = predict_average_delta(self.n_users, self.puda, return_unit='days')

        # calibrate log-normal scaling
        sample = np.random.lognormal(mean=self.log_mean, sigma=self.log_sigma, size=10000)
        base_mean = np.mean(sample)
        self.scale_factor = self.target_delta / base_mean

    def _base_delta(self) -> float:
        """Draw base inter-event time from log-normal, scaled to target mean."""
        return np.random.lognormal(mean=self.log_mean, sigma=self.log_sigma) * self.scale_factor

    def _circadian_factor(self, t: float) -> float:
        """24h sinusoidal modulation, normalized to mean=1."""
        frac_day = t % 1.0
        raw = np.sin(frac_day * 2 * np.pi - np.pi / 2) + 1.2
        mean_raw = 1.2  # average over a cycle
        return raw / mean_raw

    def next_time(self) -> float:
        """Return current time, then advance clock."""
        current = self.current_time
        delta = self._base_delta()

        # spike regime
        if self._spike_remaining == 0 and np.random.rand() < self.spike_prob:
            self._spike_remaining = np.random.randint(1, self.spike_duration + 1)
            self._spike_multiplier = np.random.uniform(*self.spike_magnitude)

        if self._spike_remaining > 0:
            if self.spike_mode == "delay":
                delta *= self._spike_multiplier
            elif self.spike_mode == "burst":
                delta /= self._spike_multiplier
            self._spike_remaining -= 1

        # circadian rhythm
        if self.circadian:
            delta *= self._circadian_factor(self.current_time)

        self.current_time += delta
        return current
