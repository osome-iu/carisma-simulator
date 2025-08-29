import numpy as np

class ClockManager:
    def __init__(self, n_users: int = 1000, circadian: bool = True, spike_prob: float = 0.001):
        self.current_time = 0.0
        self.n_users = n_users
        self.circadian = circadian
        self.spike_prob = spike_prob

    def _base_delta(self):
        """
        Genera un delta heavy-tailed usando log-normal.
        """
        return np.random.lognormal(mean=-8.5, sigma=1.0)  # ordini di grandezza ~0.0001–0.05

    def _circadian_factor(self, t):
        """
        Modula i delta in base all'ora del giorno.
        """
        frac_day = t % 1.0
        # attività alta di giorno, bassa di notte
        activity = np.sin(frac_day * 2 * np.pi - np.pi/2) + 1.2  # >0
        return 1 / activity

    def next_time(self):
        current = self.current_time

        # base distribution
        delta = self._base_delta()

        # occasional spikes
        if np.random.rand() < self.spike_prob:
            delta *= np.random.randint(10, 50)

        # user scaling (più utenti = attività più fitta = delta più piccolo)
        delta /= np.log1p(self.n_users)

        # circadian modulation
        if self.circadian:
            delta *= self._circadian_factor(self.current_time)

        self.current_time += delta

        return current
