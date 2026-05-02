"""Channel perturbation for robustness experiments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChannelPerturbation:
    """Scale nominal noise sigma and gateway reception probability."""

    noise_sigma_multiplier: float = 1.0
    reception_prob_multiplier: float = 1.0

    def effective_reception_prob(self, base_prob: float) -> float:
        p = float(base_prob) * float(self.reception_prob_multiplier)
        return max(0.0, min(1.0, p))

    def effective_noise_sigma_db(self, base_sigma_db: float) -> float:
        return max(1e-6, float(base_sigma_db) * float(self.noise_sigma_multiplier))
