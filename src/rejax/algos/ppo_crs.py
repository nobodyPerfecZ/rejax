import chex
import jax
from flax import struct

from rejax.statistics import conditional_value_at_risk

from .ppo import PPO


class PPOCVaRRejectionSampling(PPO):
    alpha: chex.Scalar = struct.field(pytree_node=True, default=0.05)
    threshold: chex.Scalar = struct.field(pytree_node=True, default=0.05)

    def train_iteration(self, ts):
        ts, _ = self.collect_trajectories(ts)
        cvar = conditional_value_at_risk(ts.episode_return, self.alpha)
        cvar = jax.lax.cond(
            cvar >= 0.0,
            lambda: (1 - self.threshold) * cvar,
            lambda: (1 + self.threshold) * cvar,
        )

        new_ts = super().train_iteration(ts)
        new_ts, _ = self.collect_trajectories(new_ts)
        new_cvar = conditional_value_at_risk(new_ts.episode_return, self.alpha)

        next_ts = jax.lax.cond(
            new_cvar >= cvar,
            lambda: new_ts,
            lambda: ts,
        )

        return next_ts
