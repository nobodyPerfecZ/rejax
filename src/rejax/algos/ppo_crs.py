import chex
import jax
import jax.numpy as jnp
from flax import struct

from rejax.statistics import conditional_value_at_risk

from .ppo import PPO


class PPOCVaRRejectionSampling(PPO):
    alpha: chex.Scalar = struct.field(pytree_node=True, default=0.05)
    threshold: chex.Scalar = struct.field(pytree_node=True, default=0.05)

    def train_iteration(self, ts):
        ts, _ = self.collect_trajectories(ts)
        cvar = conditional_value_at_risk(ts.episode_return, self.alpha)
        cvar_threshold = cvar - jnp.abs(cvar) * self.threshold

        new_ts, new_metrics = super().train_iteration(ts)
        new_ts, _ = self.collect_trajectories(new_ts)
        new_cvar = conditional_value_at_risk(new_ts.episode_return, self.alpha)
        accept = new_cvar >= cvar_threshold

        new_metrics.update(
            {
                "crs/cvar": cvar,
                "crs/cvar_threshold": cvar_threshold,
                "crs/new_cvar": new_cvar,
                "crs/accept": accept,
            }
        )
        metrics = jax.tree.map(jnp.zeros_like, new_metrics)

        next_ts, next_metrics = jax.lax.cond(
            accept,
            lambda: (new_ts, new_metrics),
            lambda: (ts, metrics),
        )

        return next_ts, next_metrics
