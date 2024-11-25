import chex
import jax
import jax.numpy as jnp
from flax import struct

from rejax.statistics import conditional_value_at_risk

from .sac import SAC


class SACCVaRRejectionSampling(SAC):
    alpha: chex.Scalar = struct.field(pytree_node=True, default=0.05)
    threshold: chex.Scalar = struct.field(pytree_node=True, default=0.05)

    def train_iteration(self, ts):
        def undis_ret(last_ret, reward_and_done):
            reward, done = reward_and_done
            ret = jnp.where(
                done,
                jnp.zeros_like(reward),
                last_ret + reward,
            )
            return ret, (reward, done)

        _, batch = self.collect_transitions(ts)
        returns, _ = jax.lax.scan(
            undis_ret,
            jnp.zeros_like(batch.reward),
            (batch.reward, batch.done),
        )
        cvar = conditional_value_at_risk(returns, self.alpha)
        cvar = jax.lax.cond(
            cvar >= 0.0,
            lambda: (1 - self.threshold) * cvar,
            lambda: (1 + self.threshold) * cvar,
        )

        new_ts = super().train_iteration(ts)

        _, new_batch = self.collect_transitions(new_ts)
        new_returns, _ = jax.lax.scan(
            undis_ret,
            jnp.zeros_like(new_batch.reward),
            (new_batch.reward, new_batch.done),
        )
        new_cvar = conditional_value_at_risk(new_returns, self.alpha)

        next_ts = jax.lax.cond(
            new_cvar >= cvar,
            lambda: new_ts,
            lambda: ts,
        )

        return next_ts
