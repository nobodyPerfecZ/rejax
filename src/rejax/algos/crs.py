import chex
import jax
import jax.numpy as jnp
from flax import struct

from rejax.statistics import conditional_value_at_risk

from .ppo import PPO, AdvantageMinibatch


class PPOCVaRRejectionSampling(PPO):
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

        ts, trajectories = self.collect_trajectories(ts)
        returns, _ = jax.lax.scan(
            undis_ret,
            jnp.zeros_like(trajectories.reward),
            (trajectories.reward, trajectories.done),
        )
        cvar = conditional_value_at_risk(returns[-1, :], self.alpha)

        last_val = self.critic.apply(ts.critic_ts.params, ts.last_obs)
        last_val = jnp.where(ts.last_done, 0, last_val)
        advantages, targets = self.calculate_gae(trajectories, last_val)

        def update_epoch(ts, unused):
            rng, minibatch_rng = jax.random.split(ts.rng)
            ts = ts.replace(rng=rng)
            batch = AdvantageMinibatch(trajectories, advantages, targets)
            minibatches = self.shuffle_and_split(batch, minibatch_rng)
            ts, _ = jax.lax.scan(
                lambda ts, mbs: (self.update(ts, mbs), None),
                ts,
                minibatches,
            )
            return ts, None

        new_ts, _ = jax.lax.scan(update_epoch, ts, None, self.num_epochs)

        new_ts, new_trajectories = self.collect_trajectories(new_ts)
        new_returns, _ = jax.lax.scan(
            undis_ret,
            jnp.zeros_like(new_trajectories.reward),
            (new_trajectories.reward, new_trajectories.done),
        )
        new_cvar = conditional_value_at_risk(new_returns[-1, :], self.alpha)

        next_ts = jax.lax.cond(
            new_cvar >= (1 - self.threshold) * cvar,
            lambda: new_ts,
            lambda: ts,
        )

        return next_ts
