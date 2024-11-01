import chex
import flax.linen as nn
import gymnax
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from rejax.networks import DiscretePolicy, GaussianPolicy, VQuantileNetwork

from .ppo import PPO, AdvantageMinibatch, Trajectory


class DPPO(PPO):
    sr_lambda: chex.Scalar = struct.field(pytree_node=True, default=0.95)
    alpha: chex.Scalar = struct.field(pytree_node=True, default=0.0)

    @classmethod
    def create_agent(cls, config, env, env_params):
        action_space = env.action_space(env_params)
        discrete = isinstance(action_space, gymnax.environments.spaces.Discrete)

        agent_kwargs = config.pop("agent_kwargs", {})
        activation = agent_kwargs.pop("activation", "swish")
        agent_kwargs["activation"] = getattr(nn, activation)

        hidden_layer_sizes = agent_kwargs.pop("hidden_layer_sizes", (64, 64))
        agent_kwargs["hidden_layer_sizes"] = tuple(hidden_layer_sizes)

        num_quantiles = agent_kwargs.pop("num_quantiles", 200)

        if discrete:
            actor = DiscretePolicy(action_space.n, **agent_kwargs)
        else:
            actor = GaussianPolicy(
                np.prod(action_space.shape),
                (action_space.low, action_space.high),
                **agent_kwargs,
            )

        critic = VQuantileNetwork(**agent_kwargs, num_quantiles=num_quantiles)
        return {"actor": actor, "critic": critic}

    def train_iteration(self, ts):
        ts, trajectories = self.collect_trajectories(ts)

        last_val = self.critic.apply(ts.critic_ts.params, ts.last_obs)
        last_val = jnp.where(
            ts.last_done[:, jnp.newaxis], jnp.zeros_like(last_val), last_val
        )
        ts, advantages, targets = self.calculate_gae(ts, trajectories, last_val)

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

        ts, _ = jax.lax.scan(update_epoch, ts, None, self.num_epochs)
        return ts

    def collect_trajectories(self, ts):
        def env_step(ts, unused):
            # Get keys for sampling action and stepping environment
            rng, new_rng = jax.random.split(ts.rng)
            ts = ts.replace(rng=rng)
            rng_steps, rng_action = jax.random.split(new_rng, 2)
            rng_steps = jax.random.split(rng_steps, self.num_envs)

            # Sample action
            unclipped_action, log_prob = self.actor.apply(
                ts.actor_ts.params, ts.last_obs, rng_action, method="action_log_prob"
            )
            value = self.critic.apply(ts.critic_ts.params, ts.last_obs)

            # Clip action
            if self.discrete:
                action = unclipped_action
            else:
                low = self.env.action_space(self.env_params).low
                high = self.env.action_space(self.env_params).high
                action = jnp.clip(unclipped_action, low, high)

            # Step environment
            t = self.vmap_step(rng_steps, ts.env_state, action, self.env_params)
            next_obs, env_state, reward, done, _ = t

            if self.normalize_observations:
                rms_state, next_obs = self.update_and_normalize(ts.rms_state, next_obs)
                ts = ts.replace(rms_state=rms_state)

            # Return updated runner state and transition
            transition = Trajectory(
                ts.last_obs, unclipped_action, log_prob, reward, value, done
            )
            ts = ts.replace(
                env_state=env_state,
                last_obs=next_obs,
                last_done=done,
                global_step=ts.global_step + self.num_envs,
            )
            return ts, transition

        ts, trajectories = jax.lax.scan(env_step, ts, None, self.num_steps)
        return ts, trajectories

    def calculate_gae(self, ts, trajectories, last_val):
        def get_advantages(runner_state, transition):
            ts, gae, next_value, next_value_quants = runner_state

            done, value_quants, reward = (
                transition.done,
                transition.value,
                transition.reward.squeeze(),
            )
            value = self.risk_measure(value_quants)

            delta = reward + self.gamma * next_value * (1 - done) - value
            gae = delta + self.gamma * self.gae_lambda * (1 - done) * gae

            target_value_quants = (
                reward[:, jnp.newaxis]
                + self.gamma * (1 - done)[:, jnp.newaxis] * next_value_quants
            )

            rng, new_rng = jax.random.split(ts.rng)
            ts = ts.replace(rng=rng)
            condition = (1 - done)[:, jnp.newaxis] * (
                jax.random.uniform(new_rng, next_value_quants.shape) < self.sr_lambda
            )
            next_value_quants = jnp.where(condition, target_value_quants, value_quants)

            runner_state = (ts, gae, value, next_value_quants)
            metrics = (gae, target_value_quants)

            return runner_state, metrics

        last_val_risk = self.risk_measure(last_val)
        runner_state, metrics = jax.lax.scan(
            get_advantages,
            (ts, jnp.zeros(last_val.shape[:-1]), last_val_risk, last_val),
            trajectories,
            reverse=True,
        )
        ts = runner_state[0]
        advantages, targets = metrics
        return ts, advantages, targets

    def update_critic(self, ts, batch):
        def critic_loss_fn(params):
            predictions = self.critic.apply(params, batch.trajectories.obs)
            predictions_collected = batch.trajectories.value
            targets = batch.targets
            num_minibatches, num_quantiles = predictions.shape
            tau_hat = jnp.linspace(
                start=1 / (2 * num_quantiles),
                stop=1 - 1 / (2 * num_quantiles),
                num=num_quantiles,
                endpoint=True,
            )
            predictions = jnp.repeat(
                jnp.expand_dims(predictions, axis=0),
                num_quantiles,
                axis=0,
            )
            predictions_collected = jnp.repeat(
                jnp.expand_dims(predictions_collected, axis=0),
                num_quantiles,
                axis=0,
            )
            targets = jnp.repeat(
                jnp.expand_dims(jnp.transpose(targets), axis=-1),
                num_quantiles,
                axis=-1,
            )
            tau_hat = jnp.repeat(
                jnp.expand_dims(tau_hat, axis=0),
                num_minibatches,
                axis=0,
            )
            delta = targets - predictions

            # CLIPPED LOSS
            delta_clipped = (
                predictions_collected
                + (predictions - predictions_collected).clip(
                    -self.clip_eps, +self.clip_eps
                )
            ) - targets
            value_losses_clipped = jnp.square(delta_clipped)

            # NORMAL LOSS
            delta = predictions - targets
            value_losses = jnp.square(delta)

            # CLIPPED L2 LOSS
            clipped_l2_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped)

            # QUANTILE CLIPPED L2 LOSS
            value_loss = (
                jnp.abs(jnp.where(delta < 0, (tau_hat - 1), tau_hat)) * clipped_l2_loss
            ).mean()
            return self.vf_coef * value_loss

        grads = jax.grad(critic_loss_fn)(ts.critic_ts.params)
        return ts.replace(critic_ts=ts.critic_ts.apply_gradients(grads=grads))

    def risk_measure(self, quantiles):
        if self.alpha == 0.0:
            # Neutral Distortion
            return jnp.mean(quantiles, axis=-1)
        else:
            # CVAR Distortion
            tau = jnp.linspace(0, 1, num=quantiles.shape[-1] + 1, endpoint=True)
            distorted_tau = jnp.minimum(tau / self.alpha, 1.0)
            distortion = distorted_tau[1:] - distorted_tau[:-1]
            sorted_quantiles = jnp.sort(quantiles, axis=-1)
            return jnp.sum(distortion * sorted_quantiles, axis=-1)
