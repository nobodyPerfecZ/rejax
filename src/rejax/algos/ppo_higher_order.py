import chex
import gymnax
import jax
import numpy as np
from flax import linen as nn
from flax import struct
from jax import numpy as jnp

from rejax.algos.ppo import PPO, Trajectory
from rejax.networks import DiscretePolicy, GaussianPolicy, VHigherOrderNetwork
from rejax.statistics import kurtosis, skewness


class AdvantageMinibatchHigherOrder(struct.PyTreeNode):
    trajectories: Trajectory
    advantages: chex.Array
    value_targets: chex.Array
    higher_order_targets: chex.Array


class PPOKurt(PPO):
    kurt_coef: chex.Scalar = struct.field(pytree_node=True, default=1e-4)

    @classmethod
    def create_agent(cls, config, env, env_params):
        action_space = env.action_space(env_params)
        discrete = isinstance(action_space, gymnax.environments.spaces.Discrete)

        agent_kwargs = config.pop("agent_kwargs", {})
        activation = agent_kwargs.pop("activation", "swish")
        activation = getattr(nn, activation)

        hidden_layer_sizes = agent_kwargs.pop("hidden_layer_sizes", (64, 64))
        agent_kwargs["hidden_layer_sizes"] = tuple(hidden_layer_sizes)

        if discrete:
            actor = DiscretePolicy(
                action_dim=action_space.n,
                activation=activation,
                **agent_kwargs,
            )
        else:
            actor = GaussianPolicy(
                action_dim=np.prod(action_space.shape),
                action_range=(action_space.low, action_space.high),
                activation=activation,
                **agent_kwargs,
            )

        critic = VHigherOrderNetwork(
            activation=activation,
            **agent_kwargs,
        )
        return {"actor": actor, "critic": critic}

    def train_iteration(self, ts):
        ts, trajectories = self.collect_trajectories(ts)

        last_val = self.critic.apply(ts.critic_ts.params, ts.last_obs)
        last_val = jnp.where(
            ts.last_done[:, jnp.newaxis], jnp.zeros_like(last_val), last_val
        )
        advantages, value_targets, kurtosis_targets = self.calculate_gae(
            trajectories, last_val
        )

        def update_epoch(ts, unused):
            rng, minibatch_rng = jax.random.split(ts.rng)
            ts = ts.replace(rng=rng)
            batch = AdvantageMinibatchHigherOrder(
                trajectories, advantages, value_targets, kurtosis_targets
            )
            minibatches = self.shuffle_and_split(batch, minibatch_rng)
            ts, _ = jax.lax.scan(
                lambda ts, mbs: (self.update(ts, mbs), None),
                ts,
                minibatches,
            )
            return ts, None

        ts, _ = jax.lax.scan(update_epoch, ts, None, self.num_epochs)
        return ts

    def calculate_gae(self, trajectories, last_val):
        def get_advantages(runner_state, transition):
            advantage, next_value = runner_state
            next_value_, _ = jnp.split(next_value, 2, axis=-1)
            value, _ = jnp.split(transition.value, 2, axis=-1)
            delta = (
                transition.reward.squeeze()  # For gymnax envs that return shape (1, )
                + self.gamma * next_value_.squeeze() * (1 - transition.done)
                - value.squeeze()
            )
            advantage = (
                delta + self.gamma * self.gae_lambda * (1 - transition.done) * advantage
            )
            return (advantage, transition.value), advantage

        last_vals, _ = jnp.split(last_val, 2, axis=-1)
        values, kurtosises = jnp.split(trajectories.value, 2, axis=-1)
        _, advantages = jax.lax.scan(
            get_advantages,
            (jnp.zeros_like(last_vals).squeeze(), last_val),
            trajectories,
            reverse=True,
        )
        return (
            advantages + self.kurt_coef * -kurtosises.squeeze(),
            advantages + values.squeeze(),
            jnp.ones_like(advantages) * kurtosis(values, axis=-1),
        )

    def update_critic(self, ts, batch):
        def critic_loss_fn(params):
            old_values, old_kurtosis = jnp.split(batch.trajectories.value, 2, axis=-1)
            value_targets, kurtosis_targets = (
                batch.value_targets,
                batch.higher_order_targets,
            )

            value = self.critic.apply(params, batch.trajectories.obs)
            values, kurtosis = jnp.split(value, 2, axis=-1)

            # Compute clipped value loss
            value_pred_clipped = old_values + (values - old_values).clip(
                -self.clip_eps, self.clip_eps
            )
            value_losses = jnp.square(values - value_targets)
            value_losses_clipped = jnp.square(value_pred_clipped - value_targets)
            value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

            # Compute clipped kurtosis loss
            kurtosis_pred_clipped = old_kurtosis + (kurtosis - old_kurtosis).clip(
                -self.clip_eps, self.clip_eps
            )
            kurtosis_losses = jnp.square(kurtosis - kurtosis_targets)
            kurtosis_losses_clipped = jnp.square(
                kurtosis_pred_clipped - kurtosis_targets
            )
            kurtosis_loss = (
                0.5 * jnp.maximum(kurtosis_losses, kurtosis_losses_clipped).mean()
            )

            return self.vf_coef * value_loss + self.kurt_coef * kurtosis_loss

        grads = jax.grad(critic_loss_fn)(ts.critic_ts.params)
        return ts.replace(critic_ts=ts.critic_ts.apply_gradients(grads=grads))


class PPOSkew(PPO):
    skew_coef: chex.Scalar = struct.field(pytree_node=True, default=1e-3)

    @classmethod
    def create_agent(cls, config, env, env_params):
        action_space = env.action_space(env_params)
        discrete = isinstance(action_space, gymnax.environments.spaces.Discrete)

        agent_kwargs = config.pop("agent_kwargs", {})
        activation = agent_kwargs.pop("activation", "swish")
        activation = getattr(nn, activation)

        hidden_layer_sizes = agent_kwargs.pop("hidden_layer_sizes", (64, 64))
        agent_kwargs["hidden_layer_sizes"] = tuple(hidden_layer_sizes)

        if discrete:
            actor = DiscretePolicy(
                action_dim=action_space.n,
                activation=activation,
                **agent_kwargs,
            )
        else:
            actor = GaussianPolicy(
                action_dim=np.prod(action_space.shape),
                action_range=(action_space.low, action_space.high),
                activation=activation,
                **agent_kwargs,
            )

        critic = VHigherOrderNetwork(
            activation=activation,
            **agent_kwargs,
        )
        return {"actor": actor, "critic": critic}

    def train_iteration(self, ts):
        ts, trajectories = self.collect_trajectories(ts)

        last_val = self.critic.apply(ts.critic_ts.params, ts.last_obs)
        last_val = jnp.where(
            ts.last_done[:, jnp.newaxis], jnp.zeros_like(last_val), last_val
        )
        advantages, value_targets, skewness_targets = self.calculate_gae(
            trajectories, last_val
        )

        def update_epoch(ts, unused):
            rng, minibatch_rng = jax.random.split(ts.rng)
            ts = ts.replace(rng=rng)
            batch = AdvantageMinibatchHigherOrder(
                trajectories, advantages, value_targets, skewness_targets
            )
            minibatches = self.shuffle_and_split(batch, minibatch_rng)
            ts, _ = jax.lax.scan(
                lambda ts, mbs: (self.update(ts, mbs), None),
                ts,
                minibatches,
            )
            return ts, None

        ts, _ = jax.lax.scan(update_epoch, ts, None, self.num_epochs)
        return ts

    def calculate_gae(self, trajectories, last_val):
        def get_advantages(runner_state, transition):
            advantage, next_value = runner_state
            next_value_, _ = jnp.split(next_value, 2, axis=-1)
            value, _ = jnp.split(transition.value, 2, axis=-1)
            delta = (
                transition.reward.squeeze()  # For gymnax envs that return shape (1, )
                + self.gamma * next_value_.squeeze() * (1 - transition.done)
                - value.squeeze()
            )
            advantage = (
                delta + self.gamma * self.gae_lambda * (1 - transition.done) * advantage
            )
            return (advantage, transition.value), advantage

        last_vals, _ = jnp.split(last_val, 2, axis=-1)
        values, skewnesses = jnp.split(trajectories.value, 2, axis=-1)
        _, advantages = jax.lax.scan(
            get_advantages,
            (jnp.zeros_like(last_vals).squeeze(), last_val),
            trajectories,
            reverse=True,
        )
        return (
            advantages + self.skew_coef * -skewnesses.squeeze(),
            advantages + values.squeeze(),
            jnp.ones_like(advantages) * skewness(values, axis=-1),
        )

    def update_critic(self, ts, batch):
        def critic_loss_fn(params):
            old_values, old_skewness = jnp.split(batch.trajectories.value, 2, axis=-1)
            value_targets, skewness_targets = (
                batch.value_targets,
                batch.higher_order_targets,
            )

            value = self.critic.apply(params, batch.trajectories.obs)
            values, skewness = jnp.split(value, 2, axis=-1)

            # Compute clipped value loss
            value_pred_clipped = old_values + (values - old_values).clip(
                -self.clip_eps, self.clip_eps
            )
            value_losses = jnp.square(values - value_targets)
            value_losses_clipped = jnp.square(value_pred_clipped - value_targets)
            value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

            # Compute clipped skewness loss
            skewness_pred_clipped = old_skewness + (skewness - old_skewness).clip(
                -self.clip_eps, self.clip_eps
            )
            skewness_losses = jnp.square(skewness - skewness_targets)
            skewness_losses_clipped = jnp.square(
                skewness_pred_clipped - skewness_targets
            )
            skewness_loss = (
                0.5 * jnp.maximum(skewness_losses, skewness_losses_clipped).mean()
            )

            return self.vf_coef * value_loss + self.skew_coef * skewness_loss

        grads = jax.grad(critic_loss_fn)(ts.critic_ts.params)
        return ts.replace(critic_ts=ts.critic_ts.apply_gradients(grads=grads))
