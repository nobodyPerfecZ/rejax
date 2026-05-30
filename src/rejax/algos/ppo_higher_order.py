import chex
import gymnax
import jax
import numpy as np
import optax
from flax import linen as nn
from flax import struct
from jax import numpy as jnp

from rejax.algos.ppo import PPO, Trajectory
from rejax.networks import DiscretePolicy, GaussianPolicy, VHigherOrderNetwork
from rejax.statistics import explained_variance, kurtosis, skewness


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
        discrete = isinstance(action_space, gymnax.environments.spaces.Discrete)  # ty:ignore[possibly-missing-submodule]

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
            ts.last_done[:, jnp.newaxis],
            jnp.zeros_like(last_val),  # ty:ignore[invalid-argument-type]
            last_val,
        )  # ty:ignore[no-matching-overload]
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
            ts, loss_metrics = jax.lax.scan(
                lambda ts, mbs: self.update(ts, mbs),
                ts,
                minibatches,
            )
            return ts, jax.tree.map(lambda x: jnp.mean(x, axis=0), loss_metrics)

        ts, loss_metrics = jax.lax.scan(update_epoch, ts, None, self.num_epochs)
        return ts, jax.tree.map(lambda x: jnp.mean(x, axis=0), loss_metrics)

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
            values, kurtosis = jnp.split(value, 2, axis=-1)  # ty:ignore[invalid-argument-type]

            # Compute clipped value loss
            value_pred_clipped = old_values + (values - old_values).clip(
                -self.clip_eps, self.clip_eps
            )
            value_losses = jnp.square(values - value_targets)
            value_losses_clipped = jnp.square(value_pred_clipped - value_targets)
            value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped)

            # Compute clipped kurtosis loss
            kurtosis_pred_clipped = old_kurtosis + (kurtosis - old_kurtosis).clip(
                -self.clip_eps, self.clip_eps
            )
            kurtosis_losses = jnp.square(kurtosis - kurtosis_targets)
            kurtosis_losses_clipped = jnp.square(
                kurtosis_pred_clipped - kurtosis_targets
            )
            kurtosis_loss = 0.5 * jnp.maximum(kurtosis_losses, kurtosis_losses_clipped)

            loss = (
                self.vf_coef * value_loss.mean() + self.kurt_coef * kurtosis_loss.mean()
            )

            return loss, (
                values,
                {
                    "critic/value": values.mean(),
                    "critic/value_clipped": value_pred_clipped.mean(),
                    "critic/value_loss": value_loss.mean(),
                    "critic/kurtosis": kurtosis.mean(),
                    "critic/kurtosis_clipped": kurtosis_pred_clipped.mean(),
                    "critic/kurtosis_loss": kurtosis_loss.mean(),
                },
            )

        (loss, (values, aux)), grads = jax.value_and_grad(critic_loss_fn, has_aux=True)(
            ts.critic_ts.params
        )
        ts = ts.replace(critic_ts=ts.critic_ts.apply_gradients(grads=grads))
        return ts, {
            "critic/total_loss": loss,
            "critic/explained_variance": explained_variance(
                batch.value_targets, values
            ),
            "critic/grad_norm": optax.global_norm(grads),
            "critic/param_norm": optax.global_norm(ts.critic_ts.params),
            "critic/momentum_norm": optax.global_norm(ts.critic_ts.opt_state[1][0].mu),
            "critic/variance_norm": optax.global_norm(ts.critic_ts.opt_state[1][0].nu),
            **aux,
        }


class PPOSkew(PPO):
    skew_coef: chex.Scalar = struct.field(pytree_node=True, default=1e-3)

    @classmethod
    def create_agent(cls, config, env, env_params):
        action_space = env.action_space(env_params)
        discrete = isinstance(action_space, gymnax.environments.spaces.Discrete)  # ty:ignore[possibly-missing-submodule]

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
            ts.last_done[:, jnp.newaxis],
            jnp.zeros_like(last_val),  # ty:ignore[invalid-argument-type]
            last_val,
        )  # ty:ignore[no-matching-overload]
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
            ts, loss_metrics = jax.lax.scan(
                lambda ts, mbs: self.update(ts, mbs),
                ts,
                minibatches,
            )
            return ts, jax.tree.map(lambda x: jnp.mean(x, axis=0), loss_metrics)

        ts, loss_metrics = jax.lax.scan(update_epoch, ts, None, self.num_epochs)
        return ts, jax.tree.map(lambda x: jnp.mean(x, axis=0), loss_metrics)

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
            values, skewness = jnp.split(value, 2, axis=-1)  # ty:ignore[invalid-argument-type]

            # Compute clipped value loss
            value_pred_clipped = old_values + (values - old_values).clip(
                -self.clip_eps, self.clip_eps
            )
            value_losses = jnp.square(values - value_targets)
            value_losses_clipped = jnp.square(value_pred_clipped - value_targets)
            value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped)

            # Compute clipped skewness loss
            skewness_pred_clipped = old_skewness + (skewness - old_skewness).clip(
                -self.clip_eps, self.clip_eps
            )
            skewness_losses = jnp.square(skewness - skewness_targets)
            skewness_losses_clipped = jnp.square(
                skewness_pred_clipped - skewness_targets
            )
            skewness_loss = 0.5 * jnp.maximum(skewness_losses, skewness_losses_clipped)

            loss = (
                self.vf_coef * value_loss.mean() + self.skew_coef * skewness_loss.mean()
            )

            return loss, (
                values,
                {
                    "critic/value": values.mean(),
                    "critic/value_clipped": value_pred_clipped.mean(),
                    "critic/value_loss": value_loss.mean(),
                    "critic/skewness": skewness.mean(),
                    "critic/skewness_clipped": skewness_pred_clipped.mean(),
                    "critic/skewness_loss": skewness_loss.mean(),
                },
            )

        (loss, (values, aux)), grads = jax.value_and_grad(critic_loss_fn, has_aux=True)(
            ts.critic_ts.params
        )
        ts = ts.replace(critic_ts=ts.critic_ts.apply_gradients(grads=grads))
        return ts, {
            "critic/total_loss": loss,
            "critic/explained_variance": explained_variance(
                batch.value_targets, values
            ),
            "critic/grad_norm": optax.global_norm(grads),
            "critic/param_norm": optax.global_norm(ts.critic_ts.params),
            "critic/momentum_norm": optax.global_norm(ts.critic_ts.opt_state[1][0].mu),
            "critic/variance_norm": optax.global_norm(ts.critic_ts.opt_state[1][0].nu),
            **aux,
        }
