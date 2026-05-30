import chex
import flax.linen as nn
import gymnax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct

from rejax.distortion import (
    risk_measure_cpw,
    risk_measure_cvar,
    risk_measure_neutral,
    risk_measure_wang,
)
from rejax.networks import DiscretePolicy, GaussianPolicy, VQuantileNetwork
from rejax.statistics import explained_variance, kurtosis, skewness

from .ppo import PPO, AdvantageMinibatch


class DPPO(PPO):
    beta: chex.Scalar = struct.field(pytree_node=True, default=0.0)
    risk_fn: str = struct.field(pytree_node=False, default="neutral")
    sr_lambda: chex.Scalar = struct.field(pytree_node=True, default=0.95)

    @classmethod
    def create_agent(cls, config, env, env_params):
        action_space = env.action_space(env_params)
        discrete = isinstance(action_space, gymnax.environments.spaces.Discrete)  # ty:ignore[possibly-missing-submodule]

        agent_kwargs = config.pop("agent_kwargs", {})
        activation = agent_kwargs.pop("activation", "swish")
        activation = getattr(nn, activation)

        hidden_layer_sizes = agent_kwargs.pop("hidden_layer_sizes", (64, 64))
        agent_kwargs["hidden_layer_sizes"] = tuple(hidden_layer_sizes)

        num_quantiles = agent_kwargs.pop("num_quantiles", 200)

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

        critic = VQuantileNetwork(
            activation=activation,
            num_quantiles=num_quantiles,
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
        ts, advantages, targets = self.calculate_gae(ts, trajectories, last_val)

        def update_epoch(ts, unused):
            rng, minibatch_rng = jax.random.split(ts.rng)
            ts = ts.replace(rng=rng)
            batch = AdvantageMinibatch(trajectories, advantages, targets)
            minibatches = self.shuffle_and_split(batch, minibatch_rng)
            ts, loss_metrics = jax.lax.scan(
                lambda ts, mbs: self.update(ts, mbs),
                ts,
                minibatches,
            )
            return ts, jax.tree.map(lambda x: jnp.mean(x, axis=0), loss_metrics)

        ts, loss_metrics = jax.lax.scan(update_epoch, ts, None, self.num_epochs)
        return ts, jax.tree.map(lambda x: jnp.mean(x, axis=0), loss_metrics)

    def calculate_gae(self, ts, trajectories, last_val):  # ty:ignore[invalid-method-override]
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
            value = self.critic.apply(params, batch.trajectories.obs)
            value_pred_clipped = batch.trajectories.value + (
                value - batch.trajectories.value
            ).clip(-self.clip_eps, self.clip_eps)
            targets = batch.targets
            value_loss = self.quantile_clipped_mse_loss(
                value, batch.trajectories.value, targets
            )
            return self.vf_coef * value_loss.mean(), (
                value,
                {
                    "critic/value": value.mean(),  # ty:ignore[unresolved-attribute]
                    "critic/value_clipped": value_pred_clipped.mean(),
                    "critic/value_loss": value_loss.mean(),
                },
            )

        (loss, (value, aux)), grads = jax.value_and_grad(critic_loss_fn, has_aux=True)(
            ts.critic_ts.params
        )
        return ts.replace(critic_ts=ts.critic_ts.apply_gradients(grads=grads)), {
            "critic/total_loss": loss,
            "critic/explained_variance": explained_variance(batch.targets, value),
            "critic/grad_norm": optax.global_norm(grads),
            "critic/param_norm": optax.global_norm(ts.critic_ts.params),
            "critic/momentum_norm": optax.global_norm(ts.critic_ts.opt_state[1][0].mu),
            "critic/variance_norm": optax.global_norm(ts.critic_ts.opt_state[1][0].nu),
            **aux,
        }

    def risk_measure(self, quantiles):
        if self.risk_fn == "neutral":
            return risk_measure_neutral(quantiles, self.beta)
        elif self.risk_fn == "cvar":
            return risk_measure_cvar(quantiles, self.beta)
        elif self.risk_fn == "cpw":
            return risk_measure_cpw(quantiles, self.beta)
        elif self.risk_fn == "wang":
            return risk_measure_wang(quantiles, self.beta)
        else:
            raise ValueError("Invalid risk measure")

    def quantile_clipped_mse_loss(self, predictions, predictions_collected, targets):
        _, num_quantiles = predictions.shape
        tau_hat = jnp.linspace(
            start=1 / (2 * num_quantiles),
            stop=1 - 1 / (2 * num_quantiles),
            num=num_quantiles,
            endpoint=True,
        )
        predictions = jnp.expand_dims(predictions, axis=-1)
        predictions_collected = jnp.expand_dims(predictions_collected, axis=-1)
        targets = jnp.expand_dims(targets, axis=1)
        tau_hat = jnp.expand_dims(tau_hat, axis=[0, -1])

        delta = targets - predictions
        delta1 = targets - (
            predictions_collected
            + (predictions - predictions_collected).clip(-self.clip_eps, +self.clip_eps)
        )
        loss1 = jnp.square(delta1)
        delta2 = targets - predictions
        loss2 = jnp.square(delta2)

        # CLIPPED L2 LOSS
        clipped_l2_loss = 0.5 * jnp.maximum(loss1, loss2)

        # QUANTILE CLIPPED L2 LOSS
        loss = jnp.abs(jnp.where(delta < 0, (tau_hat - 1), tau_hat)) * clipped_l2_loss

        return loss.mean()


class DPPOVar(DPPO):
    std_coef: chex.Scalar = struct.field(pytree_node=True, default=1e-3)

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
        advantages = advantages + self.std_coef * -advantages.std(
            axis=-1, keepdims=True
        )
        return ts, advantages, targets


class DPPOKurt(DPPO):
    kurt_coef: chex.Scalar = struct.field(pytree_node=True, default=1e-4)

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
        advantages = advantages + self.kurt_coef * -kurtosis(
            advantages,
            axis=-1,
            keepdims=True,
        )  # ty:ignore[unsupported-operator]
        return ts, advantages, targets


class DPPOSkew(DPPO):
    skew_coef: chex.Scalar = struct.field(pytree_node=True, default=1e-3)

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
        advantages = advantages + self.skew_coef * -skewness(
            advantages, axis=-1, keepdims=True
        )  # ty:ignore[unsupported-operator]

        return ts, advantages, targets
