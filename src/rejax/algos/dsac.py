import chex
import flax.linen as nn
import gymnax
import jax
import numpy as np
from flax import linen as nn
from flax import struct
from jax import numpy as jnp

from rejax.distortion import (
    risk_measure_cpw,
    risk_measure_cvar,
    risk_measure_neutral,
    risk_measure_wang,
)
from rejax.networks import (
    DiscretePolicy,
    DiscreteQQuantileNetwork,
    QQuantileNetwork,
    SquashedGaussianPolicy,
)
from rejax.statistics import kurtosis, skewness

from .sac import SAC


class DSAC(SAC):
    beta: chex.Scalar = struct.field(pytree_node=True, default=0.0)
    risk_fn: str = struct.field(pytree_node=False, default="neutral")

    @classmethod
    def create_agent(cls, config, env, env_params):
        agent_kwargs = config.pop("agent_kwargs", {})
        activation = agent_kwargs.pop("activation", "relu")
        activation = getattr(nn, activation)
        layers = agent_kwargs.pop("hidden_layer_sizes", (64, 64))
        agent_kwargs["hidden_layer_sizes"] = tuple(layers)
        num_quantiles = agent_kwargs.pop("num_quantiles", 200)

        action_space = env.action_space(env_params)
        if isinstance(action_space, gymnax.environments.spaces.Discrete):
            actor = DiscretePolicy(
                action_dim=action_space.n,
                activation=activation,
                **agent_kwargs,
            )
            critic = DiscreteQQuantileNetwork(
                action_dim=action_space.n,
                activation=activation,
                num_quantiles=num_quantiles,
                **agent_kwargs,
            )
        else:
            actor = SquashedGaussianPolicy(
                action_dim=np.prod(action_space.shape),
                action_range=(action_space.low, action_space.high),
                activation=activation,
                log_std_range=(-10, 2),
                **agent_kwargs,
            )
            critic = QQuantileNetwork(
                activation=activation,
                num_quantiles=num_quantiles,
                **agent_kwargs,
            )
        return {"actor": actor, "critic": critic}

    def udpate_actor(self, ts, mb):
        rng, action_rng = jax.random.split(ts.rng)
        ts = ts.replace(rng=rng)
        alpha = jnp.exp(ts.alpha_ts.params["log_alpha"])

        def actor_loss_fn(params):
            if self.discrete:
                logprob = jnp.log(
                    self.actor.apply(params, mb.obs, method="_action_dist").probs
                )
                qs = self.vmap_critic(ts.critic_ts.params, mb.obs)
                qs = self.risk_measure(qs)
                loss_pi = alpha * logprob - qs.min(axis=0)
                loss_pi = jnp.sum(jnp.exp(logprob) * loss_pi, axis=1)
            else:
                action, logprob = self.actor.apply(
                    params, mb.obs, action_rng, method="action_log_prob"
                )
                qs = self.vmap_critic(ts.critic_ts.params, mb.obs, action)
                qs = self.risk_measure(qs)
                loss_pi = alpha * logprob - qs.min(axis=0)
            return loss_pi.mean(), logprob

        grads, logprob = jax.grad(actor_loss_fn, has_aux=True)(ts.actor_ts.params)
        ts = ts.replace(actor_ts=ts.actor_ts.apply_gradients(grads=grads))
        return ts, logprob

    def update_critic(self, ts, mb):
        rng, action_rng = jax.random.split(ts.rng)
        ts = ts.replace(rng=rng)
        alpha = jnp.exp(ts.alpha_ts.params["log_alpha"])

        def critic_loss_fn(params):
            # Calculate target without gradient wrt `params`
            if self.discrete:
                action_dist = self.actor.apply(
                    ts.actor_ts.params, mb.next_obs, method="_action_dist"
                )
                logprob = jnp.log(action_dist.probs)
                logprob = jnp.expand_dims(logprob, axis=-1)
                qs = self.vmap_critic(ts.critic_target_params, mb.next_obs)
                q_target = jnp.min(qs, axis=0) - alpha * logprob
                q_target = jnp.sum(jnp.exp(logprob) * q_target, axis=1)
                qs = jax.vmap(
                    lambda *args: self.critic.apply(*args, method="take"),
                    in_axes=(0, None, None),
                )(params, mb.obs, mb.action)
            else:
                action, logprob = self.actor.apply(
                    ts.actor_ts.params,
                    mb.next_obs,
                    action_rng,
                    method="action_log_prob",
                )
                logprob = jnp.expand_dims(logprob, axis=-1)
                qs = self.vmap_critic(ts.critic_target_params, mb.next_obs, action)
                q_target = jnp.min(qs, axis=0) - alpha * logprob
                qs = self.vmap_critic(params, mb.obs, mb.action)
            target = (
                jnp.expand_dims(mb.reward, axis=-1)
                + self.gamma * (1 - jnp.expand_dims(mb.done, axis=-1)) * q_target
            )
            losses = jax.vmap(lambda q: self.quantile_mse_loss(q, target))(qs)
            return losses.sum(axis=0).mean()

        grads = jax.grad(critic_loss_fn)(ts.critic_ts.params)
        ts = ts.replace(critic_ts=ts.critic_ts.apply_gradients(grads=grads))
        return ts

    def update_alpha(self, ts, logprob):
        def alpha_loss_fn(params, logprob):
            alpha = jnp.exp(params["log_alpha"])
            loss_alpha = -alpha * (logprob + self.target_entropy)
            if self.discrete:
                loss_alpha = jnp.sum(jnp.exp(logprob) * loss_alpha, axis=1)
            return loss_alpha.mean()

        grads = jax.grad(alpha_loss_fn)(ts.alpha_ts.params, logprob)
        ts = ts.replace(alpha_ts=ts.alpha_ts.apply_gradients(grads=grads))
        return ts

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

    def quantile_mse_loss(self, predictions, targets):
        _, num_quantiles = predictions.shape
        tau_hat = jnp.linspace(
            start=1 / (2 * num_quantiles),
            stop=1 - 1 / (2 * num_quantiles),
            num=num_quantiles,
            endpoint=True,
        )
        predictions = jnp.expand_dims(predictions, axis=-1)
        targets = jnp.expand_dims(targets, axis=1)
        tau_hat = jnp.expand_dims(tau_hat, axis=[0, -1])

        delta = targets - predictions

        # L2 LOSS
        l2_loss = 0.5 * jnp.square(delta)

        # QUANTILE L2 LOSS
        loss = jnp.abs(jnp.where(delta < 0, (tau_hat - 1), tau_hat)) * l2_loss

        return loss


class DSACVar(DSAC):
    std_coef: chex.Scalar = struct.field(pytree_node=True, default=1e-3)

    def udpate_actor(self, ts, mb):
        rng, action_rng = jax.random.split(ts.rng)
        ts = ts.replace(rng=rng)
        alpha = jnp.exp(ts.alpha_ts.params["log_alpha"])

        def actor_loss_fn(params):
            if self.discrete:
                logprob = jnp.log(
                    self.actor.apply(params, mb.obs, method="_action_dist").probs
                )
                qs = self.vmap_critic(ts.critic_ts.params, mb.obs)
                qs = self.risk_measure(qs)
                loss_pi = (
                    self.std_coef * -qs.min(axis=0).std(axis=-1, keepdims=True)
                    + alpha * logprob
                    - qs.min(axis=0)
                )
                loss_pi = jnp.sum(jnp.exp(logprob) * loss_pi, axis=1)
            else:
                action, logprob = self.actor.apply(
                    params, mb.obs, action_rng, method="action_log_prob"
                )
                qs = self.vmap_critic(ts.critic_ts.params, mb.obs, action)
                qs = self.risk_measure(qs)
                loss_pi = (
                    self.std_coef * -qs.min(axis=0).std(axis=-1, keepdims=True)
                    + alpha * logprob
                    - qs.min(axis=0)
                )
            return loss_pi.mean(), logprob

        grads, logprob = jax.grad(actor_loss_fn, has_aux=True)(ts.actor_ts.params)
        ts = ts.replace(actor_ts=ts.actor_ts.apply_gradients(grads=grads))
        return ts, logprob


class DSACKurt(DSAC):
    kurt_coef: chex.Scalar = struct.field(pytree_node=True, default=1e-4)

    def udpate_actor(self, ts, mb):
        rng, action_rng = jax.random.split(ts.rng)
        ts = ts.replace(rng=rng)
        alpha = jnp.exp(ts.alpha_ts.params["log_alpha"])

        def actor_loss_fn(params):
            if self.discrete:
                logprob = jnp.log(
                    self.actor.apply(params, mb.obs, method="_action_dist").probs
                )
                qs = self.vmap_critic(ts.critic_ts.params, mb.obs)
                qs = self.risk_measure(qs)
                loss_pi = (
                    self.kurt_coef * -kurtosis(qs.min(axis=0), axis=-1, keepdims=True)
                    + alpha * logprob
                    - qs.min(axis=0)
                )
                loss_pi = jnp.sum(jnp.exp(logprob) * loss_pi, axis=1)
            else:
                action, logprob = self.actor.apply(
                    params, mb.obs, action_rng, method="action_log_prob"
                )
                qs = self.vmap_critic(ts.critic_ts.params, mb.obs, action)
                qs = self.risk_measure(qs)
                loss_pi = (
                    self.kurt_coef * -kurtosis(qs.min(axis=0), axis=-1, keepdims=True)
                    + alpha * logprob
                    - qs.min(axis=0)
                )
            return loss_pi.mean(), logprob

        grads, logprob = jax.grad(actor_loss_fn, has_aux=True)(ts.actor_ts.params)
        ts = ts.replace(actor_ts=ts.actor_ts.apply_gradients(grads=grads))
        return ts, logprob


class DSACSkew(DSAC):
    skew_coef: chex.Scalar = struct.field(pytree_node=True, default=1e-3)

    def udpate_actor(self, ts, mb):
        rng, action_rng = jax.random.split(ts.rng)
        ts = ts.replace(rng=rng)
        alpha = jnp.exp(ts.alpha_ts.params["log_alpha"])

        def actor_loss_fn(params):
            if self.discrete:
                logprob = jnp.log(
                    self.actor.apply(params, mb.obs, method="_action_dist").probs
                )
                qs = self.vmap_critic(ts.critic_ts.params, mb.obs)
                qs = self.risk_measure(qs)
                loss_pi = (
                    self.skew_coef * -skewness(qs.min(axis=0), axis=-1, keepdims=True)
                    + alpha * logprob
                    - qs.min(axis=0)
                )
                loss_pi = jnp.sum(jnp.exp(logprob) * loss_pi, axis=1)
            else:
                action, logprob = self.actor.apply(
                    params, mb.obs, action_rng, method="action_log_prob"
                )
                qs = self.vmap_critic(ts.critic_ts.params, mb.obs, action)
                qs = self.risk_measure(qs)
                loss_pi = (
                    self.skew_coef * -skewness(qs.min(axis=0), axis=-1, keepdims=True)
                    + alpha * logprob
                    - qs.min(axis=0)
                )
            return loss_pi.mean(), logprob

        grads, logprob = jax.grad(actor_loss_fn, has_aux=True)(ts.actor_ts.params)
        ts = ts.replace(actor_ts=ts.actor_ts.apply_gradients(grads=grads))
        return ts, logprob
