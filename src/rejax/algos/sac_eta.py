import chex
import jax
import optax
from flax import struct
from flax.core.frozen_dict import FrozenDict
from flax.training.train_state import TrainState
from jax import numpy as jnp

from rejax.algos.algorithm import register_init
from rejax.statistics import explained_variance

from .sac import SAC


class SACEta(SAC):
    target_risk_ratio: chex.Scalar = struct.field(pytree_node=True, default=0.1)

    @register_init
    def initialize_network_params(self, rng):
        obs_ph = jnp.empty((1, *self.obs_space.shape))

        rng, rng_actor, rng_critic = jax.random.split(rng, 3)
        actor_params = self.actor.init(rng_actor, obs_ph, rng_actor)

        rng_critic = jax.random.split(rng_critic, self.num_critics)
        if self.discrete:
            critic_params = jax.vmap(self.critic.init, in_axes=(0, None))(
                rng_critic, obs_ph
            )
        else:
            act_ph = jnp.empty((1, *self.env.action_space(self.env_params).shape))
            critic_params = jax.vmap(self.critic.init, in_axes=(0, None, None))(
                rng_critic, obs_ph, act_ph
            )

        tx = optax.chain(
            optax.clip_by_global_norm(self.max_grad_norm),
            optax.adam(learning_rate=self.learning_rate),
        )
        actor_ts = TrainState.create(apply_fn=(), params=actor_params, tx=tx)
        critic_ts = TrainState.create(apply_fn=(), params=critic_params, tx=tx)
        critic_target_params = critic_params

        if self.target_entropy is None:
            self.target_entropy = -self.env.action_space(self.env_params).shape[0]  # ty:ignore[invalid-assignment]

        alpha_params = FrozenDict({"log_alpha": jnp.array(0.0)})
        alpha_ts = TrainState.create(apply_fn=(), params=alpha_params, tx=tx)

        eta_params = FrozenDict({"log_eta": jnp.array(0.0)})
        eta_ts = TrainState.create(apply_fn=(), params=eta_params, tx=tx)

        return {
            "actor_ts": actor_ts,
            "critic_ts": critic_ts,
            "critic_target_params": critic_target_params,
            "alpha_ts": alpha_ts,
            "eta_ts": eta_ts,
        }

    def train_iteration(self, ts):
        # Collect transitions
        old_global_step = ts.global_step

        ts, batch = self.collect_transitions(ts)
        ts = ts.replace(replay_buffer=ts.replay_buffer.extend(batch))

        def update_iteration(ts):
            # Sample minibatch
            rng, rng_sample = jax.random.split(ts.rng)
            ts = ts.replace(rng=rng)
            minibatch = ts.replay_buffer.sample(self.batch_size, rng_sample)
            if self.normalize_observations:
                minibatch = minibatch._replace(
                    obs=self.normalize_obs(ts.obs_rms_state, minibatch.obs),
                    next_obs=self.normalize_obs(ts.obs_rms_state, minibatch.next_obs),
                )
            if self.normalize_rewards:
                minibatch = minibatch._replace(
                    reward=self.normalize_rew(ts.rew_rms_state, minibatch.reward)
                )

            # Update networks
            return self.update(ts, minibatch)

        def do_updates(ts):
            return jax.lax.scan(
                lambda ts, _: update_iteration(ts),
                ts,
                None,
                length=self.num_epochs,
            )

        start_training = ts.global_step > self.fill_buffer

        _, mock_metrics = jax.eval_shape(do_updates, ts)
        mock_metrics = jax.tree.map(lambda x: jnp.zeros(x.shape, x.dtype), mock_metrics)

        ts, loss_metrics = jax.lax.cond(
            start_training,
            lambda: do_updates(ts),
            lambda: (ts, mock_metrics),
        )

        # Update target network
        if self.target_update_freq == 1:
            target_params = self.polyak_update(
                ts.critic_ts.params, ts.critic_target_params
            )
        else:
            update_target_params = (
                ts.global_step % self.target_update_freq
                <= old_global_step % self.target_update_freq
            )
            target_params = jax.tree.map(
                lambda q, qt: jax.lax.select(update_target_params, q, qt),
                self.polyak_update(ts.critic_ts.params, ts.critic_target_params),
                ts.critic_target_params,
            )

        return ts.replace(critic_target_params=target_params), jax.tree.map(
            lambda x: jnp.mean(x, axis=0), loss_metrics
        )

    def update_critic(self, ts, mb):
        rng, action_rng = jax.random.split(ts.rng)
        ts = ts.replace(rng=rng)
        alpha = jnp.exp(ts.alpha_ts.params["log_alpha"])
        eta = jnp.exp(ts.eta_ts.params["log_eta"])

        def critic_loss_fn(params):
            # Calculate target without gradient wrt `params`
            if self.discrete:
                action_dist = self.actor.apply(
                    ts.actor_ts.params, mb.next_obs, method="_action_dist"
                )
                log_prob = jnp.log(action_dist.probs)  # ty:ignore[unresolved-attribute]
                qs = self.vmap_critic(ts.critic_target_params, mb.next_obs)
                q_target = jnp.min(qs, axis=0) - alpha * log_prob
                q_target = jnp.sum(jnp.exp(log_prob) * q_target, axis=1)
                qs = jax.vmap(
                    lambda *args: self.critic.apply(*args, method="take"),
                    in_axes=(0, None, None),
                )(params, mb.obs, mb.action)
            else:
                action, log_prob = self.actor.apply(
                    ts.actor_ts.params,
                    mb.next_obs,
                    action_rng,
                    method="action_log_prob",
                )
                qs = self.vmap_critic(ts.critic_target_params, mb.next_obs, action)
                q_target = jnp.min(qs, axis=0) - alpha * log_prob  # ty:ignore[unsupported-operator]
                qs = self.vmap_critic(params, mb.obs, mb.action)

            q_diff = jnp.abs(qs[0] - qs[1])
            target = mb.reward + self.gamma * (1 - mb.done) * q_target
            # TODO: Sum cannot be used ...
            v_entropic = (1.0 / eta) * jnp.log(jnp.exp(eta * target))
            losses = jax.vmap(lambda q: optax.l2_loss(q, v_entropic))(qs)
            return losses.sum(axis=0).mean(), (
                target,
                {
                    "critic/log_prob": log_prob.mean(),  # ty:ignore[unresolved-attribute]
                    "critic/qs": qs.mean(),
                    "critic/q_target": q_target.mean(),
                    "critic/q_diff": q_diff.mean(),
                    "critic/explained_variance": explained_variance(qs[0], target),
                },
            )

        (loss, (targets, aux)), grads = jax.value_and_grad(
            critic_loss_fn, has_aux=True
        )(ts.critic_ts.params)
        return ts.replace(critic_ts=ts.critic_ts.apply_gradients(grads=grads)), (
            targets,
            {
                "critic/total_loss": loss,
                "critic/grad_norm": optax.global_norm(grads),
                "critic/param_norm": optax.global_norm(ts.critic_ts.params),
                "critic/momentum_norm": optax.global_norm(
                    ts.critic_ts.opt_state[1][0].mu
                ),
                "critic/variance_norm": optax.global_norm(
                    ts.critic_ts.opt_state[1][0].nu
                ),
                **aux,
            },
        )

    def update_eta(self, ts, target):
        def eta_loss_fn(params, target):
            eta = jnp.exp(params["log_eta"])
            target_bar = jnp.mean(target)
            loss_eta = eta * (self.target_risk_ratio - (target - target_bar))
            return loss_eta.mean(), {"eta/value": eta}

        (loss, aux), grads = jax.value_and_grad(eta_loss_fn, has_aux=True)(
            ts.eta_ts.params, target
        )
        return ts.replace(eta_ts=ts.eta_ts.apply_gradients(grads=grads)), {
            "eta/total_loss": loss,
            **aux,
        }

    def update(self, ts, mb):
        ts, (log_prob, actor_loss_metrics) = self.udpate_actor(ts, mb)
        ts, (target, critic_loss_metrics) = self.update_critic(ts, mb)
        ts, alpha_loss_metrics = self.update_alpha(
            ts,
            log_prob,
        )
        ts, eta_loss_metrics = self.update_eta(ts, target)
        return ts, {
            **actor_loss_metrics,
            **critic_loss_metrics,
            **alpha_loss_metrics,
            **eta_loss_metrics,
        }
