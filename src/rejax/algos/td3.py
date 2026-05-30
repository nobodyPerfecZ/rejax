import chex
import jax
import numpy as np
import optax
from flax import linen as nn
from flax import struct
from flax.training.train_state import TrainState
from jax import numpy as jnp

from rejax.algos.algorithm import Algorithm, register_init
from rejax.algos.mixins import (
    NormalizeObservationsMixin,
    NormalizeRewardsMixin,
    ReplayBufferMixin,
    TargetNetworkMixin,
)
from rejax.buffers import Minibatch
from rejax.networks import DeterministicPolicy, QNetwork
from rejax.statistics import explained_variance


# Algorithm outline
# num_eval_iterations = total_timesteps / eval_freq
# num_train_iterations = eval_freq / (num_envs * policy_delay)
# for _ in range(num_eval_iterations):
#   for _ in range(num_train_iterations):
#     for _ in range(policy_delay):
#       M = collect num_gradient_steps minibatches
#       update critic using M
#     update actor using M
#     update target networks


class TD3(
    ReplayBufferMixin,
    TargetNetworkMixin,
    NormalizeObservationsMixin,
    NormalizeRewardsMixin,
    Algorithm,
):
    actor: nn.Module = struct.field(pytree_node=False, default=None)
    critic: nn.Module = struct.field(pytree_node=False, default=None)
    num_critics: int = struct.field(pytree_node=False, default=2)
    num_epochs: int = struct.field(pytree_node=False, default=1)
    exploration_noise: chex.Scalar = struct.field(pytree_node=True, default=0.3)
    target_noise: chex.Scalar = struct.field(pytree_node=True, default=0.2)
    target_noise_clip: chex.Scalar = struct.field(pytree_node=True, default=0.5)
    policy_delay: int = struct.field(pytree_node=False, default=2)

    def make_act(self, ts):
        def act(obs, rng):
            if self.normalize_observations:
                obs = self.normalize_obs(ts.obs_rms_state, obs)

            obs = jnp.expand_dims(obs, 0)
            action = self.actor.apply(ts.actor_ts.params, obs)
            return jnp.squeeze(action)  # ty:ignore[invalid-argument-type]

        return act

    @classmethod
    def create_agent(cls, config, env, env_params):
        actor_kwargs = config.pop("actor_kwargs", {})
        activation = actor_kwargs.pop("activation", "swish")
        activation = getattr(nn, activation)
        action_range = (
            env.action_space(env_params).low,
            env.action_space(env_params).high,
        )
        action_dim = np.prod(env.action_space(env_params).shape)
        actor = DeterministicPolicy(
            action_dim=action_dim,
            action_range=action_range,
            hidden_layer_sizes=(64, 64),  # ty:ignore[invalid-argument-type]
            activation=activation,
            **actor_kwargs,
        )

        critic_kwargs = config.pop("critic_kwargs", {})
        activation = critic_kwargs.pop("activation", "swish")
        activation = getattr(nn, activation)
        critic = QNetwork(
            hidden_layer_sizes=(64, 64),
            activation=activation,
            **critic_kwargs,
        )

        return {"actor": actor, "critic": critic}

    @register_init
    def initialize_network_params(self, rng):
        rng, rng_actor, rng_critic = jax.random.split(rng, 3)
        rng_critic = jax.random.split(rng_critic, self.num_critics)
        obs_ph = jnp.empty((1, *self.env.observation_space(self.env_params).shape))
        action_ph = jnp.empty((1, *self.env.action_space(self.env_params).shape))

        tx = optax.chain(
            optax.clip_by_global_norm(self.max_grad_norm),
            optax.adam(learning_rate=self.learning_rate),
        )

        actor_params = self.actor.init(rng_actor, obs_ph)
        actor_ts = TrainState.create(apply_fn=(), params=actor_params, tx=tx)

        vmap_init = jax.vmap(self.critic.init, in_axes=(0, None, None))
        critic_params = vmap_init(rng_critic, obs_ph, action_ph)
        critic_ts = TrainState.create(apply_fn=(), params=critic_params, tx=tx)
        return {
            "actor_ts": actor_ts,
            "actor_target_params": actor_params,
            "critic_ts": critic_ts,
            "critic_target_params": critic_params,
        }

    @property
    def vmap_critic(self):
        return jax.vmap(self.critic.apply, in_axes=(0, None, None))

    def train(self, rng=None, train_state=None):
        if train_state is None and rng is None:
            raise ValueError("Either train_state or rng must be provided")

        ts = train_state or self.init_state(rng)  # ty:ignore[invalid-argument-type]

        if not self.skip_initial_evaluation:
            _, sample_loss_metrics = self.train_iteration(ts)
            zero_loss_metrics = jax.tree.map(jnp.zeros_like, sample_loss_metrics)
            initial_evaluation = self.eval_callback(self, ts, ts.rng, zero_loss_metrics)

        def eval_iteration(ts, unused):
            # Run a few trainig iterations
            steps_per_train_it = self.num_envs * self.policy_delay
            num_train_its = np.ceil(self.eval_freq / steps_per_train_it).astype(int)
            ts, loss_metrics = jax.lax.scan(
                lambda ts, _: self.train_iteration(ts),
                ts,
                None,
                length=num_train_its,
            )

            # Average loss metrics over the training iterations
            avg_loss_metrics = jax.tree.map(lambda x: jnp.mean(x, axis=0), loss_metrics)

            # Run evaluation
            return ts, self.eval_callback(self, ts, ts.rng, avg_loss_metrics)

        ts, evaluation = jax.lax.scan(
            eval_iteration,
            ts,
            None,
            np.ceil(self.total_timesteps / self.eval_freq).astype(int),
        )

        if not self.skip_initial_evaluation:
            evaluation = jax.tree.map(
                lambda i, ev: jnp.concatenate((jnp.expand_dims(i, 0), ev)),
                initial_evaluation,
                evaluation,
            )

        return ts, evaluation

    def train_iteration(self, ts):
        old_global_step = ts.global_step
        placeholder_minibatch = jax.tree.map(
            lambda sdstr: jnp.empty((self.num_epochs, *sdstr.shape), sdstr.dtype),
            ts.replay_buffer.sample(self.batch_size, jax.random.PRNGKey(0)),
        )

        def critic_step(carry, _):
            ts, _ = carry
            ts, minibatch, critic_metrics = self.train_critic(ts)
            return (ts, minibatch), critic_metrics

        (ts, last_minibatch), critic_metrics = jax.lax.scan(
            critic_step,
            (ts, placeholder_minibatch),
            None,
            length=self.policy_delay,
        )

        ts, actor_metrics = self.train_policy(ts, last_minibatch, old_global_step)

        # Average over policy_delay and num_epochs
        critic_metrics = jax.tree.map(
            lambda x: jnp.mean(x, axis=(0, 1)), critic_metrics
        )
        actor_metrics = jax.tree.map(lambda x: jnp.mean(x, axis=0), actor_metrics)

        return ts, {**critic_metrics, **actor_metrics}

    def train_critic(self, ts):
        start_training = ts.global_step > self.fill_buffer

        # Collect transition
        uniform = jnp.logical_not(start_training)
        ts, transitions = self.collect_transitions(ts, uniform=uniform)
        ts = ts.replace(replay_buffer=ts.replay_buffer.extend(transitions))

        def update_iteration(ts, unused):
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

            # Update network
            ts, critic_metrics = self.update_critic(ts, minibatch)
            return ts, (minibatch, critic_metrics)

        def do_updates(ts):
            ts, (minibatches, critic_metrics) = jax.lax.scan(
                update_iteration, ts, None, self.num_epochs
            )
            return ts, (minibatches, critic_metrics)

        placeholder_minibatch = jax.tree.map(
            lambda sdstr: jnp.empty((self.num_epochs, *sdstr.shape), sdstr.dtype),
            ts.replay_buffer.sample(self.batch_size, jax.random.PRNGKey(0)),
        )
        mock_critic_metrics = {
            "critic/total_loss": jnp.zeros((self.num_epochs,)),
            "critic/explained_variance": jnp.zeros((self.num_epochs,)),
            "critic/grad_norm": jnp.zeros((self.num_epochs,)),
            "critic/param_norm": jnp.zeros((self.num_epochs,)),
            "critic/momentum_norm": jnp.zeros((self.num_epochs,)),
            "critic/variance_norm": jnp.zeros((self.num_epochs,)),
            "critic/q1": jnp.zeros((self.num_epochs,)),
            "critic/q2": jnp.zeros((self.num_epochs,)),
            "critic/q_target": jnp.zeros((self.num_epochs,)),
            "critic/loss_q1": jnp.zeros((self.num_epochs,)),
            "critic/loss_q2": jnp.zeros((self.num_epochs,)),
        }
        ts, (minibatches, critic_metrics) = jax.lax.cond(
            start_training,
            do_updates,
            lambda ts: (ts, (placeholder_minibatch, mock_critic_metrics)),
            ts,
        )
        return ts, minibatches, critic_metrics

    def train_policy(self, ts, minibatches, old_global_step):
        def do_updates(ts):
            def update_actor_step(ts, minibatch):
                ts, actor_metrics = self.update_actor(ts, minibatch)
                return ts, actor_metrics

            ts, actor_metrics = jax.lax.scan(
                update_actor_step,
                ts,
                minibatches,
            )
            return ts, actor_metrics

        start_training = ts.global_step > self.fill_buffer
        mock_actor_metrics = {
            "actor/total_loss": jnp.zeros((self.num_epochs,)),
            "actor/q": jnp.zeros((self.num_epochs,)),
            "actor/grad_norm": jnp.zeros((self.num_epochs,)),
            "actor/param_norm": jnp.zeros((self.num_epochs,)),
            "actor/momentum_norm": jnp.zeros((self.num_epochs,)),
            "actor/variance_norm": jnp.zeros((self.num_epochs,)),
        }
        ts, actor_metrics = jax.lax.cond(
            start_training,
            do_updates,
            lambda ts: (ts, mock_actor_metrics),
            ts,
        )

        # Update target networks
        if self.target_update_freq == 1:
            critic_tp = self.polyak_update(ts.critic_ts.params, ts.critic_target_params)
            actor_tp = self.polyak_update(ts.actor_ts.params, ts.actor_target_params)
        else:
            update_target_params = (
                ts.global_step % self.target_update_freq
                <= old_global_step % self.target_update_freq
            )
            critic_tp = jax.tree.map(
                lambda q, qt: jax.lax.select(update_target_params, q, qt),
                self.polyak_update(ts.critic_ts.params, ts.critic_target_params),
                ts.critic_target_params,
            )
            actor_tp = jax.tree.map(
                lambda pi, pit: jax.lax.select(update_target_params, pi, pit),
                self.polyak_update(ts.actor_ts.params, ts.actor_target_params),
                ts.actor_target_params,
            )

        ts = ts.replace(critic_target_params=critic_tp, actor_target_params=actor_tp)
        return ts, actor_metrics

    def collect_transitions(self, ts, uniform=False):
        # Sample actions
        rng, rng_action = jax.random.split(ts.rng)
        ts = ts.replace(rng=rng)

        def sample_uniform(rng):
            sample_fn = self.env.action_space(self.env_params).sample
            return jax.vmap(sample_fn)(jax.random.split(rng, self.num_envs))

        def sample_policy(rng):
            if self.normalize_observations:
                last_obs = self.normalize_obs(ts.obs_rms_state, ts.last_obs)
            else:
                last_obs = ts.last_obs

            actions = self.actor.apply(ts.actor_ts.params, last_obs)
            noise = self.exploration_noise * jax.random.normal(rng, actions.shape)  # ty:ignore[unresolved-attribute]
            action_low, action_high = self.action_space.low, self.action_space.high
            return jnp.clip(actions + noise, action_low, action_high)  # ty:ignore[unsupported-operator]

        actions = jax.lax.cond(uniform, sample_uniform, sample_policy, rng_action)

        # Step environment
        rng, rng_steps = jax.random.split(ts.rng)
        ts = ts.replace(rng=rng)
        rng_steps = jax.random.split(rng_steps, self.num_envs)
        next_obs, env_state, rewards, dones, _ = self.vmap_step(
            rng_steps, ts.env_state, actions, self.env_params
        )
        new_global_step = ts.global_step + self.num_envs
        new_episode_return = (ts.episode_return + rewards) * (1 - dones)

        if self.normalize_observations:
            ts = ts.replace(
                obs_rms_state=self.update_obs_rms(ts.obs_rms_state, next_obs)
            )
        if self.normalize_rewards:
            ts = ts.replace(
                rew_rms_state=self.update_rew_rms(ts.rew_rms_state, rewards, dones)
            )

        # Return minibatch and updated train state
        minibatch = Minibatch(
            obs=ts.last_obs,
            action=actions,
            reward=rewards,
            next_obs=next_obs,
            done=dones,
        )
        ts = ts.replace(
            last_obs=next_obs,
            env_state=env_state,
            global_step=new_global_step,
            episode_return=new_episode_return,
        )
        return ts, minibatch

    def update_critic(self, ts, minibatch):
        def critic_loss_fn(params):
            action = self.actor.apply(ts.actor_target_params, minibatch.next_obs)
            noise = jnp.clip(
                self.target_noise * jax.random.normal(ts.rng, action.shape),  # ty:ignore[unresolved-attribute]
                -self.target_noise_clip,
                self.target_noise_clip,
            )
            action_low, action_high = self.action_space.low, self.action_space.high
            action = jnp.clip(action + noise, action_low, action_high)  # ty:ignore[unsupported-operator]

            qs_target = self.vmap_critic(
                ts.critic_target_params, minibatch.next_obs, action
            )
            q_target = jnp.min(qs_target, axis=0)
            target = minibatch.reward + (1 - minibatch.done) * self.gamma * q_target
            q1, q2 = self.vmap_critic(params, minibatch.obs, minibatch.action)

            loss_q1 = optax.l2_loss(q1, target).mean()
            loss_q2 = optax.l2_loss(q2, target).mean()
            return loss_q1 + loss_q2, (
                (target, q1),
                {
                    "critic/q1": q1.mean(),
                    "critic/q2": q2.mean(),
                    "critic/q_target": target.mean(),
                    "critic/loss_q1": loss_q1,
                    "critic/loss_q2": loss_q2,
                },
            )

        (loss, ((target, value), aux)), grads = jax.value_and_grad(
            critic_loss_fn, has_aux=True
        )(ts.critic_ts.params)
        ts = ts.replace(critic_ts=ts.critic_ts.apply_gradients(grads=grads))
        return ts, {
            "critic/total_loss": loss,
            "critic/explained_variance": explained_variance(target, value),
            "critic/grad_norm": optax.global_norm(grads),
            "critic/param_norm": optax.global_norm(ts.critic_ts.params),
            "critic/momentum_norm": optax.global_norm(ts.critic_ts.opt_state[1][0].mu),
            "critic/variance_norm": optax.global_norm(ts.critic_ts.opt_state[1][0].nu),
            **aux,
        }

    def update_actor(self, ts, minibatch):
        def actor_loss_fn(params):
            action = self.actor.apply(params, minibatch.obs)
            q = self.vmap_critic(ts.critic_ts.params, minibatch.obs, action)
            return -q.mean(), {
                "actor/q": q.mean(),
            }

        (loss, aux), grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(
            ts.actor_ts.params
        )
        ts = ts.replace(actor_ts=ts.actor_ts.apply_gradients(grads=grads))
        return ts, {
            "actor/total_loss": loss,
            "actor/grad_norm": optax.global_norm(grads),
            "actor/param_norm": optax.global_norm(ts.actor_ts.params),
            "actor/momentum_norm": optax.global_norm(ts.actor_ts.opt_state[1][0].mu),
            "actor/variance_norm": optax.global_norm(ts.actor_ts.opt_state[1][0].nu),
            **aux,
        }
