from functools import partial
from typing import Any

import chex
import jax
import numpy as np
import optax
from flax.core.frozen_dict import FrozenDict
from flax.struct import PyTreeNode
from flax.training.train_state import TrainState
from jax import numpy as jnp

from rejax.algos.algorithm import Algorithm
from rejax.algos.buffers import Minibatch, ReplayBuffer
from rejax.normalize import RMSState, normalize_obs, update_rms


class IQNTrainState(PyTreeNode):
    q_ts: TrainState
    target_params: FrozenDict
    replay_buffer: ReplayBuffer
    env_state: Any
    last_obs: chex.Array
    global_step: int
    rms_state: RMSState
    rng: chex.PRNGKey

    @property
    def params(self):
        return self.q_ts.params

    def get_rng(self):
        rng, rng_new = jax.random.split(self.rng)
        return self.replace(rng=rng), rng_new


class IQN(Algorithm):
    @classmethod
    def make_act(cls, config, ts):
        def act(obs, rng):
            if getattr(config, "normalize_observations", False):
                obs = normalize_obs(ts.rms_state, obs)

            obs = jnp.expand_dims(obs, 0)
            action = config.agent.apply(
                ts.params, obs, rng, epsilon=0.005, method="act"
            )
            return jnp.squeeze(action)

        return act

    @classmethod
    def initialize_train_state(cls, config, rng):
        rng, rng_agent = jax.random.split(rng)
        q_params = config.agent.init(
            rng_agent,
            jnp.zeros((1, *config.env.observation_space(config.env_params).shape)),
            rng_agent,
        )
        tx = optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm),
            optax.adam(config.learning_rate, eps=1e-5),
        )
        q_ts = TrainState.create(apply_fn=(), params=q_params, tx=tx)
        q_target_params = q_params

        rng, rng_reset = jax.random.split(rng)
        rng_reset = jax.random.split(rng_reset, config.num_envs)
        vmap_reset = jax.vmap(config.env.reset, in_axes=(0, None))
        obs, env_state = vmap_reset(rng_reset, config.env_params)

        replay_buffer = ReplayBuffer.empty(
            size=config.buffer_size,
            obs_space=config.env.observation_space(config.env_params),
            action_space=config.env.action_space(config.env_params),
        )

        rms_state = RMSState.create(obs.shape[1:])
        if config.normalize_observations:
            rms_state = update_rms(rms_state, obs)

        train_state = IQNTrainState(
            q_ts=q_ts,
            target_params=q_target_params,
            env_state=env_state,
            replay_buffer=replay_buffer,
            last_obs=obs,
            global_step=0,
            rms_state=rms_state,
            rng=rng,
        )

        return train_state

    @classmethod
    def train(cls, config, rng=None, train_state=None):
        if train_state is None and rng is None:
            raise ValueError("Either train_state or rng must be provided")

        ts = train_state or cls.initialize_train_state(config, rng)

        if not config.skip_initial_evaluation:
            initial_evaluation = config.eval_callback(config, ts, ts.rng)

        def eval_iteration(ts, unused):
            # Run a few trainig iterations
            ts = jax.lax.fori_loop(
                0,
                np.ceil(config.eval_freq / config.num_envs).astype(int),
                lambda _, ts: cls.train_iteration(config, ts),
                ts,
            )

            # Run evaluation
            return ts, config.eval_callback(config, ts, ts.rng)

        ts, evaluation = jax.lax.scan(
            eval_iteration,
            ts,
            None,
            np.ceil(config.total_timesteps / config.eval_freq).astype(int),
        )

        if not config.skip_initial_evaluation:
            evaluation = jax.tree_map(
                lambda i, ev: jnp.concatenate((jnp.expand_dims(i, 0), ev)),
                initial_evaluation,
                evaluation,
            )

        return ts, evaluation

    @classmethod
    def train_iteration(cls, config, ts):
        start_training = ts.global_step > config.fill_buffer
        old_global_step = ts.global_step

        # Calculate epsilon
        epsilon = config.epsilon_schedule(ts.global_step)

        # Collect transitions
        uniform = jnp.logical_not(start_training)
        ts, batch = cls.collect_transitions(config, ts, epsilon, uniform=uniform)
        ts = ts.replace(replay_buffer=ts.replay_buffer.extend(batch))

        # Perform updates to q network
        def update_iteration(ts):
            # Sample minibatch
            ts, rng_sample = ts.get_rng()
            minibatch = ts.replay_buffer.sample(config.batch_size, rng_sample)
            if config.normalize_observations:
                minibatch = minibatch._replace(
                    obs=normalize_obs(ts.rms_state, minibatch.obs),
                    next_obs=normalize_obs(ts.rms_state, minibatch.next_obs),
                )

            # Update network
            ts = cls.update(config, ts, minibatch)
            return ts

        def do_updates(ts):
            return jax.lax.fori_loop(
                0, config.gradient_steps, lambda _, ts: update_iteration(ts), ts
            )

        ts = jax.lax.cond(start_training, lambda: do_updates(ts), lambda: ts)

        # Update target network
        update_target_params = (
            ts.global_step % config.target_update_freq
            <= old_global_step % config.target_update_freq
        )
        target_network = jax.tree_map(
            lambda q, qt: jax.lax.select(update_target_params, q, qt),
            ts.q_ts.params,
            ts.target_params,
        )
        ts = ts.replace(target_params=target_network)

        return ts

    @classmethod
    def collect_transitions(cls, config, ts, epsilon, uniform=False):
        # Sample actions
        ts, rng_action = ts.get_rng()

        def sample_uniform(rng):
            sample_fn = config.env.action_space(config.env_params).sample
            return jax.vmap(sample_fn)(jax.random.split(rng, config.num_envs))

        def sample_policy(rng):
            if config.normalize_observations:
                last_obs = normalize_obs(ts.rms_state, ts.last_obs)
            else:
                last_obs = ts.last_obs

            return config.agent.apply(
                ts.q_ts.params, last_obs, rng, epsilon=epsilon, method="act"
            )

        actions = jax.lax.cond(uniform, sample_uniform, sample_policy, rng_action)

        ts, rng_steps = ts.get_rng()
        rng_steps = jax.random.split(rng_steps, config.num_envs)
        vmap_step = jax.vmap(config.env.step, in_axes=(0, 0, 0, None))
        next_obs, env_state, rewards, dones, _ = vmap_step(
            rng_steps, ts.env_state, actions, config.env_params
        )
        if config.normalize_observations:
            ts = ts.replace(rms_state=update_rms(ts.rms_state, next_obs))

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
            global_step=ts.global_step + config.num_envs,
        )
        return ts, minibatch

    @classmethod
    def update(cls, config, ts, mb):
        # Move tau to axis 1, leaving batch as leading axis
        vmapped_apply = jax.vmap(
            config.agent.apply,
            in_axes=(None, None, 0),
            out_axes=1,
        )

        # Split off multiple keys for tau and tau_prime
        ts, rng = ts.get_rng()
        rng_action, rng_tau, rng_tau_prime = jax.random.split(rng, 3)
        rng_tau = jax.random.split(rng_tau, config.num_tau_samples)
        rng_tau_prime = jax.random.split(rng_tau_prime, config.num_tau_prime_samples)

        best_action = config.agent.apply(
            ts.q_ts.params, mb.next_obs, rng_action, method="best_action"
        )
        zs, _ = vmapped_apply(ts.q_ts.params, mb.next_obs, rng_tau_prime)
        best_z = jnp.take_along_axis(zs, best_action[:, None, None], axis=2).squeeze(2)

        targets = mb.reward[:, None] + config.gamma * (1 - mb.done[:, None]) * best_z
        assert targets.shape == (
            config.batch_size,
            config.num_tau_prime_samples,
        )

        # Vmap over batch and sampled taus
        @jax.vmap
        @jax.vmap
        def rho(td_err, tau):
            l = jnp.where(
                jnp.abs(td_err) <= config.kappa,
                td_err**2 / 2,
                config.kappa * (jnp.abs(td_err) - config.kappa / 2),
            )
            return jnp.abs(tau - (td_err < 0)) * l / config.kappa

        def loss_fn(q_params):
            z, tau = vmapped_apply(q_params, mb.obs, rng_tau)
            z = jnp.take_along_axis(z, mb.action[:, None, None], axis=2).squeeze(2)
            assert z.shape == (config.batch_size, config.num_tau_samples), z.shape

            td_err = jax.vmap(lambda x, y: x[None, :] - y[:, None])(targets, z)

            assert td_err.shape == (
                config.batch_size,
                config.num_tau_samples,
                config.num_tau_prime_samples,
            )
            assert tau.shape == (config.batch_size, config.num_tau_samples)
            assert rho(td_err, tau).shape == (
                config.batch_size,
                config.num_tau_samples,
                config.num_tau_prime_samples,
            )
            loss = rho(td_err, tau).sum(axis=(1, 2)) / config.num_tau_prime_samples
            return loss.mean()

        grads = jax.grad(loss_fn)(ts.q_ts.params)
        ts = ts.replace(q_ts=ts.q_ts.apply_gradients(grads=grads))
        return ts
