import warnings
from copy import deepcopy
from typing import Any, Callable, Type

import chex
import distrax
import gymnax
import jax
from flax import linen as nn
from flax import struct
from gymnax.environments.environment import Environment
from jax import numpy as jnp
from optax import linear_schedule

from rejax.algos.iqn.iqn import IQN
from rejax.algos.networks import ImplicitQuantileNetwork
from rejax.brax2gymnax import Brax2GymnaxEnv
from rejax.evaluate import make_evaluate


def EpsilonGreedyPolicy(iqn: nn.Module) -> Type[nn.Module]:
    class EpsilonGreedyPolicy(iqn):
        def _action_dist(self, obs, rng, epsilon):
            q = self.q(obs, rng)
            return distrax.EpsilonGreedy(q, epsilon=epsilon)

        def act(self, obs, rng, epsilon):
            rng_tau, rng_epsilon = jax.random.split(rng)
            action_dist = self._action_dist(obs, rng_tau, epsilon)
            action = action_dist.sample(seed=rng_epsilon)
            return action

    return EpsilonGreedyPolicy


class IQNConfig(struct.PyTreeNode):
    # fmt: off
    # Non-static parameters
    env: Environment                = struct.field(pytree_node=False)
    env_params: Any                 = struct.field(pytree_node=True)
    agent: nn.Module                = struct.field(pytree_node=False)
    eval_callback: Callable         = struct.field(pytree_node=False)

    learning_rate: chex.Scalar      = struct.field(pytree_node=True, default=0.005)
    gamma: chex.Scalar              = struct.field(pytree_node=True, default=0.99)
    max_grad_norm: chex.Scalar      = struct.field(pytree_node=True, default=jnp.inf)
    eps_start: chex.Scalar          = struct.field(pytree_node=True, default=1.0)
    eps_end: chex.Scalar            = struct.field(pytree_node=True, default=0.05)
    target_update_freq: int         = struct.field(pytree_node=True, default=200)
    ddqn: bool                      = struct.field(pytree_node=True, default=True)
    kappa: chex.Scalar              = struct.field(pytree_node=True, default=1.0)

    # Static parameters
    total_timesteps: int            = struct.field(pytree_node=False, default=100_000)
    eval_freq: int                  = struct.field(pytree_node=False, default=10_000)
    num_envs: int                   = struct.field(pytree_node=False, default=1)
    exploration_fraction: float     = struct.field(pytree_node=False, default=0.5)
    buffer_size: int                = struct.field(pytree_node=False, default=100_000)
    fill_buffer: int                = struct.field(pytree_node=False, default=1_000)
    batch_size: int                 = struct.field(pytree_node=False, default=100)
    gradient_steps: int             = struct.field(pytree_node=False, default=1)
    num_tau_samples: int            = struct.field(pytree_node=False, default=63)
    num_tau_prime_samples: int      = struct.field(pytree_node=False, default=64)
    normalize_observations: bool    = struct.field(pytree_node=False, default=False)
    skip_initial_evaluation: bool   = struct.field(pytree_node=False, default=False)
    # fmt: on

    @property
    def epsilon_schedule(self):
        return linear_schedule(
            self.eps_start,
            self.eps_end,
            int(self.exploration_fraction * self.total_timesteps),
        )

    @classmethod
    def create(cls, **kwargs):
        """Create a config object from keyword arguments."""
        return cls._from_dict(kwargs)

    @classmethod
    def from_dict(cls, config):
        """Create a config object from a dictionary. Exists mainly for backwards
        compatibility and will be deprecated in the future."""
        warnings.warn("from_dict is deprecated, use create instead.")
        return cls._from_dict(config)

    @classmethod
    def _from_dict(cls, config):
        config = deepcopy(config)  # Because we're popping from it

        if isinstance(config["env"], str):
            # Get env id and convert to gymnax environment and parameters
            env_kwargs = config.pop("env_kwargs", {})
            env_id = config.pop("env")
            if env_id.startswith("brax"):
                env = Brax2GymnaxEnv(env_id.split("/")[1], **env_kwargs)
                env_params = env.default_params
            else:
                env, env_params = gymnax.make(env_id, **env_kwargs)
        else:
            env = config.pop("env")
            env_params = config.pop("env_params", env.default_params)

        agent_kwargs = config.pop("agent_kwargs", {})
        activation = agent_kwargs.pop("activation", "swish")
        agent_kwargs["activation"] = getattr(nn, activation)

        action_dim = env.action_space(env_params).n
        agent = EpsilonGreedyPolicy(ImplicitQuantileNetwork)(
            hidden_layer_sizes=(64, 64), action_dim=action_dim, **agent_kwargs
        )

        eval_callback = make_evaluate(IQN.make_act, env, env_params, 200)
        return cls(
            env_params=env_params,
            agent=agent,
            env=env,
            eval_callback=eval_callback,
            **config,
        )
