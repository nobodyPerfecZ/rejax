import warnings
from copy import copy

from flax import struct
from gymnax.environments import spaces
from gymnax.environments.environment import Environment as GymnaxEnv
from jax import numpy as jnp
from mujoco_playground import registry
from mujoco_playground._src.mjx_env import MjxEnv
from mujoco_playground._src.mjx_env import State as MjxState


def create_mujoco_playground(env_name, obs_key="state", config=None, **kwargs):
    """Create a MuJoCo Playground environment, e.g. `"CartpoleBalance"` or
    `"Go1JoystickFlatTerrain"`. Envs with dict observations (typically a `"state"` and
    a `"privileged_state"` entry) are flattened to the entry given by `obs_key`.

    Note that Playground names its classic locomotion robots after the dm_control
    suite task, not after the Gym/Brax env: use `"CheetahRun"` for HalfCheetah,
    `"WalkerRun"` (or `"WalkerWalk"`/`"WalkerStand"`) for Walker2d, and
    `"HopperHop"` (or `"HopperStand"`) for Hopper. Playground has no Ant env.
    """
    if env_name not in registry.ALL_ENVS:
        raise ValueError(
            f"Unknown MuJoCo Playground env {env_name!r}. Available envs are "
            f"{', '.join(registry.ALL_ENVS)}."
        )

    if config is None:
        config = registry.get_default_config(env_name)

    # Playground defaults to the warp backend, which rejax cannot jit and vmap over.
    if "impl" in config and "impl" not in kwargs.get("config_overrides", {}):
        config.impl = "jax"

    env = registry.load(env_name, config=config, **kwargs)
    env = MujocoPlayground2GymnaxEnv(env, config.episode_length, obs_key)
    return env, env.default_params


@struct.dataclass
class EnvState:
    env_state: MjxState
    time: int


@struct.dataclass
class EnvParams:
    # CAUTION: Passing params with a different value than on init has no effect
    max_steps_in_episode: int = 1000


class MujocoPlayground2GymnaxEnv(GymnaxEnv):
    def __init__(self, env: MjxEnv, episode_length: int, obs_key: str = "state"):
        self.env = env
        self.obs_key = obs_key
        self.max_steps_in_episode = episode_length

        obs_size = env.observation_size
        if isinstance(obs_size, dict):
            if obs_key not in obs_size:
                raise ValueError(
                    f"{env.__class__.__name__} has observations "
                    f"{sorted(obs_size)}, but obs_key is {obs_key!r}"
                )
            obs_size = obs_size[obs_key]
        self.obs_shape = obs_size if isinstance(obs_size, tuple) else (obs_size,)

    @property
    def default_params(self):
        return EnvParams(max_steps_in_episode=self.max_steps_in_episode)

    def step_env(self, key, state, action, params):
        env_state = self.env.step(state.env_state, action)
        state = EnvState(env_state=env_state, time=state.time + 1)
        return (
            self.get_obs(state),
            state,
            env_state.reward,
            self.is_terminal(state, params),
            env_state.metrics,
        )

    def reset_env(self, key, params):
        state = EnvState(env_state=self.env.reset(key), time=0)
        return self.get_obs(state), state

    def get_obs(self, state):
        obs = state.env_state.obs
        return obs[self.obs_key] if isinstance(obs, dict) else obs

    def is_terminal(self, state, params):
        # Playground envs do not enforce the step limit themselves
        truncated = state.time >= params.max_steps_in_episode
        return jnp.logical_or(state.env_state.done.astype(bool), truncated)

    @property
    def name(self):
        return self.env.__class__.__name__

    def action_space(self, params):
        # Playground envs take normalized actions and rescale them internally
        return spaces.Box(low=-1, high=1, shape=(self.env.action_size,))

    def observation_space(self, params):
        return spaces.Box(low=-jnp.inf, high=jnp.inf, shape=self.obs_shape)

    @property
    def num_actions(self) -> int:
        return self.env.action_size

    def __deepcopy__(self, memo):
        warnings.warn(
            f"Trying to deepcopy {type(self).__name__}, which contains a mujoco model. "
            "Mujoco models throw an error when deepcopying, so a shallow copy is "
            "returned.",
            category=RuntimeWarning,
            stacklevel=2,
        )
        return copy(self)
