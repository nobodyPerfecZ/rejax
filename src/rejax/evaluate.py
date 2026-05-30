from collections.abc import Callable
from functools import partial
from typing import Any, NamedTuple

import chex
import jax
from gymnax.environments import environment


class EvalState(NamedTuple):
    rng: chex.PRNGKey
    env_state: Any
    last_obs: chex.Array
    done: bool = False
    critic_: float = 0.0
    return_: float = 0.0
    length: int = 0


def evaluate_single(
    act: Callable[[chex.Array, chex.PRNGKey], chex.Array],  # act(obs, rng) -> action
    critic: Callable[[chex.Array, chex.Array], chex.Array],  # critic(obs, act) -> value
    env,
    env_params,
    rng,
    max_steps_in_episode,
):
    def no_step(state, _):
        """Performs no environment step."""
        return state, state.env_state

    def step(state, _):
        """Performs one environment step."""
        rng, rng_act, rng_step = jax.random.split(state.rng, 3)
        action = act(state.last_obs, rng_act)
        obs, env_state, reward, done, _ = env.step(
            rng_step, state.env_state, action, env_params
        )
        state = EvalState(
            rng=rng,
            env_state=env_state,
            last_obs=obs,
            done=done,
            critic_=critic(obs, action),  # ty:ignore[invalid-argument-type]
            return_=state.return_ + reward.squeeze(),
            length=state.length + 1,
        )
        return state, state.env_state

    rng_reset, rng_eval, rng_act = jax.random.split(rng, 3)
    obs, env_state = env.reset(rng_reset, env_params)
    action = act(obs, rng_act)
    critic_ = critic(obs, action)

    state = EvalState(rng_eval, env_state, obs, critic_=critic_)  # ty:ignore[invalid-argument-type]
    state, trajectory = jax.lax.scan(
        # Decides to perform an environment step if done is False
        lambda s, _: jax.lax.cond(s.done, no_step, step, s, _),
        state,
        None,
        length=max_steps_in_episode,
    )
    return state.length, state.return_, state.critic_, trajectory


@partial(
    jax.jit,
    static_argnames=("act", "critic", "env", "num_seeds", "max_steps_in_episode"),
)
def evaluate(
    act: Callable[[chex.Array, chex.PRNGKey], chex.Array],
    critic: Callable[[chex.Array, chex.Array], chex.Array],
    rng: chex.PRNGKey,
    env: environment.Environment,
    env_params: Any,
    num_seeds: int = 128,
    max_steps_in_episode: int | None = None,
) -> tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
    """Evaluate a policy given by `act` on `num_seeds` environments.

    Args:
        act (Callable[[chex.Array, chex.PRNGKey], chex.Array]): A policy represented as
        a function of type (obs, rng) -> action.
        rng (chex.PRNGKey): Initial seed, will be split into `num_seeds` seeds for
        parallel evaluation.
        env (environment.Environment): The environment to evaluate on.
        env_params (Any): The parameters of the environment.
        num_seeds (int): Number of initializations of the environment.

    Returns:
        Tuple[chex.Array, chex.Array, chex.Array]: Tuple of episode length, cumultative
        reward and trajectory for each seed.
    """
    if max_steps_in_episode is None:
        max_steps_in_episode = env_params.max_steps_in_episode

    seeds = jax.random.split(rng, num_seeds)
    vmap_collect = jax.vmap(evaluate_single, in_axes=(None, None, None, None, 0, None))
    return vmap_collect(act, critic, env, env_params, seeds, max_steps_in_episode)
