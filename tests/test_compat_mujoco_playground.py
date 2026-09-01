import typing
import unittest

import jax

from rejax import PPO
from rejax.compat import create


try:
    import mujoco_playground  # noqa: F401
except ImportError:
    PLAYGROUND_AVAILABLE = False
else:
    PLAYGROUND_AVAILABLE = True


@unittest.skipUnless(PLAYGROUND_AVAILABLE, "mujoco_playground is not installed")
class TestMujocoPlaygroundCompat(unittest.TestCase):
    # Small and short: this is a smoke test, it does not check that PPO learns
    args: typing.ClassVar[dict] = {
        "num_envs": 4,
        "num_steps": 16,
        "num_epochs": 2,
        "total_timesteps": 128,
        "eval_freq": 128,
        "skip_initial_evaluation": True,
    }

    def test_ppo_trains_on_dm_control_suite(self):
        """PPO runs end-to-end on a flat-observation Playground env."""
        ppo = PPO.create(env="playground/CartpoleBalance", **self.args)
        rng = jax.random.PRNGKey(0)
        _ts, (lengths, returns, _values, _traj) = PPO.train(ppo, rng=rng)

        self.assertTrue(jax.numpy.isfinite(returns).all())
        self.assertTrue((lengths > 0).all())

    def test_ppo_trains_on_dict_obs_env(self):
        """Locomotion envs have dict observations, which the wrapper flattens."""
        ppo = PPO.create(env="playground/Go1JoystickFlatTerrain", **self.args)
        rng = jax.random.PRNGKey(0)
        _ts, (lengths, returns, _values, _traj) = PPO.train(ppo, rng=rng)

        self.assertTrue(jax.numpy.isfinite(returns).all())
        self.assertTrue((lengths > 0).all())

    def test_env_interface(self):
        """Reset and step conform to the gymnax API under jit."""
        rng = jax.random.PRNGKey(0)
        env, params = create("playground/CartpoleBalance")

        obs, state = jax.jit(env.reset)(rng, params)
        obs_space = env.observation_space(params)
        self.assertEqual(obs.shape, obs_space.shape)

        action_space = env.action_space(params)
        action = action_space.sample(rng)
        self.assertEqual(action.shape, (env.num_actions,))

        obs, state, reward, done, _info = jax.jit(env.step)(rng, state, action, params)
        self.assertEqual(obs.shape, obs_space.shape)
        self.assertEqual(reward.shape, ())
        self.assertEqual(done.dtype, jax.numpy.bool_)

    def test_obs_key_selects_observation(self):
        """obs_key picks which entry of a dict observation is used."""
        env, params = create("playground/Go1JoystickFlatTerrain", obs_key="state")
        state_shape = env.observation_space(params).shape

        env, params = create(
            "playground/Go1JoystickFlatTerrain", obs_key="privileged_state"
        )
        privileged_shape = env.observation_space(params).shape

        self.assertNotEqual(state_shape, privileged_shape)

        with self.assertRaises(ValueError):
            create("playground/Go1JoystickFlatTerrain", obs_key="does_not_exist")

    def test_locomotion_envs(self):
        """The classic locomotion robots reset and step under jit.

        Playground names these after the dm_control suite task rather than after
        the Gym/Brax env, and has no Ant equivalent.
        """
        for env_name in ("CheetahRun", "WalkerRun", "HopperHop"):
            with self.subTest(env=env_name):
                rng = jax.random.PRNGKey(0)
                env, params = create(f"playground/{env_name}")

                obs, state = jax.jit(env.reset)(rng, params)
                obs_space = env.observation_space(params)
                self.assertEqual(obs.shape, obs_space.shape)
                self.assertTrue(jax.numpy.isfinite(obs).all())

                action = env.action_space(params).sample(rng)
                self.assertEqual(action.shape, (env.num_actions,))

                obs, _state, reward, done, _info = jax.jit(env.step)(
                    rng, state, action, params
                )
                self.assertEqual(obs.shape, obs_space.shape)
                self.assertTrue(jax.numpy.isfinite(obs).all())
                self.assertTrue(jax.numpy.isfinite(reward))
                self.assertEqual(done.dtype, jax.numpy.bool_)

    def test_unknown_env_name_lists_available_envs(self):
        """Gym/Brax names do not exist in Playground, so the error must guide."""
        with self.assertRaises(ValueError) as ctx:
            create("playground/HalfCheetah")
        self.assertIn("CheetahRun", str(ctx.exception))

    def test_episode_is_truncated_at_step_limit(self):
        """Playground envs do not enforce the step limit, the wrapper must."""
        rng = jax.random.PRNGKey(0)
        env, params = create("playground/CartpoleBalance")
        params = params.replace(max_steps_in_episode=10)

        obs, state = jax.jit(env.reset)(rng, params)
        action = jax.numpy.zeros(env.num_actions)

        def step(carry, key):
            _obs, state = carry
            obs, state, _reward, done, _info = env.step(key, state, action, params)
            return (obs, state), (done, state.time)

        _, (done, time) = jax.lax.scan(step, (obs, state), jax.random.split(rng, 20))

        # CartpoleBalance never terminates on its own, so dones are pure truncation
        self.assertEqual([int(i) for i, d in enumerate(done, 1) if d], [10, 20])
        # ...and the auto-reset puts the step counter back to zero
        self.assertEqual(int(time[9]), 0)


if __name__ == "__main__":
    unittest.main()
