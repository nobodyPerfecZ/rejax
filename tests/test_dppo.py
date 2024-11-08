import unittest

import jax
import numpy as np

from rejax import DPPO, DPPOKurt, DPPOSkew

from .environments import (
    TestEnv1Continuous,
    TestEnv1Discrete,
    TestEnv2Continuous,
    TestEnv2Discrete,
    TestEnv3Continuous,
    TestEnv3Discrete,
    TestEnv4Continuous,
    TestEnv4Discrete,
    TestEnv5Continuous,
    TestEnv5Discrete,
)


class TestEnvironmentsDPPO(unittest.TestCase):
    args = {
        "alpha": 0.0,
        "sr_lambda": 0.95,
        "kappa": 0.2,
        "num_envs": 64,
        "num_steps": 16,
        "num_epochs": 10,
        "learning_rate": 0.0003,
        "total_timesteps": 131072,
        "eval_freq": 131072,
        "skip_initial_evaluation": True,
    }

    def train_fn(self, dppo):
        return DPPO.train(dppo, rng=jax.random.PRNGKey(0))

    def test_env1(self):
        for discrete, env in enumerate([TestEnv1Continuous(), TestEnv1Discrete()]):
            with self.subTest(discrete=bool(discrete)):
                dppo = DPPO.create(env=env, **self.args)
                ts, _ = self.train_fn(dppo)
                value = dppo.critic.apply(ts.critic_ts.params, jax.numpy.array([0]))
                np.testing.assert_allclose(value, 1.0, atol=0.1)

    def test_env2(self):
        for discrete, env in enumerate([TestEnv2Continuous(), TestEnv2Discrete()]):
            with self.subTest(discrete=bool(discrete)):
                dppo = DPPO.create(env=env, **self.args)
                ts, _ = self.train_fn(dppo)

                obs = jax.numpy.array([[-1], [1]])
                rew = obs
                value = dppo.critic.apply(ts.critic_ts.params, obs)

                for v, r in zip(value, rew):
                    np.testing.assert_allclose(v, r.item(), atol=0.1)

    def test_env3(self):
        for discrete, env in enumerate([TestEnv3Continuous(), TestEnv3Discrete()]):
            with self.subTest(discrete=bool(discrete)):
                dppo = DPPO.create(env=env, **self.args)
                ts, _ = self.train_fn(dppo)

                obs = jax.numpy.array([[-1], [1]])
                rew = [1 * dppo.gamma, 1]
                value = dppo.critic.apply(ts.critic_ts.params, obs)

                for v, r in zip(value, rew):
                    np.testing.assert_allclose(v, r, atol=0.1)

    def test_env4(self):
        for discrete, env in enumerate([TestEnv4Continuous(), TestEnv4Discrete()]):
            with self.subTest(discrete=bool(discrete)):
                dppo = DPPO.create(env=env, **self.args)
                ts, _ = self.train_fn(dppo)

                best_action = jax.numpy.array(1.0 if discrete else 2.0)
                value = dppo.critic.apply(ts.critic_ts.params, jax.numpy.array([0]))
                np.testing.assert_allclose(value, best_action, atol=0.1)

                act = dppo.make_act(ts)
                rngs = jax.random.split(jax.random.PRNGKey(0), 10)
                actions = jax.vmap(act, in_axes=(None, 0))(jax.numpy.array([0]), rngs)

                for a in actions:
                    self.assertAlmostEqual(a, best_action, delta=0.1)

    def test_env5(self):
        for discrete, env in enumerate([TestEnv5Continuous(), TestEnv5Discrete()]):
            with self.subTest(discrete=bool(discrete)):
                dppo = DPPO.create(env=env, **self.args)
                ts, _ = self.train_fn(dppo)

                rng = jax.random.PRNGKey(0)
                if not discrete:
                    obs = jax.random.uniform(rng, (10, 1), minval=-1, maxval=1)
                else:
                    obs = 2 * jax.random.bernoulli(rng, shape=(10, 1)) - 1

                if not discrete:
                    value = dppo.critic.apply(ts.critic_ts.params, obs)
                    for v in value:
                        np.testing.assert_allclose(v, 0.0, atol=0.1)

                act = dppo.make_act(ts)
                rngs = jax.random.split(rng, 10)
                actions = jax.vmap(act)(obs, rngs)

                for o, a in zip(obs, actions):
                    if discrete:
                        self.assertEqual(a > 0.5, o > 0)
                    else:
                        self.assertAlmostEqual(a, o, delta=0.2)


class TestEnvironmentsDPPOKurt(unittest.TestCase):
    args = {
        "kurtosis_coef": 1e-4,
        "alpha": 0.0,
        "sr_lambda": 0.95,
        "kappa": 0.2,
        "num_envs": 64,
        "num_steps": 16,
        "num_epochs": 10,
        "learning_rate": 0.0003,
        "total_timesteps": 131072,
        "eval_freq": 131072,
        "skip_initial_evaluation": True,
    }

    def train_fn(self, dppo):
        return DPPOKurt.train(dppo, rng=jax.random.PRNGKey(0))

    def test_env1(self):
        for discrete, env in enumerate([TestEnv1Continuous(), TestEnv1Discrete()]):
            with self.subTest(discrete=bool(discrete)):
                dppo = DPPOKurt.create(env=env, **self.args)
                ts, _ = self.train_fn(dppo)
                value = dppo.critic.apply(ts.critic_ts.params, jax.numpy.array([0]))
                np.testing.assert_allclose(value, 1.0, atol=0.1)

    def test_env2(self):
        for discrete, env in enumerate([TestEnv2Continuous(), TestEnv2Discrete()]):
            with self.subTest(discrete=bool(discrete)):
                dppo = DPPOKurt.create(env=env, **self.args)
                ts, _ = self.train_fn(dppo)

                obs = jax.numpy.array([[-1], [1]])
                rew = obs
                value = dppo.critic.apply(ts.critic_ts.params, obs)

                for v, r in zip(value, rew):
                    np.testing.assert_allclose(v, r.item(), atol=0.1)

    def test_env3(self):
        for discrete, env in enumerate([TestEnv3Continuous(), TestEnv3Discrete()]):
            with self.subTest(discrete=bool(discrete)):
                dppo = DPPOKurt.create(env=env, **self.args)
                ts, _ = self.train_fn(dppo)

                obs = jax.numpy.array([[-1], [1]])
                rew = [1 * dppo.gamma, 1]
                value = dppo.critic.apply(ts.critic_ts.params, obs)

                for v, r in zip(value, rew):
                    np.testing.assert_allclose(v, r, atol=0.1)

    def test_env4(self):
        for discrete, env in enumerate([TestEnv4Continuous(), TestEnv4Discrete()]):
            with self.subTest(discrete=bool(discrete)):
                dppo = DPPOKurt.create(env=env, **self.args)
                ts, _ = self.train_fn(dppo)

                best_action = jax.numpy.array(1.0 if discrete else 2.0)
                value = dppo.critic.apply(ts.critic_ts.params, jax.numpy.array([0]))
                np.testing.assert_allclose(value, best_action, atol=0.1)

                act = dppo.make_act(ts)
                rngs = jax.random.split(jax.random.PRNGKey(0), 10)
                actions = jax.vmap(act, in_axes=(None, 0))(jax.numpy.array([0]), rngs)

                for a in actions:
                    self.assertAlmostEqual(a, best_action, delta=0.1)

    def test_env5(self):
        for discrete, env in enumerate([TestEnv5Continuous(), TestEnv5Discrete()]):
            with self.subTest(discrete=bool(discrete)):
                dppo = DPPOKurt.create(env=env, **self.args)
                ts, _ = self.train_fn(dppo)

                rng = jax.random.PRNGKey(0)
                if not discrete:
                    obs = jax.random.uniform(rng, (10, 1), minval=-1, maxval=1)
                else:
                    obs = 2 * jax.random.bernoulli(rng, shape=(10, 1)) - 1

                if not discrete:
                    value = dppo.critic.apply(ts.critic_ts.params, obs)
                    for v in value:
                        np.testing.assert_allclose(v, 0.0, atol=0.2)

                act = dppo.make_act(ts)
                rngs = jax.random.split(rng, 10)
                actions = jax.vmap(act)(obs, rngs)

                for o, a in zip(obs, actions):
                    if discrete:
                        self.assertEqual(a > 0.5, o > 0)
                    else:
                        self.assertAlmostEqual(a, o, delta=0.2)


class TestEnvironmentsDPPOSkew(unittest.TestCase):
    args = {
        "skewness_coef": 1e-3,
        "alpha": 0.0,
        "sr_lambda": 0.95,
        "kappa": 0.2,
        "num_envs": 64,
        "num_steps": 16,
        "num_epochs": 10,
        "learning_rate": 0.0003,
        "total_timesteps": 131072,
        "eval_freq": 131072,
        "skip_initial_evaluation": True,
    }

    def train_fn(self, dppo):
        return DPPOSkew.train(dppo, rng=jax.random.PRNGKey(0))

    def test_env1(self):
        for discrete, env in enumerate([TestEnv1Continuous(), TestEnv1Discrete()]):
            with self.subTest(discrete=bool(discrete)):
                dppo = DPPOSkew.create(env=env, **self.args)
                ts, _ = self.train_fn(dppo)
                value = dppo.critic.apply(ts.critic_ts.params, jax.numpy.array([0]))
                np.testing.assert_allclose(value, 1.0, atol=0.1)

    def test_env2(self):
        for discrete, env in enumerate([TestEnv2Continuous(), TestEnv2Discrete()]):
            with self.subTest(discrete=bool(discrete)):
                dppo = DPPOSkew.create(env=env, **self.args)
                ts, _ = self.train_fn(dppo)

                obs = jax.numpy.array([[-1], [1]])
                rew = obs
                value = dppo.critic.apply(ts.critic_ts.params, obs)

                for v, r in zip(value, rew):
                    np.testing.assert_allclose(v, r.item(), atol=0.1)

    def test_env3(self):
        for discrete, env in enumerate([TestEnv3Continuous(), TestEnv3Discrete()]):
            with self.subTest(discrete=bool(discrete)):
                dppo = DPPOSkew.create(env=env, **self.args)
                ts, _ = self.train_fn(dppo)

                obs = jax.numpy.array([[-1], [1]])
                rew = [1 * dppo.gamma, 1]
                value = dppo.critic.apply(ts.critic_ts.params, obs)

                for v, r in zip(value, rew):
                    np.testing.assert_allclose(v, r, atol=0.1)

    def test_env4(self):
        for discrete, env in enumerate([TestEnv4Continuous(), TestEnv4Discrete()]):
            with self.subTest(discrete=bool(discrete)):
                dppo = DPPOSkew.create(env=env, **self.args)
                ts, _ = self.train_fn(dppo)

                best_action = jax.numpy.array(1.0 if discrete else 2.0)
                value = dppo.critic.apply(ts.critic_ts.params, jax.numpy.array([0]))
                np.testing.assert_allclose(value, best_action, atol=0.1)

                act = dppo.make_act(ts)
                rngs = jax.random.split(jax.random.PRNGKey(0), 10)
                actions = jax.vmap(act, in_axes=(None, 0))(jax.numpy.array([0]), rngs)

                for a in actions:
                    self.assertAlmostEqual(a, best_action, delta=0.1)

    def test_env5(self):
        for discrete, env in enumerate([TestEnv5Continuous(), TestEnv5Discrete()]):
            with self.subTest(discrete=bool(discrete)):
                dppo = DPPOSkew.create(env=env, **self.args)
                ts, _ = self.train_fn(dppo)

                rng = jax.random.PRNGKey(0)
                if not discrete:
                    obs = jax.random.uniform(rng, (10, 1), minval=-1, maxval=1)
                else:
                    obs = 2 * jax.random.bernoulli(rng, shape=(10, 1)) - 1

                if not discrete:
                    value = dppo.critic.apply(ts.critic_ts.params, obs)
                    for v in value:
                        np.testing.assert_allclose(v, 0.0, atol=0.6)

                act = dppo.make_act(ts)
                rngs = jax.random.split(rng, 10)
                actions = jax.vmap(act)(obs, rngs)

                for o, a in zip(obs, actions):
                    if discrete:
                        self.assertEqual(a > 0.5, o > 0)
                    else:
                        self.assertAlmostEqual(a, o, delta=0.6)
