import unittest
from functools import partial

import jax
import numpy as np

from rejax import DSAC, DSACKurt, DSACSkew, DSACVar

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


class TestEnvironmentsDSAC(unittest.TestCase):
    args = {
        "beta": 0.0,
        "risk_fn": "neutral",
        "num_envs": 1,
        "learning_rate": 0.0003,
        "total_timesteps": 16384,
        "eval_freq": 16384,
        "target_entropy_ratio": 0.05,
        "skip_initial_evaluation": True,
    }

    def train_fn(self, dsac):
        return DSAC.train(dsac, rng=jax.random.PRNGKey(0))

    def test_env1(self):
        for discrete, env in enumerate([TestEnv1Continuous(), TestEnv1Discrete()]):
            with self.subTest(discrete=bool(discrete)):
                dsac = DSAC.create(env=env, **self.args)
                ts, _ = self.train_fn(dsac)
                act = dsac.make_act(ts)

                rng = jax.random.PRNGKey(0)
                rngs = jax.random.split(rng, 10)
                obs = jax.numpy.zeros((10, 1))
                actions = jax.vmap(act)(obs, rngs)

                if discrete:
                    q_fn = jax.vmap(
                        lambda *args: dsac.critic.apply(*args, method="take"),
                        in_axes=(0, None, None),
                    )
                else:
                    actions = jax.numpy.expand_dims(actions, 1)
                    q_fn = jax.vmap(dsac.critic.apply, in_axes=(0, None, None))

                qs = q_fn(ts.critic_ts.params, obs, actions)
                value = qs.min(axis=0)

                for v in value:
                    np.testing.assert_allclose(v, 1.0, atol=0.1)

    def test_env2(self):
        for discrete, env in enumerate([TestEnv2Continuous(), TestEnv2Discrete()]):
            with self.subTest(discrete=bool(discrete)):
                dsac = DSAC.create(env=env, **self.args)
                ts, _ = self.train_fn(dsac)
                act = dsac.make_act(ts)

                rng = jax.random.PRNGKey(0)
                rngs = jax.random.split(rng, 10)
                obs = jax.random.uniform(rng, (10, 1), minval=-1, maxval=1)
                actions = jax.vmap(act)(obs, rngs)

                if discrete:
                    q_fn = jax.vmap(
                        lambda *args: dsac.critic.apply(*args, method="take"),
                        in_axes=(0, None, None),
                    )
                else:
                    actions = jax.numpy.expand_dims(actions, 1)
                    q_fn = jax.vmap(dsac.critic.apply, in_axes=(0, None, None))

                qs = q_fn(ts.critic_ts.params, obs, actions)
                value = qs.min(axis=0)

                for v, r in zip(value, obs):
                    np.testing.assert_allclose(v, r.item(), atol=0.1)

    def test_env3(self):
        for discrete, env in enumerate([TestEnv3Continuous(), TestEnv3Discrete()]):
            with self.subTest(discrete=bool(discrete)):
                dsac = DSAC.create(env=env, **self.args)
                ts, _ = self.train_fn(dsac)
                act = dsac.make_act(ts)

                if discrete:
                    q_fn = jax.vmap(
                        lambda *args: dsac.critic.apply(*args, method="take"),
                        in_axes=(0, None, None),
                    )
                else:
                    q_fn = jax.vmap(dsac.critic.apply, in_axes=(0, None, None))

                @partial(jax.vmap, in_axes=(None, 0))
                def test_i(obs, rng):
                    action = act(obs, rng)
                    action = jax.numpy.expand_dims(action, 0)
                    obs = jax.numpy.expand_dims(obs, 0)
                    if not discrete:
                        action = jax.numpy.expand_dims(action, 1)

                    qs = q_fn(ts.critic_ts.params, obs, action)
                    value = qs.min(axis=0)
                    return value

                rngs = jax.random.split(jax.random.PRNGKey(0), 10)
                for obs in jax.numpy.array([[-1], [1]]):
                    r = 1 * dsac.gamma if obs == -1 else 1
                    for v in test_i(obs, rngs):
                        np.testing.assert_allclose(v, r, atol=0.1)

    # TODO: The following tests does not work well, presumably because of the entropy
    # term in the optimization objective

    def test_env4(self):
        for discrete, env in enumerate([TestEnv4Continuous(), TestEnv4Discrete()]):
            with self.subTest(discrete=bool(discrete)):
                dsac = DSAC.create(env=env, **self.args)
                ts, _ = self.train_fn(dsac)
                act = dsac.make_act(ts)

                if discrete:
                    q_fn = jax.vmap(
                        lambda *args: dsac.critic.apply(*args, method="take"),
                        in_axes=(0, None, None),
                    )
                else:
                    q_fn = jax.vmap(dsac.critic.apply, in_axes=(0, None, None))

                @partial(jax.vmap, in_axes=(None, 0))
                def test_i(obs, rng):
                    action = act(obs, rng)
                    action = jax.numpy.expand_dims(action, 0)
                    obs = jax.numpy.expand_dims(obs, 0)
                    if not discrete:
                        action = jax.numpy.expand_dims(action, 1)

                    qs = q_fn(ts.critic_ts.params, obs, action)
                    value = qs.min(axis=0)
                    return value, action

                num_rngs = 100
                rngs = jax.random.split(jax.random.PRNGKey(1), num_rngs)
                obs = jax.numpy.array([0])
                threshold = 0.5 if discrete else 0.0
                vv, aa = test_i(obs, rngs)

                self.assertGreaterEqual(sum(aa > threshold), 0.9 * num_rngs)
                for v, a in zip(vv, aa):
                    np.testing.assert_allclose(v, a.item(), atol=0.1)

    def test_env5(self):
        for discrete, env in enumerate([TestEnv5Continuous(), TestEnv5Discrete()]):
            with self.subTest(discrete=bool(discrete)):

                dsac = DSAC.create(env=env, **self.args)
                ts, _ = self.train_fn(dsac)

                rng = jax.random.PRNGKey(0)
                if not discrete:
                    obs = jax.random.uniform(rng, (10, 1), minval=-1, maxval=1)
                else:
                    obs = 2 * jax.random.bernoulli(rng, shape=(10, 1)) - 1

                if not discrete:
                    value = dsac.vmap_critic(ts.critic_ts.params, obs, obs)
                    value = value.min(axis=0)
                    for v in value:
                        np.testing.assert_allclose(v, 0.0, atol=0.1)

                act = dsac.make_act(ts)
                vmap_act = jax.vmap(jax.vmap(act, in_axes=(0, None)), in_axes=(None, 0))
                num_rngs = 100
                rngs = jax.random.split(rng, num_rngs)
                actions = vmap_act(obs, rngs)

                if discrete:
                    for i in range(num_rngs):
                        self.assertGreaterEqual(
                            ((actions[:, i] > 0.5) == (obs[i] > 0)).sum(),
                            0.9 * num_rngs,
                        )
                else:
                    for i in range(obs.size):
                        np.testing.assert_allclose(
                            actions[:, i].mean(), obs[i], atol=0.1
                        )
                        self.assertGreaterEqual(
                            jax.numpy.isclose(actions[:, i], obs[i], atol=0.5).sum(),
                            0.9 * num_rngs,
                        )


class TestEnvironmentsDSACVar(unittest.TestCase):
    args = {
        "std_coef": 1e-4,
        "beta": 0.0,
        "risk_fn": "neutral",
        "num_envs": 1,
        "learning_rate": 0.0003,
        "total_timesteps": 16384,
        "eval_freq": 16384,
        "target_entropy_ratio": 0.05,
        "skip_initial_evaluation": True,
    }

    def train_fn(self, dsac):
        return DSACVar.train(dsac, rng=jax.random.PRNGKey(0))

    def test_env1(self):
        for discrete, env in enumerate([TestEnv1Continuous(), TestEnv1Discrete()]):
            with self.subTest(discrete=bool(discrete)):
                dsac = DSACVar.create(env=env, **self.args)
                ts, _ = self.train_fn(dsac)
                act = dsac.make_act(ts)

                rng = jax.random.PRNGKey(0)
                rngs = jax.random.split(rng, 10)
                obs = jax.numpy.zeros((10, 1))
                actions = jax.vmap(act)(obs, rngs)

                if discrete:
                    q_fn = jax.vmap(
                        lambda *args: dsac.critic.apply(*args, method="take"),
                        in_axes=(0, None, None),
                    )
                else:
                    actions = jax.numpy.expand_dims(actions, 1)
                    q_fn = jax.vmap(dsac.critic.apply, in_axes=(0, None, None))

                qs = q_fn(ts.critic_ts.params, obs, actions)
                value = qs.min(axis=0)

                for v in value:
                    np.testing.assert_allclose(v, 1.0, atol=0.1)

    def test_env2(self):
        for discrete, env in enumerate([TestEnv2Continuous(), TestEnv2Discrete()]):
            with self.subTest(discrete=bool(discrete)):
                dsac = DSACVar.create(env=env, **self.args)
                ts, _ = self.train_fn(dsac)
                act = dsac.make_act(ts)

                rng = jax.random.PRNGKey(0)
                rngs = jax.random.split(rng, 10)
                obs = jax.random.uniform(rng, (10, 1), minval=-1, maxval=1)
                actions = jax.vmap(act)(obs, rngs)

                if discrete:
                    q_fn = jax.vmap(
                        lambda *args: dsac.critic.apply(*args, method="take"),
                        in_axes=(0, None, None),
                    )
                else:
                    actions = jax.numpy.expand_dims(actions, 1)
                    q_fn = jax.vmap(dsac.critic.apply, in_axes=(0, None, None))

                qs = q_fn(ts.critic_ts.params, obs, actions)
                value = qs.min(axis=0)

                for v, r in zip(value, obs):
                    np.testing.assert_allclose(v, r.item(), atol=0.1)

    def test_env3(self):
        for discrete, env in enumerate([TestEnv3Continuous(), TestEnv3Discrete()]):
            with self.subTest(discrete=bool(discrete)):
                dsac = DSACVar.create(env=env, **self.args)
                ts, _ = self.train_fn(dsac)
                act = dsac.make_act(ts)

                if discrete:
                    q_fn = jax.vmap(
                        lambda *args: dsac.critic.apply(*args, method="take"),
                        in_axes=(0, None, None),
                    )
                else:
                    q_fn = jax.vmap(dsac.critic.apply, in_axes=(0, None, None))

                @partial(jax.vmap, in_axes=(None, 0))
                def test_i(obs, rng):
                    action = act(obs, rng)
                    action = jax.numpy.expand_dims(action, 0)
                    obs = jax.numpy.expand_dims(obs, 0)
                    if not discrete:
                        action = jax.numpy.expand_dims(action, 1)

                    qs = q_fn(ts.critic_ts.params, obs, action)
                    value = qs.min(axis=0)
                    return value

                rngs = jax.random.split(jax.random.PRNGKey(0), 10)
                for obs in jax.numpy.array([[-1], [1]]):
                    r = 1 * dsac.gamma if obs == -1 else 1
                    for v in test_i(obs, rngs):
                        np.testing.assert_allclose(v, r, atol=0.1)

    # TODO: The following tests does not work well, presumably because of the entropy
    # term in the optimization objective

    def test_env4(self):
        for discrete, env in enumerate([TestEnv4Continuous(), TestEnv4Discrete()]):
            with self.subTest(discrete=bool(discrete)):
                dsac = DSACVar.create(env=env, **self.args)
                ts, _ = self.train_fn(dsac)
                act = dsac.make_act(ts)

                if discrete:
                    q_fn = jax.vmap(
                        lambda *args: dsac.critic.apply(*args, method="take"),
                        in_axes=(0, None, None),
                    )
                else:
                    q_fn = jax.vmap(dsac.critic.apply, in_axes=(0, None, None))

                @partial(jax.vmap, in_axes=(None, 0))
                def test_i(obs, rng):
                    action = act(obs, rng)
                    action = jax.numpy.expand_dims(action, 0)
                    obs = jax.numpy.expand_dims(obs, 0)
                    if not discrete:
                        action = jax.numpy.expand_dims(action, 1)

                    qs = q_fn(ts.critic_ts.params, obs, action)
                    value = qs.min(axis=0)
                    return value, action

                num_rngs = 100
                rngs = jax.random.split(jax.random.PRNGKey(1), num_rngs)
                obs = jax.numpy.array([0])
                threshold = 0.5 if discrete else 0.0
                vv, aa = test_i(obs, rngs)

                self.assertGreaterEqual(sum(aa > threshold), 0.9 * num_rngs)
                for v, a in zip(vv, aa):
                    np.testing.assert_allclose(v, a.item(), atol=0.1)

    def test_env5(self):
        for discrete, env in enumerate([TestEnv5Continuous(), TestEnv5Discrete()]):
            with self.subTest(discrete=bool(discrete)):

                dsac = DSACVar.create(env=env, **self.args)
                ts, _ = self.train_fn(dsac)

                rng = jax.random.PRNGKey(0)
                if not discrete:
                    obs = jax.random.uniform(rng, (10, 1), minval=-1, maxval=1)
                else:
                    obs = 2 * jax.random.bernoulli(rng, shape=(10, 1)) - 1

                if not discrete:
                    value = dsac.vmap_critic(ts.critic_ts.params, obs, obs)
                    value = value.min(axis=0)
                    for v in value:
                        np.testing.assert_allclose(v, 0.0, atol=0.1)

                act = dsac.make_act(ts)
                vmap_act = jax.vmap(jax.vmap(act, in_axes=(0, None)), in_axes=(None, 0))
                num_rngs = 100
                rngs = jax.random.split(rng, num_rngs)
                actions = vmap_act(obs, rngs)

                if discrete:
                    for i in range(num_rngs):
                        self.assertGreaterEqual(
                            ((actions[:, i] > 0.5) == (obs[i] > 0)).sum(),
                            0.9 * num_rngs,
                        )
                else:
                    for i in range(obs.size):
                        np.testing.assert_allclose(
                            actions[:, i].mean(), obs[i], atol=0.1
                        )
                        self.assertGreaterEqual(
                            jax.numpy.isclose(actions[:, i], obs[i], atol=0.5).sum(),
                            0.9 * num_rngs,
                        )


class TestEnvironmentsDSACKurt(unittest.TestCase):
    args = {
        "kurt_coef": 1e-4,
        "beta": 0.0,
        "risk_fn": "neutral",
        "num_envs": 1,
        "learning_rate": 0.0003,
        "total_timesteps": 16384,
        "eval_freq": 16384,
        "target_entropy_ratio": 0.05,
        "skip_initial_evaluation": True,
    }

    def train_fn(self, dsac):
        return DSACKurt.train(dsac, rng=jax.random.PRNGKey(0))

    def test_env1(self):
        for discrete, env in enumerate([TestEnv1Continuous(), TestEnv1Discrete()]):
            with self.subTest(discrete=bool(discrete)):
                dsac = DSACKurt.create(env=env, **self.args)
                ts, _ = self.train_fn(dsac)
                act = dsac.make_act(ts)

                rng = jax.random.PRNGKey(0)
                rngs = jax.random.split(rng, 10)
                obs = jax.numpy.zeros((10, 1))
                actions = jax.vmap(act)(obs, rngs)

                if discrete:
                    q_fn = jax.vmap(
                        lambda *args: dsac.critic.apply(*args, method="take"),
                        in_axes=(0, None, None),
                    )
                else:
                    actions = jax.numpy.expand_dims(actions, 1)
                    q_fn = jax.vmap(dsac.critic.apply, in_axes=(0, None, None))

                qs = q_fn(ts.critic_ts.params, obs, actions)
                value = qs.min(axis=0)

                for v in value:
                    np.testing.assert_allclose(v, 1.0, atol=0.1)

    def test_env2(self):
        for discrete, env in enumerate([TestEnv2Continuous(), TestEnv2Discrete()]):
            with self.subTest(discrete=bool(discrete)):
                dsac = DSACKurt.create(env=env, **self.args)
                ts, _ = self.train_fn(dsac)
                act = dsac.make_act(ts)

                rng = jax.random.PRNGKey(0)
                rngs = jax.random.split(rng, 10)
                obs = jax.random.uniform(rng, (10, 1), minval=-1, maxval=1)
                actions = jax.vmap(act)(obs, rngs)

                if discrete:
                    q_fn = jax.vmap(
                        lambda *args: dsac.critic.apply(*args, method="take"),
                        in_axes=(0, None, None),
                    )
                else:
                    actions = jax.numpy.expand_dims(actions, 1)
                    q_fn = jax.vmap(dsac.critic.apply, in_axes=(0, None, None))

                qs = q_fn(ts.critic_ts.params, obs, actions)
                value = qs.min(axis=0)

                for v, r in zip(value, obs):
                    np.testing.assert_allclose(v, r.item(), atol=0.1)

    def test_env3(self):
        for discrete, env in enumerate([TestEnv3Continuous(), TestEnv3Discrete()]):
            with self.subTest(discrete=bool(discrete)):
                dsac = DSACKurt.create(env=env, **self.args)
                ts, _ = self.train_fn(dsac)
                act = dsac.make_act(ts)

                if discrete:
                    q_fn = jax.vmap(
                        lambda *args: dsac.critic.apply(*args, method="take"),
                        in_axes=(0, None, None),
                    )
                else:
                    q_fn = jax.vmap(dsac.critic.apply, in_axes=(0, None, None))

                @partial(jax.vmap, in_axes=(None, 0))
                def test_i(obs, rng):
                    action = act(obs, rng)
                    action = jax.numpy.expand_dims(action, 0)
                    obs = jax.numpy.expand_dims(obs, 0)
                    if not discrete:
                        action = jax.numpy.expand_dims(action, 1)

                    qs = q_fn(ts.critic_ts.params, obs, action)
                    value = qs.min(axis=0)
                    return value

                rngs = jax.random.split(jax.random.PRNGKey(0), 10)
                for obs in jax.numpy.array([[-1], [1]]):
                    r = 1 * dsac.gamma if obs == -1 else 1
                    for v in test_i(obs, rngs):
                        np.testing.assert_allclose(v, r, atol=0.1)

    # TODO: The following tests does not work well, presumably because of the entropy
    # term in the optimization objective

    def test_env4(self):
        for discrete, env in enumerate([TestEnv4Continuous(), TestEnv4Discrete()]):
            with self.subTest(discrete=bool(discrete)):
                dsac = DSACKurt.create(env=env, **self.args)
                ts, _ = self.train_fn(dsac)
                act = dsac.make_act(ts)

                if discrete:
                    q_fn = jax.vmap(
                        lambda *args: dsac.critic.apply(*args, method="take"),
                        in_axes=(0, None, None),
                    )
                else:
                    q_fn = jax.vmap(dsac.critic.apply, in_axes=(0, None, None))

                @partial(jax.vmap, in_axes=(None, 0))
                def test_i(obs, rng):
                    action = act(obs, rng)
                    action = jax.numpy.expand_dims(action, 0)
                    obs = jax.numpy.expand_dims(obs, 0)
                    if not discrete:
                        action = jax.numpy.expand_dims(action, 1)

                    qs = q_fn(ts.critic_ts.params, obs, action)
                    value = qs.min(axis=0)
                    return value, action

                num_rngs = 100
                rngs = jax.random.split(jax.random.PRNGKey(1), num_rngs)
                obs = jax.numpy.array([0])
                threshold = 0.5 if discrete else 0.0
                vv, aa = test_i(obs, rngs)

                self.assertGreaterEqual(sum(aa > threshold), 0.9 * num_rngs)
                for v, a in zip(vv, aa):
                    np.testing.assert_allclose(v, a.item(), atol=0.1)

    def test_env5(self):
        for discrete, env in enumerate([TestEnv5Continuous(), TestEnv5Discrete()]):
            with self.subTest(discrete=bool(discrete)):

                dsac = DSACKurt.create(env=env, **self.args)
                ts, _ = self.train_fn(dsac)

                rng = jax.random.PRNGKey(0)
                if not discrete:
                    obs = jax.random.uniform(rng, (10, 1), minval=-1, maxval=1)
                else:
                    obs = 2 * jax.random.bernoulli(rng, shape=(10, 1)) - 1

                if not discrete:
                    value = dsac.vmap_critic(ts.critic_ts.params, obs, obs)
                    value = value.min(axis=0)
                    for v in value:
                        np.testing.assert_allclose(v, 0.0, atol=0.1)

                act = dsac.make_act(ts)
                vmap_act = jax.vmap(jax.vmap(act, in_axes=(0, None)), in_axes=(None, 0))
                num_rngs = 100
                rngs = jax.random.split(rng, num_rngs)
                actions = vmap_act(obs, rngs)

                if discrete:
                    for i in range(num_rngs):
                        self.assertGreaterEqual(
                            ((actions[:, i] > 0.5) == (obs[i] > 0)).sum(),
                            0.9 * num_rngs,
                        )
                else:
                    for i in range(obs.size):
                        np.testing.assert_allclose(
                            actions[:, i].mean(), obs[i], atol=0.1
                        )
                        self.assertGreaterEqual(
                            jax.numpy.isclose(actions[:, i], obs[i], atol=0.5).sum(),
                            0.9 * num_rngs,
                        )


class TestEnvironmentsDSACSkew(unittest.TestCase):
    args = {
        "skew_coef": 1e-3,
        "beta": 0.0,
        "risk_fn": "neutral",
        "num_envs": 1,
        "learning_rate": 0.0003,
        "total_timesteps": 16384,
        "eval_freq": 16384,
        "target_entropy_ratio": 0.05,
        "skip_initial_evaluation": True,
    }

    def train_fn(self, dsac):
        return DSACSkew.train(dsac, rng=jax.random.PRNGKey(0))

    def test_env1(self):
        for discrete, env in enumerate([TestEnv1Continuous(), TestEnv1Discrete()]):
            with self.subTest(discrete=bool(discrete)):
                dsac = DSACSkew.create(env=env, **self.args)
                ts, _ = self.train_fn(dsac)
                act = dsac.make_act(ts)

                rng = jax.random.PRNGKey(0)
                rngs = jax.random.split(rng, 10)
                obs = jax.numpy.zeros((10, 1))
                actions = jax.vmap(act)(obs, rngs)

                if discrete:
                    q_fn = jax.vmap(
                        lambda *args: dsac.critic.apply(*args, method="take"),
                        in_axes=(0, None, None),
                    )
                else:
                    actions = jax.numpy.expand_dims(actions, 1)
                    q_fn = jax.vmap(dsac.critic.apply, in_axes=(0, None, None))

                qs = q_fn(ts.critic_ts.params, obs, actions)
                value = qs.min(axis=0)

                for v in value:
                    np.testing.assert_allclose(v, 1.0, atol=0.1)

    def test_env2(self):
        for discrete, env in enumerate([TestEnv2Continuous(), TestEnv2Discrete()]):
            with self.subTest(discrete=bool(discrete)):
                dsac = DSACSkew.create(env=env, **self.args)
                ts, _ = self.train_fn(dsac)
                act = dsac.make_act(ts)

                rng = jax.random.PRNGKey(0)
                rngs = jax.random.split(rng, 10)
                obs = jax.random.uniform(rng, (10, 1), minval=-1, maxval=1)
                actions = jax.vmap(act)(obs, rngs)

                if discrete:
                    q_fn = jax.vmap(
                        lambda *args: dsac.critic.apply(*args, method="take"),
                        in_axes=(0, None, None),
                    )
                else:
                    actions = jax.numpy.expand_dims(actions, 1)
                    q_fn = jax.vmap(dsac.critic.apply, in_axes=(0, None, None))

                qs = q_fn(ts.critic_ts.params, obs, actions)
                value = qs.min(axis=0)

                for v, r in zip(value, obs):
                    np.testing.assert_allclose(v, r.item(), atol=0.1)

    def test_env3(self):
        for discrete, env in enumerate([TestEnv3Continuous(), TestEnv3Discrete()]):
            with self.subTest(discrete=bool(discrete)):
                dsac = DSACSkew.create(env=env, **self.args)
                ts, _ = self.train_fn(dsac)
                act = dsac.make_act(ts)

                if discrete:
                    q_fn = jax.vmap(
                        lambda *args: dsac.critic.apply(*args, method="take"),
                        in_axes=(0, None, None),
                    )
                else:
                    q_fn = jax.vmap(dsac.critic.apply, in_axes=(0, None, None))

                @partial(jax.vmap, in_axes=(None, 0))
                def test_i(obs, rng):
                    action = act(obs, rng)
                    action = jax.numpy.expand_dims(action, 0)
                    obs = jax.numpy.expand_dims(obs, 0)
                    if not discrete:
                        action = jax.numpy.expand_dims(action, 1)

                    qs = q_fn(ts.critic_ts.params, obs, action)
                    value = qs.min(axis=0)
                    return value

                rngs = jax.random.split(jax.random.PRNGKey(0), 10)
                for obs in jax.numpy.array([[-1], [1]]):
                    r = 1 * dsac.gamma if obs == -1 else 1
                    for v in test_i(obs, rngs):
                        np.testing.assert_allclose(v, r, atol=0.1)

    # TODO: The following tests does not work well, presumably because of the entropy
    # term in the optimization objective

    def test_env4(self):
        for discrete, env in enumerate([TestEnv4Continuous(), TestEnv4Discrete()]):
            with self.subTest(discrete=bool(discrete)):
                dsac = DSACSkew.create(env=env, **self.args)
                ts, _ = self.train_fn(dsac)
                act = dsac.make_act(ts)

                if discrete:
                    q_fn = jax.vmap(
                        lambda *args: dsac.critic.apply(*args, method="take"),
                        in_axes=(0, None, None),
                    )
                else:
                    q_fn = jax.vmap(dsac.critic.apply, in_axes=(0, None, None))

                @partial(jax.vmap, in_axes=(None, 0))
                def test_i(obs, rng):
                    action = act(obs, rng)
                    action = jax.numpy.expand_dims(action, 0)
                    obs = jax.numpy.expand_dims(obs, 0)
                    if not discrete:
                        action = jax.numpy.expand_dims(action, 1)

                    qs = q_fn(ts.critic_ts.params, obs, action)
                    value = qs.min(axis=0)
                    return value, action

                num_rngs = 100
                rngs = jax.random.split(jax.random.PRNGKey(1), num_rngs)
                obs = jax.numpy.array([0])
                threshold = 0.5 if discrete else 0.0
                vv, aa = test_i(obs, rngs)

                self.assertGreaterEqual(sum(aa > threshold), 0.9 * num_rngs)
                for v, a in zip(vv, aa):
                    np.testing.assert_allclose(v, a.item(), atol=0.1)

    def test_env5(self):
        for discrete, env in enumerate([TestEnv5Continuous(), TestEnv5Discrete()]):
            with self.subTest(discrete=bool(discrete)):

                dsac = DSACSkew.create(env=env, **self.args)
                ts, _ = self.train_fn(dsac)

                rng = jax.random.PRNGKey(0)
                if not discrete:
                    obs = jax.random.uniform(rng, (10, 1), minval=-1, maxval=1)
                else:
                    obs = 2 * jax.random.bernoulli(rng, shape=(10, 1)) - 1

                if not discrete:
                    value = dsac.vmap_critic(ts.critic_ts.params, obs, obs)
                    value = value.min(axis=0)
                    for v in value:
                        np.testing.assert_allclose(v, 0.0, atol=0.1)

                act = dsac.make_act(ts)
                vmap_act = jax.vmap(jax.vmap(act, in_axes=(0, None)), in_axes=(None, 0))
                num_rngs = 100
                rngs = jax.random.split(rng, num_rngs)
                actions = vmap_act(obs, rngs)

                if discrete:
                    for i in range(num_rngs):
                        self.assertGreaterEqual(
                            ((actions[:, i] > 0.5) == (obs[i] > 0)).sum(),
                            0.9 * num_rngs,
                        )
                else:
                    for i in range(obs.size):
                        np.testing.assert_allclose(
                            actions[:, i].mean(), obs[i], atol=0.1
                        )
                        self.assertGreaterEqual(
                            jax.numpy.isclose(actions[:, i], obs[i], atol=0.5).sum(),
                            0.9 * num_rngs,
                        )
