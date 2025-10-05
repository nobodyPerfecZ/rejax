import unittest

import flax.linen as nn
import jax
import jax.numpy as jnp

from rejax.networks import (
    DiscreteQQuantileNetwork,
    QQuantileNetwork,
    VHigherOrderNetwork,
    VNetwork,
    VQuantileNetwork,
)


class TestVNetwork(unittest.TestCase):
    kwargs = {
        "hidden_layer_sizes": (64, 64, 64),
        "activation": nn.relu,
    }

    def test_call_with_jit(self):
        network = VNetwork(**self.kwargs)
        x = jnp.ones((10, 100))
        theta = network.init(jax.random.PRNGKey(0), x)
        apply_fn = jax.jit(network.apply)
        y = apply_fn(theta, x)
        self.assertEqual(y.shape, (10,))


class TestVHigherOrderNetwork(unittest.TestCase):
    kwargs = {
        "hidden_layer_sizes": (64, 64, 64),
        "activation": nn.relu,
    }

    def test_call_with_jit(self):
        network = VHigherOrderNetwork(**self.kwargs)
        x = jnp.ones((10, 100))
        theta = network.init(jax.random.PRNGKey(0), x)
        apply_fn = jax.jit(network.apply)
        y = apply_fn(theta, x)
        self.assertEqual(y.shape, (10, 2))


class TestVQuantileNetwork(unittest.TestCase):
    kwargs = {
        "hidden_layer_sizes": (64, 64, 64),
        "activation": nn.relu,
        "num_quantiles": 200,
    }

    def test_call_with_jit(self):
        network = VQuantileNetwork(**self.kwargs)
        x = jnp.ones((10, 100))
        theta = network.init(jax.random.PRNGKey(0), x)
        apply_fn = jax.jit(network.apply)
        y = apply_fn(theta, x)
        self.assertEqual(y.shape, (10, 200))


class TestQQuantileNetwork(unittest.TestCase):
    kwargs = {
        "hidden_layer_sizes": (64, 64, 64),
        "activation": nn.relu,
        "num_quantiles": 200,
    }

    def test_call_with_jit(self):
        network = QQuantileNetwork(**self.kwargs)
        x = jnp.ones((10, 100))
        a = jnp.ones((10, 100))
        theta = network.init(jax.random.PRNGKey(0), x, a)
        apply_fn = jax.jit(network.apply)
        y = apply_fn(theta, x, a)
        self.assertEqual(y.shape, (10, 200))


class TestDiscreteQQuantileNetwork(unittest.TestCase):
    kwargs = {
        "hidden_layer_sizes": (64, 64, 64),
        "activation": nn.relu,
        "action_dim": 4,
        "num_quantiles": 200,
    }

    def test_call_with_jit(self):
        network = DiscreteQQuantileNetwork(**self.kwargs)
        x = jnp.ones((10, 100))
        theta = network.init(jax.random.PRNGKey(0), x)
        apply_fn = jax.jit(network.apply)
        y = apply_fn(theta, x)
        self.assertEqual(y.shape, (10, 4, 200))

    def test_take(self):
        network = DiscreteQQuantileNetwork(**self.kwargs)
        x = jnp.ones((10, 100))
        a = jnp.zeros((10,), dtype=jnp.int32)
        theta = network.init(jax.random.PRNGKey(0), x)
        apply_fn = jax.jit(lambda *args: network.apply(*args, method="take"))
        y = apply_fn(theta, x, a)
        self.assertEqual(y.shape, (10, 200))
