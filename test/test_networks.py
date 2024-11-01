import unittest

import flax.linen as nn
import jax
import jax.numpy as jnp

from rejax.networks import VQuantileNetwork

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