import chex
import jax.numpy as jnp
from jax.scipy.stats import norm


def _distorted_expectation(
    quantiles: chex.Array,
    distorted_taus: chex.Array,
) -> chex.Array:
    distortion = distorted_taus[1:] - distorted_taus[:-1]
    sorted_quantiles = jnp.sort(quantiles, axis=-1)
    return jnp.sum(distortion * sorted_quantiles, axis=-1)


def risk_measure_neutral(quantiles: chex.Array, _) -> chex.Array:
    return jnp.mean(quantiles, axis=-1)


def risk_measure_cvar(quantiles: chex.Array, beta: float = 1.0) -> chex.Array:
    tau = jnp.linspace(0, 1, num=quantiles.shape[-1] + 1, endpoint=True)
    distorted_tau = jnp.minimum(tau / beta, 1.0)
    return _distorted_expectation(quantiles, distorted_tau)


def risk_measure_cpw(quantiles: chex.Array, beta: float = 1.0) -> chex.Array:
    tau = jnp.linspace(0, 1, num=quantiles.shape[-1] + 1, endpoint=True)
    distorted_tau = tau**beta / (tau**beta + (1 - tau) ** beta) ** (1 / beta)
    return _distorted_expectation(quantiles, distorted_tau)


def risk_measure_wang(quantiles: chex.Array, beta: float = 1.0) -> chex.Array:
    tau = jnp.linspace(0, 1, num=quantiles.shape[-1] + 1, endpoint=True)
    distorted_tau = norm.cdf(norm.ppf(tau) + beta)
    return _distorted_expectation(quantiles, distorted_tau)
