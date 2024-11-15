import chex
import jax.numpy as jnp


def conditional_value_at_risk(
    x: chex.Array,
    alpha: float = 0.05,
    axis: int | None = None,
    keepdims: bool = False,
) -> chex.Array:
    """
    Computes the conditional value at risk (CVaR) along the specified axis.

    Args:
        x (chex.Array):
            Array of shape (?, ...)

        alpha (float):
            The confidence level

        axis (int, optional):
            Axis or axes along which the conditional value at risk (CVaR) is computed

        keepdims (bool):
            Controls whether to keep the dimension of x

    Returns:
        chex.Array:
            Array of shape (?, ...)
            The conditional value at risk (CVaR) along the specified axis
    """
    # Compute Value at Risk (VaR)
    var = value_at_risk(x, alpha, axis=axis, keepdims=True)

    cvar = var + jnp.mean(jnp.minimum(x - var, 0), axis=axis, keepdims=True) / (
        alpha + 1e-10
    )

    if keepdims:
        # Case: Keep the dimension of the input
        return cvar
    else:
        # Case: Remove the axis dimension of the input
        return cvar.squeeze(axis=axis)


def kurtosis(
    x: chex.Array,
    fisher: bool = True,
    axis: int | None = None,
    keepdims: bool = False,
) -> chex.Array:
    """
    Computes the kurtosis (Fisher or Pearson) along the specified axis.

    Args:
        x (chex.Array):
            Array of shape (?, ...)

        fisher (bool):
            Controls whether to normalize the kurtosis to the normal distribution

        axis (int, optional):
            Axis or axes along which the kurtosis is computed

        keepdims (bool):
            Controls whether to keep the dimension of x

    Returns:
        chex.Array:
            Array of shape (?, ...)
            The kurtosis along the specified axis
    """
    # Compute mean and standard deviation along the given axis
    mean = jnp.mean(x, axis=axis, keepdims=True)
    std = jnp.std(x, axis=axis, keepdims=True)

    # Compute the kurtosis as the fourth central moment divided by the square of the variance
    kurt = jnp.mean(((x - mean) / (std + 1e-10)) ** 4, axis=axis, keepdims=keepdims)

    if fisher is True:
        # Case: Normalize according to the normal distribution
        # Fisher (normal ==> 0.0)
        # Pearson (normal ==> 3.0)
        kurt = kurt - 3

    return kurt


def skewness(
    x: chex.Array,
    axis: int | None = None,
    keepdims: bool = False,
) -> chex.Array:
    """
    Computes the skewness along the specified axis.

    Args:
        x (chex.Array):
            Array of shape (?, ...)

        axis (int, optional):
            Axis or axes along which the kurtosis is computed

        keepdims (bool):
            Controls whether to keep the dimension of x

    Returns:
        chex.Array:
            Array of shape (?, ...)
            The second pearson coefficient of skewness along the specified axis
    """
    # Compute mean and standard deviation along the given axis
    mean = jnp.mean(x, axis=axis, keepdims=True)
    std = jnp.std(x, axis=axis, keepdims=True)

    # Compute the skewness as the third central moment divided by the square of the variance
    skew = jnp.mean(((x - mean) / (std + 1e-10)) ** 3, axis=axis, keepdims=keepdims)

    return skew


def value_at_risk(
    x: chex.Array,
    alpha: float = 0.05,
    axis: int | None = None,
    keepdims: bool = False,
) -> chex.Array:
    """
    Compute the value at risk (VaR) along the specified axis.

    Args:
        x (chex.Array):
            Array of shape (?, ...)

        alpha (float):
            The confidence level

        axis (int, optional):
            Axis or axes along which the value at risk (VaR) is computed

        keepdims (bool):
            Controls whether to keep the dimension of x

    Returns:
        chex.Array:
            Array of shape (?, ...)
            The value at risk (VaR) along the specified axis
    """
    # Get the package
    var = jnp.quantile(x, q=alpha, axis=axis, keepdims=keepdims)

    if axis is None and keepdims:
        # Case: Fix error from jax where the dimension of var is not the same as x
        output_shape = [1] * x.ndim
        var = jnp.broadcast_to(var, output_shape)

    return var
