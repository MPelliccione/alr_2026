"""
Simple MSE implementation.
"""
import jax.numpy as jnp
from jax import Array


def mean_squared_displacement_loss(pred: tuple[Array], target: Array) -> Array:
    """
    Computes the mean squared error per batch item.

    Note, that predictions need to be passed as tuple due to internal implementation details.
    Then index the prediction array as shown below.

    Parameters
    ----------
    pred : tuple of an array of predicted displacement function u(x,y,z), shaped (B, num_nodes, 3).
    target: array of target displacement function u'(x,y,z), shaped (B, num_nodes, 3).

    Returns
    -------
    Array
        A 1D array of shape (B,) with mean squared error per batch item.
    """
    diff_squared = jnp.square(target - pred[0])
    mean_loss = jnp.mean(diff_squared, axis=(-1, -2))
    return mean_loss
