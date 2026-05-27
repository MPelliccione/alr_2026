"""
Location and value functions for Dirichlet boundary conditions.
"""
import jax.numpy as jnp


def left_x1(point):
    return jnp.isclose(point[0], 0.0, atol=1e-5)


def right_x1(point):
    return jnp.isclose(point[0], 140.0, atol=1e-5)


def left_x2(point):
    return jnp.isclose(point[1], 140.0, atol=1e-5)


def right_x2(point):
    return jnp.isclose(point[1], 0.0, atol=1e-5)


def top_x3(point):
    return jnp.isclose(point[2], 2.0, atol=1e-5)


def bottom_x3(point):
    return jnp.isclose(point[2], 0.0, atol=1e-5)


def dirichlet_val(point):
    return 0.0
