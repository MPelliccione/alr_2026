"""
Utilities for running parallel finite element simulations and computing gradients.

This module provides wrappers for performing simulation steps on single problems, multiple
problems in parallel, including support for automatic differentiation through
the simulation via the adjoint method.
"""
from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax import Array

from explicit_solver import parallel_ad_wrapper, solve_parallel
from pde import LinElasticity


def parallel_simulation_step(
    problems: list[LinElasticity],
    loads: list[Array],
) -> list[list[Array]]:
    """
    Perform a single simulation step for multiple problems with the given loads.

    Parameters
    ----------
    problems : list of LinElasticity instances.
        List of problem instances to simulate.
    loads : list of Array
        List of loads, one for each problem.

    Returns
    -------
    list of list of Array
        The solution list for each problem after the simulation step.
    """
    for problem, load in zip(problems, loads, strict=True):
        problem.set_params(load)
    sol_list = solve_parallel(
        problems,
    )
    return sol_list


def single_simulation_step(
    problem: LinElasticity,
    load: Array,
) -> Array:
    """
    Perform a single simulation step for one problem
    with the given loads.

    Parameters
    ----------
    problem : LinElasticity instance.
    loads : Array
        Forces for problem.

    Returns
    -------
    Array
        The solution array after the simulation step.
    """
    problem.set_params(load)
    sol_list = solve_parallel(
        [problem],
    )
    return sol_list[0][0]


def single_simulation_and_grad_step(
    problem: LinElasticity,
    load: Array,
    objective_fn: Callable[[Array, Array], Array],
    target: Array,
) -> tuple[Array, Array, Array]:
    """
    Performs a simulation step for one problem and computes the gradient w.r.t.
    the given loads.

    Parameters
    ----------
    problem : LinElasticity
        LinElasticity instance to simulate.
    load : Array
        Load for problem.
    objective_fn : Callable
        Objective function for gradient calculation.
    target : Array
        Target field for objective function.

    Returns
    -------
    sol : Array
        The simulation solution.
    grad : Array
        The gradient of the loss with respect to the loads.
    loss : Array
        The objective value.
    """
    fwd_pred = parallel_ad_wrapper([problem])
    loads = jnp.expand_dims(load, axis=0)
    # target = jnp.expand_dims(target, axis=0)

    def composed_fn(loads):
        pred = fwd_pred(loads)
        batch_loss = objective_fn(pred, target)
        total_loss = jnp.sum(batch_loss)
        return total_loss, (pred)

    # also a tuple of stacked gradients
    (tot_loss, (sols)), load_grads = jax.value_and_grad(
        composed_fn, has_aux=True
    )(loads)

    return jnp.squeeze(sols[0]), jnp.squeeze(load_grads), tot_loss


def parallel_simulation_and_grad_step(
    problems: list[LinElasticity],
    loadings: list[Array],
    objective_fn: Callable[[Array, Array], Array],
    target: Array,
) -> tuple[Array, Array, Array, Array]:
    """
    Perform a single simulation step with the given loading and compute
    the gradients w.r.t. the nodal loads in parallel.

    Parameters
    ----------
    problems : list of LinElasticity
        List of problem instances to simulate.
    loadings : list of Array
        List of loadings corresponding to each problem.
    objective_fn : Callable
        Objective function for gradient calculation.
    target : Array
        Target field for objective function.

    Returns
    -------
    sols : Array
        The batched solutions.
    load_grads : Array
        The batched gradients with respect to loadings.
    tot_loss : Array
        The total summed objective value.
    batch_loss_vals : Array
        The individual objective values for each item in the batch.
    """
    fwd_pred = parallel_ad_wrapper(problems)
    loadings = jnp.stack(loadings, axis=0)

    def composed_fn(loadings):
        preds = fwd_pred(loadings)
        batch_losses = objective_fn(preds, target)
        total_loss = jnp.sum(batch_losses)
        return total_loss, (preds, batch_losses)

    # also a tuple of stacked gradients
    (tot_loss, (sols, batch_loss_vals)), load_grads = jax.value_and_grad(
        composed_fn, has_aux=True
    )(loadings)

    return sols, load_grads, tot_loss, batch_loss_vals
