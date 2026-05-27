# ruff: noqa: E741
"""
Explicit sparse cholesky solver (cpu) and differentiation via the adjoint method.
Uses a custom jax vjp hook. Allows to solve multiple forward and backward problems in parallel.
Some functions and the general structure
is taken from https://github.com/deepmodeling/jax-fem.
"""

import logging
from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
from jax import Array
from jax.flatten_util import ravel_pytree

try:
    from scipy.sparse import csc_matrix
    from sksparse.cholmod import cholesky
except ImportError:
    sksparse = None
import numpy as np

from pde import LinElasticity

logger = logging.getLogger(__name__)


def apply_bc_vec(res_vec: Array, dofs: Array, problem: LinElasticity, scale: float = 1.0) -> Array:
    """
    Apply boundary conditions to a residual vector.

    Parameters
    ----------
    res_vec : Array
        The residual vector.
    dofs : Array
        The degrees of freedom.
    problem : LinElasticity
        The problem instance containing BC definitions.
    scale : float, optional
        Scaling factor for BC values. Defaults to 1.0.

    Returns
    -------
    Array
        The residual vector with boundary conditions applied.
    """
    # version supporting only one fe mesh
    res_list = problem.unflatten_fn_sol_list(res_vec)
    sol_list = problem.unflatten_fn_sol_list(dofs)
    fe = problem.fes[0]
    res = res_list[0]
    sol = sol_list[0]
    for i in range(len(fe.node_inds_list)):
        res = res.at[fe.node_inds_list[i], fe.vec_inds_list[i]].set(
            sol[fe.node_inds_list[i], fe.vec_inds_list[i]], unique_indices=True
        )
        res = res.at[fe.node_inds_list[i], fe.vec_inds_list[i]].add(
            -fe.vals_list[i] * scale
        )

    res_list[0] = res

    return ravel_pytree(res_list)[0]


def assign_bc(dofs: Array, problem: LinElasticity) -> Array:
    """
    Assign boundary condition values to degrees of freedom.

    Parameters
    ----------
    dofs : Array
        The degrees of freedom vector.
    problem : LinElasticity
        The problem instance containing BC definitions.

    Returns
    -------
    Array
        The DOFs vector with BC values set.
    """
    # version supporting only one fe mesh
    sol_list = problem.unflatten_fn_sol_list(dofs)
    fe = problem.fes[0]
    ind = 0
    sol = sol_list[ind]
    for i in range(len(fe.node_inds_list)):
        sol = sol.at[fe.node_inds_list[i], fe.vec_inds_list[i]].set(fe.vals_list[i])
    sol_list[ind] = sol
    return ravel_pytree(sol_list)[0]


def copy_bc(dofs: Array, problem: LinElasticity) -> Array:
    """
    Extract and copy only the boundary condition values from a DOF vector.

    Parameters
    ----------
    dofs : Array
        The input degrees of freedom vector.
    problem : LinElasticity
        The problem instance containing BC definitions.

    Returns
    -------
    Array
        A zero-initialized vector with only BC values copied.
    """
    # version supporting only one fe mesh
    new_dofs = jnp.zeros_like(dofs)
    sol_list = problem.unflatten_fn_sol_list(dofs)
    new_sol_list = problem.unflatten_fn_sol_list(new_dofs)
    fe = problem.fes[0]
    ind = 0

    sol = sol_list[ind]
    new_sol = new_sol_list[ind]
    for i in range(len(fe.node_inds_list)):
        new_sol = new_sol.at[fe.node_inds_list[i], fe.vec_inds_list[i]].set(
            sol[fe.node_inds_list[i], fe.vec_inds_list[i]]
        )
    new_sol_list[ind] = new_sol

    return ravel_pytree(new_sol_list)[0]


def get_problem_structure(
    problem: LinElasticity,
) -> tuple[Array, Array, Array, int]:
    """
    Extracts the static structure (indices and BC mask) from a template problem.

    These do not change across the batch. This allows us to assemble the stiffness
    matrix on the gpu using the sparse_matvec function we pass to the bicgstab solver.

    Parameters
    ----------
    problem : LinElasticity
        Template problem instance. (The problems have the same mesh size and
        element type across a batch so it could be any of the problems.
        We use first one for convenience).

    Returns
    -------
    I_all : Array
        Row indices including BC diagonals.
    J_all : Array
        Column indices including BC diagonals.
    mask : Array
        Mask for BC rows (0.0 for BC rows, 1.0 otherwise).
    bc_dofs : Array
        Global DOF indices for BCs.
    num_dofs : int
        Total number of DOFs.
    """
    fe = problem.fes[0]
    bc_dofs_list = []

    for i in range(len(fe.node_inds_list)):
        node_inds = fe.node_inds_list[i]
        vec_inds = fe.vec_inds_list[i]
        global_dofs = node_inds * fe.vec + vec_inds + problem.offset[0]
        bc_dofs_list.append(global_dofs)

    if bc_dofs_list:
        bc_dofs = jnp.concatenate(bc_dofs_list)
        # remove duplicates to prevent diagonal summation
        bc_dofs = jnp.unique(bc_dofs)
    else:
        bc_dofs = jnp.array([], dtype=jnp.int32)

    num_dofs = problem.num_total_dofs_all_vars

    mask = jnp.ones(num_dofs)
    mask = mask.at[bc_dofs].set(0.0)

    I_existing = problem.I
    J_existing = problem.J

    I_bc = bc_dofs
    J_bc = bc_dofs

    I_all = jnp.concatenate([I_existing, I_bc])
    J_all = jnp.concatenate([J_existing, J_bc])
    indices_static = jnp.stack([I_all, J_all], axis=1)

    return indices_static, mask, bc_dofs, num_dofs


def get_Vall(
    problem: LinElasticity,
    mask: Array,
    bc_dofs: Array,
) -> Array:
    """
    Computes the values for the matrix.

    Parameters
    ----------
    problem : LinElasticity
        LinElasticity instance.
    mask : Array
        Mask for BC rows (0.0 for BC rows, 1.0 otherwise).
    bc_dofs : Array
        Global DOF indices for BCs.

    Returns
    -------
    Array
        Matrix values including BC diagonals.
    """
    # mask off diagonal rows and columns
    V_masked = problem.V * mask[problem.I] * mask[problem.J]

    V_bc = jnp.ones_like(bc_dofs, dtype=V_masked.dtype)

    V_all = jnp.concatenate([V_masked, V_bc])
    return V_all


@partial(jax.custom_vjp, nondiff_argnums=(0, 1, 3))
def sparse_cho_solve(I, J, V, n, rhs):
    K = csc_matrix((np.array(V), (np.array(I), np.array(J))), shape=(n, n))
    factor = cholesky(K)
    rhs = np.transpose(np.array(rhs))
    sol = jnp.array(factor(rhs))
    return np.transpose(sol)


def sparse_cho_solve_fwd(I, J, V, n, rhs):
    sol = sparse_cho_solve(I, J, V, n, rhs)
    return sol, (V, sol)


def sparse_cho_solve_bwd(I, J, n, res, g):
    V, sol = res
    # Adjoint solve: K^T lambda = g (K is symmetric so K^T = K)
    K = csc_matrix((np.array(V), (np.array(I), np.array(J))), shape=(n, n))
    factor = cholesky(K)
    # expects rhs shaped (N, batch_size)
    g_t = np.transpose(np.array(g))
    lambd = jnp.array(factor(g_t))
    # transpose back to (batch_size, N)
    lambd = np.transpose(lambd)
    # dL/dV_k = -lambda[I[k]] * sol[J[k]]
    grad_V = -lambd[jnp.array(I)] * sol[jnp.array(J)]
    # dL/drhs = lambda
    grad_rhs = lambd
    return grad_V, grad_rhs


sparse_cho_solve.defvjp(sparse_cho_solve_fwd, sparse_cho_solve_bwd)


def solve_parallel(problems: list[LinElasticity]) -> list[list[Array]]:
    """
    Solve multiple linear problems using a sparse cholesky solver.

    Impose Dirichlet B.C.s with the "row elimination method".
    See https://github.com/deepmodeling/jax-fem/blob/main/jax_fem/solver.py for details.

    Parameters
    ----------
    problems : list of LinElasticity
        LinElasticitys to solve.

    Returns
    -------
    list
        Solutions for each problem.
    """
    dofs_list = [jnp.zeros(p.num_total_dofs_all_vars) for p in problems]
    dofs_batch = jnp.stack(dofs_list)

    # Setup Static Structure (assumes all problems have same topology)
    template_problem = problems[0]

    indices_static, mask_static, bc_dofs_static, num_dofs = get_problem_structure(
        template_problem
    )

    res_vecs = []
    V_values_list = []

    for i, p in enumerate(problems):
        # Update problem state
        sol_list = p.unflatten_fn_sol_list(dofs_batch[i])
        res_list = p.newton_update(sol_list)  # Updates p.V internally

        # Process Residual
        res_vec = ravel_pytree(res_list)[0]
        res_vec = apply_bc_vec(res_vec, dofs_batch[i], p)
        res_vecs.append(res_vec)

        # Process Matrix Values
        V_vals = get_Vall(p, mask_static, bc_dofs_static)
        V_values_list.append(V_vals)

    V_batch = jnp.stack(V_values_list)
    res_batch = jnp.stack(res_vecs, axis=0)

    b_batch = -res_batch

    solutions = sparse_cho_solve(
        indices_static[:, 0], indices_static[:, 1], V_batch[0], num_dofs, b_batch
    )

    final_solutions = []
    for i, p in enumerate(problems):
        final_solutions.append(p.unflatten_fn_sol_list(solutions[i]))
    return final_solutions


def parallel_implicit_vjp(
    problems: list[LinElasticity],
    sol_list_batch: list[list[Array]],
    batched_params: Array,
    v_list_batch: list[list[Array]],
) -> tuple[Array]:
    """
    Compute VJP using the adjoint method for multiple problems in parallel.

    Solves A^T * lambda = -v, then computes lambda^T * (dR/dp).
    See https://github.com/deepmodeling/jax-fem/blob/main/jax_fem/solver.py for details.

    Parameters
    ----------
    problems : list of LinElasticity
        List of LinElasticity instances.
    sol_list_batch : list of list of Array
        Batched solutions for each problem.
    batched_params : Array
        Batched parameters for each problem.
    v_list_batch : list of list of Array
        Batched cotangents for each problem.

    Returns
    -------
    tuple
        Tuple containing batched gradients with respect to parameters.
    """
    # Setup Static Structure (assumes all problems have same topology)
    template_problem = problems[0]
    indices_static, mask_static, bc_dofs_static, num_dofs = get_problem_structure(
        template_problem
    )
    # We need the values of A to solve the adjoint system.
    V_vals_list = []

    for i, p in enumerate(problems):
        # Set params for this specific problem
        p_params = jax.tree.map(lambda x, i=i: x[i], batched_params)
        p.set_params(p_params)

        # Re-evaluate residual/Jacobian at the converged solution
        sol = sol_list_batch[i]
        p.newton_update(sol)  # Updates p.V

        # Extract matrix values
        V_all = get_Vall(p, mask_static, bc_dofs_static)
        # V_vals = get_Vs(p, mask_static, bc_dofs_static)
        V_vals_list.append(V_all)

    V_batch = jnp.stack(V_vals_list)
    # Prepare RHS for Adjoint System: b = -v
    # v_list_batch contains the cotangents (dL/du)
    v_vecs = []
    for i, _ in enumerate(problems):
        v_vec = ravel_pytree(v_list_batch[i])[0]
        v_vecs.append(v_vec)

    v_batch = jnp.stack(v_vecs)
    v_batch = -v_batch

    # Solve Adjoint System: A^T * lambda = -v
    # Use _core_solver but swap I_static and J_static to multiply by A^T
    # Transpose for Adjoint: swap col/row in indices
    # indices_static is (NSE, 2). I is col 0, J is col 1.
    # We want T_indices = [J, I]
    adjoint_indices = jnp.stack([indices_static[:, 1], indices_static[:, 0]], axis=1)

    adjoint_sol_batch = sparse_cho_solve(
        adjoint_indices[:, 0], adjoint_indices[:, 1], V_batch[0], num_dofs, v_batch
    )
    # Compute VJP: lambda^T * (dR/dp)
    grads_list = []

    for i, p in enumerate(problems):
        p_params = jax.tree.map(lambda x, i=i: x[i], batched_params)
        sol = sol_list_batch[i]
        lambda_vec = adjoint_sol_batch[i]

        # Define restricted residual function R(p) for fixed u
        def residual_wrt_params(params, p=p, sol=sol):
            p.set_params(params)
            res_list = p.compute_residual(sol)
            res_vec = ravel_pytree(res_list)[0]

            # Apply BCs to residual (must match forward pass logic)
            dofs = ravel_pytree(sol)[0]
            res_vec = apply_bc_vec(res_vec, dofs, p)
            return res_vec

        # Compute vector-jacobian product: lambda^T * (dR/dp)
        _, vjp_fn = jax.vjp(residual_wrt_params, p_params)
        grad_p = vjp_fn(lambda_vec)[0]
        grads_list.append(grad_p)

    # Stack gradients to match batched_params structure
    batched_grads = jax.tree.map(lambda *xs: jnp.stack(xs), *grads_list)

    return (batched_grads,)


def parallel_ad_wrapper(problems: list[LinElasticity]) -> Callable[[Array], Array]:
    """
    Automatic differentiation wrapper for the parallel solver.

    Returns a function that takes batched parameters and returns batched solutions,
    supporting backpropagation via the adjoint method.
    See https://github.com/deepmodeling/jax-fem/blob/main/jax_fem/solver.py for details.

    Parameters
    ----------
    problems : list of LinElasticity
        List of LinElasticity instances.

    Returns
    -------
    callable
        fwd_pred: A function f(batched_params) -> stacked_solutions
    """

    def _fwd_pred_impl(batched_params):
        # Update all problems with new parameters
        for i, p in enumerate(problems):
            p_params = jax.tree.map(lambda x, i=i: x[i], batched_params)
            p.set_params(p_params)

        # Run Parallel Solver
        # Returns List[List[Array]]
        sol_list_batch = solve_parallel(problems)

        # Stack solutions for output (Batch, NumDofs, ...)
        # Assuming all problems have same number of fields
        num_fields = len(sol_list_batch[0])
        stacked_sol = []
        for field_idx in range(num_fields):
            field_batch = jnp.stack([s[field_idx] for s in sol_list_batch])
            stacked_sol.append(field_batch)

        return tuple(stacked_sol)

    @jax.custom_vjp
    def fwd_pred(batched_params):
        return _fwd_pred_impl(batched_params)

    def f_fwd(batched_params):
        # We need to tell JAX what shapes to expect from the callback.
        # We peek at the first problem to determine shapes.
        sol_size = problems[0].num_total_dofs_all_vars
        sol_shape = (sol_size // 3, 3)
        batch_size = len(problems)

        # Create ShapeDtypeStructs: List of (BatchSize, *FieldShape)
        result_shape_dtypes = (
            jax.ShapeDtypeStruct(shape=(batch_size,) + sol_shape, dtype=jnp.float64),
        )

        # pure_callback executes _fwd_pred_impl on the host (Python),
        # passing concrete values even if f_fwd was called with Tracers.
        sol_tuple = jax.pure_callback(
            _fwd_pred_impl, result_shape_dtypes, batched_params
        )

        # Save params and solution for backward pass
        return sol_tuple, (batched_params, sol_tuple)

    def f_bwd(res, v_tuple):
        batched_params, sol_tuple = res

        # Unstack solutions back to list format for implicit_vjp
        batch_size = len(problems)
        sol_list_batch = []
        for i in range(batch_size):
            s_list = [field[i] for field in sol_tuple]
            sol_list_batch.append(s_list)

        # Unstack cotangents (v_tuple) similarly
        v_list_batch = []
        for i in range(batch_size):
            v_list = [field[i] for field in v_tuple]
            v_list_batch.append(v_list)

        return parallel_implicit_vjp(
            problems, sol_list_batch, batched_params, v_list_batch
        )

    fwd_pred.defvjp(f_fwd, f_bwd)
    return fwd_pred
