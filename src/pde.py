"""
Implementation of linear isotropic elasticity problems.

This module provides the LinIsoElasticity class, which extends the base
Problem class from jax-fem to handle 3D linear elastic structural simulations
with support for differentiable nodal loads.
"""
from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax import Array
from jax_fem.problem import Problem


class LinElasticityDisc(Problem):
    """
    Linear elasticity model with discrete nodal force application
    for finite element analysis.

    This class implements the constitutive equations for linear elasticity
    and handles the application of external nodal loads in the residual
    computation.
    """

    def custom_init(self, *additional_info) -> None:
        """
        Initialize material properties.

        Parameters
        ----------
        *additional_info : tuple
            Expects (mu, lmbda) representing Lame parameters.
        """
        self.mu, self.lmbda = additional_info

    def get_tensor_map(self) -> Callable[[Array], Array]:
        """
        Define the stress-strain relationship.

        Returns
        -------
        callable
            A function mapping displacement gradient to Cauchy stress tensor.
        """
        def stress(u_grad):
            epsilon = 0.5 * (u_grad + u_grad.T)
            sigma = (
                self.lmbda * jnp.trace(epsilon) * jnp.eye(self.dim)
                + 2 * self.mu * epsilon
            )
            return sigma

        return stress

    def _apply_point_load(self, res_list: list[Array]) -> list[Array]:
        """
        Helper to subtract external force from the internal residual.

        R = F_int - F_ext

        Parameters
        ----------
        res_list : list of Array
            The list of residual vectors.

        Returns
        -------
        list of Array
            Residual list with external forces applied.
        """
        # Subtract the force vector because we have K*u = F
        res_list[0] = res_list[0] - self.nodal_loads
        return res_list

    def compute_residual(self, sol_list: list[Array]) -> list[Array]:
        """
        Compute the residual of the linear system.

        Parameters
        ----------
        sol_list : list of Array
            Current solution vectors.

        Returns
        -------
        list of Array
            The residual vector list.
        """
        res_list = super().compute_residual(sol_list)
        return self._apply_point_load(res_list)

    def newton_update(self, sol_list: list[Array]) -> list[Array]:
        """
        Compute the update step for iterative solvers.

        Parameters
        ----------
        sol_list : list of Array
            Current solution vectors.

        Returns
        -------
        list of Array
            The update vector list.
        """
        res_list = super().newton_update(sol_list)
        return self._apply_point_load(res_list)

    # need this to make loads differentiable
    def set_params(self, params: Array) -> None:
        """
        Set the differentiable parameters for the problem.

        Parameters
        ----------
        params : Array
            The nodal loads to apply to the system.
        """
        self.nodal_loads = params


class LinElasticityCont(Problem):
    """
    See the set_params() method for details on how to specify forces.
    """
    def custom_init(self, *additional_info) -> None:
        """
        Initialize material properties.

        Parameters
        ----------
        *additional_info : tuple
            Expects (mu, lmbda) representing Lame parameters.
        """
        self.mu, self.lmbda = additional_info

    def get_tensor_map(self) -> Callable[[Array], Array]:
        """
        Define the stress-strain relationship.

        Returns
        -------
        callable
            A function mapping displacement gradient to Cauchy stress tensor.
        """
        def stress(u_grad):
            epsilon = 0.5 * (u_grad + u_grad.T)
            sigma = (
                self.lmbda * jnp.trace(epsilon) * jnp.eye(self.dim)
                + 2 * self.mu * epsilon
            )
            return sigma

        return stress

    def _apply_point_load(self, res_list: list[Array]) -> list[Array]:
        """
        Helper to subtract external force from the internal residual.

        R = F_int - F_ext

        Parameters
        ----------
        res_list : list of Array
            The list of residual vectors.

        Returns
        -------
        list of Array
            Residual list with external forces applied.
        """
        # Subtract the force vector because we have K*u = F
        res_list[0] = res_list[0] - self.nodal_loads
        return res_list

    def compute_residual(self, sol_list: list[Array]) -> list[Array]:
        """
        Compute the residual of the linear system.

        Parameters
        ----------
        sol_list : list of Array
            Current solution vectors.

        Returns
        -------
        list of Array
            The residual vector list.
        """
        res_list = super().compute_residual(sol_list)
        return self._apply_point_load(res_list)

    def newton_update(self, sol_list: list[Array]) -> list[Array]:
        """
        Compute the update step for iterative solvers.

        Parameters
        ----------
        sol_list : list of Array
            Current solution vectors.

        Returns
        -------
        list of Array
            The update vector list.
        """
        res_list = super().newton_update(sol_list)
        return self._apply_point_load(res_list)

    def _apply_forces(self, nodes: Array, forces: Array) -> Array:
        """
        Set self.nodal_loads which then update the residual
        during each Newton update.
        """
        def scan_body(accumulated_loads: Array, force: Array) -> tuple[Array, Array]:
            center = force[0:3]
            f_vec = force[3:6]
            rad = force[6]

            # Find nodes within the cylindrical area
            dist_sq = jnp.square(nodes[:, 0] - center[0]) + jnp.square(nodes[:, 1] - center[1])
            in_area = (dist_sq <= jnp.square(rad)) & jnp.isclose(nodes[:, 2], center[2])

            # Count nodes in area and distribute force evenly
            node_count = jnp.sum(in_area)
            # Prevent division by zero if an area contains no nodes
            node_count_safe = jnp.maximum(1, node_count)
            distributed_f = f_vec / node_count_safe

            # Apply forces only to nodes inside the area
            loads_to_add = jnp.where(in_area[:, None], distributed_f, 0.0)

            return accumulated_loads + loads_to_add, None

        initial_loads = jnp.zeros_like(nodes, dtype=jnp.float64)
        final_loads, _ = jax.lax.scan(scan_body, initial_loads, forces)
        return final_loads

    def set_params(self, params: Array) -> None:
        """
        Set the differentiable forces for the problem.
        Continuous forces should have the shape
        (num_forces, 7), where the second dimension is
        an array of [x, y, z, f_x, f_y, f_z, radius].
        The z-coordinates must be in {0.0, 2.0}, i.e. the top or bottom surface.
        The radius determines the size of the circular area around the center point (x, y, z)
        within which the nodes, where the respective force is applied to, lie.
        The total force vector f = (f_x, f_y, f_z)^T is then distributed evenly over the nodes
        inside the force area.
        Overlapping areas and therefore nodal forces are summed.

        Example
        -------
        params = [[15.0, 100.0, 2.0, 0.0, -0.9, -10.0, 15.0],
                [33.0, 71.0, 0.0, 1.9, -4.3, 5.0, 7.0],
                [117.0, 45.0, 2.0, -4.6, 2.2, -12.0, 30.0]]

        Parameters
        ----------
        params : Array
            The nodal loads to apply to the system.
        """
        self.nodal_loads = self._apply_forces(self.mesh[0].points, params)

LinElasticity = LinElasticityDisc | LinElasticityCont
