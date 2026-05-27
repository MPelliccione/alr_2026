"""
Common utility functions for array shape conversion
and similar nicities.
"""
import jax.numpy as jnp
import meshio
import numpy as np
from jax import Array


def get_geometric_shape(arr: Array) -> Array:
    """
    Converts array with flat vector shape to the 3d geometric shape
    of the plate (29, 29, 2, 3) which is (num_nodes_x, num_nodes_y, num_nodes_z, 3).
    Works for single and batched arrays.
    Assumes first dimension is batch dimension for 3d arrays.

    Parameters
    ----------
    arr : Array with flat vector shape (num_nodes, 3) or batched flat vector shape (B, num_nodes, 3).

    Returns
    -------
    Array of geometric shape.
    """
    assert arr.ndim in (2, 3), "Array has neither flat vector shape nor batched flat vector shape!"
    if arr.ndim == 2:
        return jnp.reshape(arr, (29, 29, 2, 3))
    else:
        return jnp.reshape(arr, (arr.shape[0], 29, 29, 2, 3))


def get_flat_vector_shape(arr: Array) -> Array:
    """
    Converts array with geometric shape (29, 29, 2, 3) to flat vector shape.
    Works for batched arrays, where first axis is assumed to be the batch dimension.

    Parameters
    ----------
    arr : Array with geometric shape (29, 29, 2, 3) or batched (B, 29, 29, 2, 3).

    Returns
    -------
    Array with flat vector shape.
    """
    assert arr.ndim in  (4, 5), "Array has neither geometric nor batched geometric shape!"
    if arr.ndim == 4:
        return jnp.reshape(arr, (-1, 3))
    if arr.ndim == 5:
        return jnp.reshape(arr, (arr.shape[0], -1, 3))


def get_zero_forces() -> Array:
    """
    Creates a single zero force field.

    Returns
    -------
    Array of zeros of shape (num_nodes, 3).
    """
    return jnp.zeros((1682, 3), dtype=jnp.float64)


def get_n_zero_forces(n: int) -> Array:
    """
    Creates n zero force field.

    Parameters
    ----------
    n : Integer number of force fields.

    Returns
    -------
    Array of zeros of shape (n, num_nodes, 3).
    """
    return jnp.zeros((n, 1682, 3), dtype=jnp.float64)


def generate_box_mesh(
    Nx: int = 28,
    Ny: int = 28,
    Nz: int = 1,
    domain_x: int = 140,
    domain_y: int = 140,
    domain_z: int = 2
    ) -> meshio.Mesh:
    """
    Generate a hexagonal mesh with 8 quadrature points -> HEX8.

    Parameters
    ----------
    Nx : Integer number of nodes in x-direction.
    Ny : Integer number of nodes in y-direction.
    Nz : Integer number of nodes in z-direction.
    domain_x : Integer length of domain in x-direction.
    domain_y : Integer length of domain in y-direction.
    domain_z : Integer length of domain in z-direction.

    Returns
    -------
    meshio.Mesh instance of domain.
    """
    dim = 3
    x = np.linspace(0, domain_x, Nx + 1)
    y = np.linspace(0, domain_y, Ny + 1)
    z = np.linspace(0, domain_z, Nz + 1)
    xv, yv, zv = np.meshgrid(x, y, z, indexing="ij")
    points_xyz = np.stack((xv, yv, zv), axis=dim)
    points = points_xyz.reshape(-1, dim)
    points_inds = np.arange(len(points))
    points_inds_xyz = points_inds.reshape(Nx + 1, Ny + 1, Nz + 1)
    inds1 = points_inds_xyz[:-1, :-1, :-1]
    inds2 = points_inds_xyz[1:, :-1, :-1]
    inds3 = points_inds_xyz[1:, 1:, :-1]
    inds4 = points_inds_xyz[:-1, 1:, :-1]
    inds5 = points_inds_xyz[:-1, :-1, 1:]
    inds6 = points_inds_xyz[1:, :-1, 1:]
    inds7 = points_inds_xyz[1:, 1:, 1:]
    inds8 = points_inds_xyz[:-1, 1:, 1:]
    cells = np.stack(
        (inds1, inds2, inds3, inds4, inds5, inds6, inds7, inds8), axis=dim
    ).reshape(-1, 8)
    out_mesh = meshio.Mesh(points=points, cells={"hexahedron": cells})
    return out_mesh
