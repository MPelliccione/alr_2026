"""
Utilities for initializing 3D linear elasticity finite element simulations.

This module provides functions to generate meshes and initialize problem
instances (LinElasticity) with pre-defined boundary conditions and load
fields, supporting both single and batch initializations.
"""
import copy

import jax.numpy as jnp
import meshio
from jax import Array
from jax_fem.generate_mesh import Mesh, box_mesh, get_meshio_cell_type

from dirichlet_bc import dirichlet_val, left_x1, left_x2, right_x1, right_x2
from pde import LinElasticity, LinElasticityCont, LinElasticityDisc


def get_mesh_from_size_and_type(
    mesh_size: int = 5,
    Lx: int = 140,
    Ly: int = 140,
    Lz: int = 2,
    Nz: int = 1,
) -> tuple[meshio.Mesh, Mesh, tuple[int, ...], tuple[int, ...]]:
    """
    Get the different mesh representations from the given mesh attributes and dimensions.

    Parameters
    ----------
    mesh_size : int, optional
        Size of the elements. Defaults to 5.
    Lx : int, optional
        Length of the domain in x-direction. Defaults to 140.
    Ly : int, optional
        Length of the domain in y-direction. Defaults to 140.
    Lz : int, optional
        Length of the domain in z-direction. Defaults to 2.
    Nz : int, optional
        Number of elements in z-direction. Defaults to 1.

    Returns
    -------
    tuple
        A tuple containing:
        - meshio_mesh: The meshio mesh object.
        - mesh: The jax-fem Mesh object.
        - model_dims: Tuple of model dimensions (Lx, Ly, Lz).
        - num_elems_per_dir: Tuple of number of elements per direction.
    """
    Nx = Ny = int(Ly // mesh_size)
    meshio_mesh = box_mesh(Nx, Ny, Nz, Lx, Ly, Lz)

    cell_type = get_meshio_cell_type("HEX8")
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type], ele_type="HEX8")
    model_dims = (Lx, Ly, Lz)
    num_elems_per_dir = (
        Lx // mesh_size,
        Ly // mesh_size,
        Nz,
    )
    return (meshio_mesh, mesh, model_dims, num_elems_per_dir)

def initialize_empty_sim(
    continuous_forces: bool, youngs_mod: float = 200.0, poisson_ratio: float = 0.3
) -> tuple[LinElasticity, Array]:
    """
    Initialize an empty Problem with initial loads for target shape optimization.

    Parameters
    ----------
    continuous_forces: bool
        Whether continuous forces are used. If not, using discrete forces.
    youngs_mod : float, optional
        Young's modulus of the material. Defaults to 200.0.
    poisson_ratio : float, optional
        Poisson's ratio of the material. Defaults to 0.3.

    Returns
    -------
    tuple
        A tuple containing:
        - problem: A LinElasticity instance.
        - loadings: The initial zero-initialized load field.
    """
    mu = youngs_mod / (2.0 * (1.0 + poisson_ratio))
    lmbda = youngs_mod * poisson_ratio / ((1 + poisson_ratio) * (1 - 2 * poisson_ratio))
    additional_info = (mu, lmbda)

    # Model dimensions and mesh
    _, mesh, _, num_elems_per_dir = get_mesh_from_size_and_type()
    Nx, Ny, Nz = num_elems_per_dir
    ele_type = "HEX8"

    # set dirichlet boundary info
    num_dif_bnd_location_fns = 4
    location_fns_dirichlet = (
        [left_x1] * 3 + [right_x1] * 3 + [left_x2] * 3 + [right_x2] * 3
    )
    dirichlet_val_fns = [dirichlet_val] * len(location_fns_dirichlet)
    vecs = [0, 1, 2] * num_dif_bnd_location_fns
    dirichlet_bc_info = [location_fns_dirichlet, vecs, dirichlet_val_fns]
    # location functions
    location_fns_neumann = []
    num_nodes = int((Nx + 1) * (Ny + 1) * (Nz + 1))
    if continuous_forces:
        problem = LinElasticityCont(
            mesh,
            vec=3,
            dim=3,
            ele_type=ele_type,
            dirichlet_bc_info=dirichlet_bc_info,
            location_fns=location_fns_neumann,
            additional_info=additional_info,
        )
        loadings = jnp.zeros((3, 7), dtype=jnp.float64)
    else:
        problem = LinElasticityDisc(
            mesh,
            vec=3,
            dim=3,
            ele_type=ele_type,
            dirichlet_bc_info=dirichlet_bc_info,
            location_fns=location_fns_neumann,
            additional_info=additional_info,
        )
        loadings = jnp.zeros((num_nodes, 3), dtype=jnp.float64)
    return problem, loadings


def initialize_empty_sims(
    n: int, continuous_forces: bool, youngs_mod: float = 200.0, poisson_ratio: float = 0.3
) -> tuple[list[LinElasticity], Array]:
    """
    Initialize a set of empty Problems and initial load fields for target shape optimization.

    Parameters
    ----------
    n : int
        Number of instances to create.
    continuous_forces: bool
        Whether continuous forces are used. If not, using discrete forces.
    youngs_mod : float, optional
        Young's modulus of the material. Defaults to 200.0.
    poisson_ratio : float, optional
        Poisson's ratio of the material. Defaults to 0.3.

    Returns
    -------
    tuple
        A tuple containing:
        - problems: List of n LinElasticity instances.
        - loadings: Array of n initial load fields.
    """
    mu = youngs_mod / (2.0 * (1.0 + poisson_ratio))
    lmbda = youngs_mod * poisson_ratio / ((1 + poisson_ratio) * (1 - 2 * poisson_ratio))
    additional_info = (mu, lmbda)

    # Model dimensions and mesh
    _, mesh, _, num_elems_per_dir = get_mesh_from_size_and_type()
    Nx, Ny, Nz = num_elems_per_dir
    ele_type = "HEX8"

    # set dirichlet boundary info
    num_dif_bnd_location_fns = 4
    location_fns_dirichlet = (
        [left_x1] * 3 + [right_x1] * 3 + [left_x2] * 3 + [right_x2] * 3
    )
    dirichlet_val_fns = [dirichlet_val] * len(location_fns_dirichlet)
    vecs = [0, 1, 2] * num_dif_bnd_location_fns
    dirichlet_bc_info = [location_fns_dirichlet, vecs, dirichlet_val_fns]
    # location functions
    location_fns_neumann = []
    num_nodes = int((Nx + 1) * (Ny + 1) * (Nz + 1))
    if continuous_forces:
        problem = LinElasticityCont(
            mesh,
            vec=3,
            dim=3,
            ele_type=ele_type,
            dirichlet_bc_info=dirichlet_bc_info,
            location_fns=location_fns_neumann,
            additional_info=additional_info,
        )
        loadings = jnp.zeros((n, 3, 7), dtype=jnp.float64)
    else:
        problem = LinElasticityDisc(
            mesh,
            vec=3,
            dim=3,
            ele_type=ele_type,
            dirichlet_bc_info=dirichlet_bc_info,
            location_fns=location_fns_neumann,
            additional_info=additional_info,
        )
        loadings = jnp.zeros((n, num_nodes, 3), dtype=jnp.float64)
    problems = [problem] + [copy.deepcopy(problem) for _ in range(n - 1)]
    return problems, loadings
