"""
This file contains a few (hopefully helpful) examples.
"""

import jax
import jax.numpy as jnp
from jax_fem.generate_mesh import Mesh, box_mesh

from dirichlet_bc import dirichlet_val, left_x1, left_x2, right_x1, right_x2
from init_pde import initialize_empty_sim, initialize_empty_sims
from mse import mean_squared_displacement_loss
from pde import LinElasticityDisc
from postprocessing import save_data_to_vtk, save_sol_to_vtk
from simulation import (
    parallel_simulation_and_grad_step,
    single_simulation_and_grad_step,
    single_simulation_step,
)
from target_shapes import generate_n_target_shapes, generate_target_shape
from utils import get_flat_vector_shape, get_geometric_shape, get_zero_forces


def manual_setup_and_simulation() -> None:
    """
    This is an example of a manual pde problem and force setup.
    The steps are as follows:
    1. Define domain dimensions, material properties and boundary conditions
    2. Define mesh for domain (here always hexahedral 8 node mesh -> 'HEX8')
    3. Instantiate a problem instance (here LinIsoElasticity) with the information.
    -> In reality, use convenience functions init_pde.initialize_empty_sim() and init_pde.intitialize_empty_sims()

    To set up discrete force fields, we need to specify a force vector at each node of the mesh.
    For convenience we can call the creation function from src/utils.py and reshape accordingly.
    Then we prescribe a small force at one node close to the center of the top of the plate in negative z direction.
    """
    # Material properties
    E = 200
    nu = 0.3
    mu = E / (2.0 * (1.0 + nu))
    lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
    # Domain dimensions
    Lx, Ly, Lz = 140, 140, 2
    # No of nodes per dimension
    Nx, Ny, Nz = 28, 28, 1
    # Dirichlet boundary conditions
    num_dif_boundary_location_fns = 4
    location_fns_dirichlet = (
        [left_x1] * 3 + [right_x1] * 3 + [left_x2] * 3 + [right_x2] * 3
    )
    dirichlet_val_fns = [dirichlet_val] * len(location_fns_dirichlet)
    vecs = [0, 1, 2] * num_dif_boundary_location_fns
    dirichlet_bc_info = [location_fns_dirichlet, vecs, dirichlet_val_fns]
    # Neumann boundary conditions (there are none when prescribing nodal forces directly)
    location_fns_neumann = []
    # Mesh setup
    meshio_mesh = box_mesh(Nx=Nx, Ny=Ny, Nz=Nz, domain_x=Lx, domain_y=Ly, domain_z=Lz)
    cell_type = "hexahedron"
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type], ele_type="HEX8")
    # Instantiate problem instance
    problem = LinElasticityDisc(
        mesh=mesh,
        vec=3,
        dim=3,
        ele_type="HEX8",
        dirichlet_bc_info=dirichlet_bc_info,
        location_fns=location_fns_neumann,
        additional_info=(mu, lmbda),
    )
    ## FORCE SETUOP
    # get forces/loads
    forces = get_zero_forces()
    geometric_forces = get_geometric_shape(forces)
    # Set a force of -1.0 kN in z-direction (pushing down from the top) at one node.
    geometric_forces = geometric_forces.at[14, 14, 1, 2].set(-1.0)
    # reshape again, to match simulation shape
    flat_forces = get_flat_vector_shape(geometric_forces)
    # Now we can simulate one forward problem
    sol = single_simulation_step(problem, flat_forces)
    print("Simulation done.")
    # convert solution and forces to vtk file
    save_sol_to_vtk(sol, "manual_setup_sol.vtk")
    save_data_to_vtk(flat_forces, "manual_setup_forces.vtk")



def single_examples() -> None:
    """
    The framework supports single simulations, i.e. for individual problem instances
    (and their differentiation).
    """
    problem, forces = initialize_empty_sim(continuous_forces=False)
    # -> we have one force field shaped (num_nodes, 3)
    geometric_forces = get_geometric_shape(forces)
    # Set a force of -1.0 kN in z-direction (pushing down from the top) at one node.
    geometric_forces = geometric_forces.at[14, 14, 1, 2].set(-1.0)
    # reshape again, to match simulation shape
    flat_forces = get_flat_vector_shape(geometric_forces)
    # we now create a simple target letter shape
    # jax rng key
    key = jax.random.key(seed=1)
    target_letter, _ = generate_target_shape(key)
    save_data_to_vtk(target_letter, "single_example_target.vtk")
    # Now we call the forward and backward solver
    # -> we get the solution, the derivative of an objective function w.r.t. the forces and the loss
    # we use a simple mean squared distance between forces as loss
    sol, grad, loss = single_simulation_and_grad_step(problem, flat_forces, mean_squared_displacement_loss, target_letter)
    print(f"The mse is {float(loss)}")
    save_data_to_vtk(grad, "single_example_grad.vtk")



def parallel_examples() -> None:
    """
    The framework also supports parallel simulations to work with batched inputs,
    common in ML contexts. Differentiating them is also possible.
    """
    problems, forces = initialize_empty_sims(3, continuous_forces=False)
    # -> we now have three force fields, i.e. shape (3, num_nodes, 3)
    geometric_forces = get_geometric_shape(forces)
    # Set a force of -1.0 kN in z-direction (pushing down from the top) at one node for all force fields.
    geometric_forces = geometric_forces.at[:, 14, 14, 1, 2].set(-1.0)
    # reshape again, to match simulation shape
    flat_forces = get_flat_vector_shape(geometric_forces)
    # we now create 3 simple random target letter shapes
    # create a jax rng key
    key = jax.random.key(seed=42)
    targets = generate_n_target_shapes(key, n=3)
    # unzip as we don't need configs
    letters, _ = zip(*targets, strict=True)
    letters = jnp.stack(letters, axis=0)
    # Call forward and backward solver on all problems at once
    # Returns solutions, gradients, total losses (summed) and individual losses
    sols, grads, total_loss, batch_losses = parallel_simulation_and_grad_step(problems, flat_forces, mean_squared_displacement_loss, letters)
    print(f"The total mse is {float(total_loss)}")
    print(f"The individual mse are {batch_losses}")


if __name__ == "__main__":
    # Manual example
    manual_setup_and_simulation()
    # Convenience examples with differentiation
    # Single
    single_examples()
    # Multi problem
    parallel_examples()
