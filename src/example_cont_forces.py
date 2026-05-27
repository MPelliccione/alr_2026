"""
This is an example of how the continuous forces can be set and used in the simulation.
"""
import jax
import jax.numpy as jnp
from jax import Array

from init_pde import initialize_empty_sim
from mse import mean_squared_displacement_loss
from simulation import single_simulation_and_grad_step
from target_shapes import generate_target_shape


def get_random_forces(key: Array, n: int = 3) -> Array:
    """
    Generate random force configurations for the example.
    """
    pkey, zkey, fvkey, rkey = jax.random.split(key, 4)
    xy_positions = jax.random.uniform(key=pkey, shape=(n, 2), dtype=jnp.float64, minval=20.0, maxval=120.0)
    z_choices = jnp.array([0.0, 2.0], dtype=jnp.float64)
    z_positions = jax.random.choice(key=zkey, a=z_choices, shape=(n, 1))
    force_mins = jnp.array([-2.0, -2.0, -10.0], dtype=jnp.float64)
    force_maxs = jnp.array([2.0, 2.0, 10.0], dtype=jnp.float64)
    force_vecs = jax.random.uniform(key=fvkey, shape=(n, 3), dtype=jnp.float64, minval=force_mins, maxval=force_maxs)
    radii = jax.random.uniform(key=rkey, shape=(n, 1), dtype=jnp.float64, minval=10, maxval=20)
    return jnp.concat((xy_positions, z_positions, force_vecs, radii), axis=1)


def example():
    """
    We create a LinElasticityCont instance which handles continuous
    force positioning.
    We then create a few random force and apply them to the plate.
    The only thing that changes is how the forces are passed to the set_params() method.
    """
    problem, _ = initialize_empty_sim(continuous_forces=True)
    key = jax.random.key(seed=1)
    target_letter, _ = generate_target_shape(key)
    # ensure force format is correct
    # -> see LinElasticityCont for details
    # Continuous forces should have the shape
    # (num_forces, 7), where the second dimension is
    # an array of [x, y, z, f_x, f_y, f_z, radius]
    forces = get_random_forces(key)
    sol, grad, loss = single_simulation_and_grad_step(problem, forces, mean_squared_displacement_loss, target_letter)
    print("Simulation done.")
    print(f"The mse is {float(loss)}")


if __name__ == '__main__':
    example()
