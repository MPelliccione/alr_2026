"""
Gymnasium environment for plate deformation via discrete nodal z-forces.

The agent applies a z-direction force at every node of the FEM mesh in a single step.
X and Y forces are fixed at zero since the target displacements are purely in z.
The environment simulates the resulting displacement and returns the negative MSE
against a randomly sampled letter target as the reward.

Observation: flattened target displacement field, shape (NUM_NODES * 3,) = (5046,)
Action:      per-node z-forces,                   shape (NUM_NODES,)    = (1682,)
Episode:     single-step (terminated=True after every step)


Design choices (Felix):
 - Single step env
 - Discrete action space
 - MSE Loss
 - Only consider z-forces
"""

import numpy as np
import jax
import jax.numpy as jnp
import gymnasium
from gymnasium import spaces

from init_pde import initialize_empty_sim
from simulation import single_simulation_step
from target_shapes import generate_target_shape

NUM_NODES = 1682


class PlateDeformationEnv(gymnasium.Env):
    """
    Single-step gymnasium environment for deforming a 140x140x2 mm elastic plate.

    Parameters
    ----------
    max_force : float
        Symmetric bound for each force component in the action space (kN).
    seed : int
        Initial JAX PRNG seed.
    """

    metadata = {"render_modes": []}

    def __init__(self, max_force: float = 0.1, seed: int = 0):
        super().__init__()

        # FEM problem — expensive to build, so we create it once and reuse
        self.problem, _ = initialize_empty_sim(continuous_forces=False)
        self.max_force = max_force

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(NUM_NODES * 3,),
            dtype=np.float64,
        )

        # Only z-forces: x/y are always zero since targets are z-only
        self.action_space = spaces.Box(
            low=-max_force,
            high=max_force,
            shape=(NUM_NODES,),
            dtype=np.float64,
        )

        self._rng_key = jax.random.key(seed)
        self._target: jnp.ndarray | None = None
        self._target_config: dict | None = None


    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng_key = jax.random.key(seed)

        self._rng_key, subkey = jax.random.split(self._rng_key)
        self._target, self._target_config = generate_target_shape(subkey)

        obs = np.asarray(self._target).reshape(-1)
        return obs, {"target_config": self._target_config}


    def step(self, action: np.ndarray):
        assert self._target is not None, "Call reset() before step()"

        # we only consider z-forces since the target displacements are purely in z
        forces_z = jnp.array(action, dtype=jnp.float64).reshape(NUM_NODES, 1)
        forces = jnp.concatenate([jnp.zeros((NUM_NODES, 2), dtype=jnp.float64), forces_z], axis=1)
        sol = single_simulation_step(self.problem, forces)

        loss = float(jnp.mean(jnp.square(self._target - sol)))
        reward = -loss

        # Observation is the target again; episode is over after this step
        obs = np.asarray(self._target).reshape(-1)
        info = {"loss": loss, "target_config": self._target_config}

        return obs, reward, True, False, info

    def render(self):
        pass
