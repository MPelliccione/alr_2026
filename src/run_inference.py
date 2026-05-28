"""
Run inference with a trained PPO-GNN policy and export VTK files.

This script loads a saved checkpoint, generates a target letter, runs the
policy to predict nodal forces, solves the FEM system, and writes VTK files
for the target, predicted forces, and resulting displacement field.
"""

from __future__ import annotations

import copy
from typing import Any

import jax
import jax.numpy as jnp

from gym_env import NUM_NODES
from gnn import build_grid_graph
from init_pde import initialize_empty_sim
from postprocessing import save_data_to_vtk, save_sol_to_vtk
from simulation import single_simulation_step
from target_shapes import generate_target_shape
from train_ppo import _clip_log_std, _forward, load_params
from utils import get_zero_forces


CONFIG: dict[str, Any] = {
    "checkpoint_path": "checkpoints/ppo_params.pkl",
    "seed": 0,
    "letter": "A",
    "rotate": False,
    "max_force": 0.1,
    "use_policy_mean": True,
    "compare_baseline": True,
    "save_dir": "paraview_out",
    "save_prefix": "inference",
    "save_target_vtk": True,
    "save_sol_vtk": True,
    "save_forces_vtk": True,
    "save_baseline_vtk": False,
}


def _make_action(
    params: dict[str, Any],
    obs: jax.Array,
    max_force: float,
    key: jax.Array,
    edge_feats: jax.Array,
    senders: jax.Array,
    receivers: jax.Array,
    node_coords: jax.Array,
    boundary_mask: jax.Array,
    use_policy_mean: bool,
) -> tuple[jax.Array, jax.Array]:
    mean, value = _forward(params, obs, edge_feats, senders, receivers, node_coords, boundary_mask)
    if use_policy_mean:
        action = jnp.tanh(mean) * max_force
        return action, value

    log_std = _clip_log_std(params["log_std"])
    noise = jax.random.normal(key, mean.shape)
    pre_tanh = mean + noise * jnp.exp(log_std)
    action = jnp.tanh(pre_tanh) * max_force
    return action, value


def run_inference(config: dict[str, Any]) -> None:
    params = load_params(config["checkpoint_path"])
    key = jax.random.key(config["seed"])

    letter = config.get("letter")
    if isinstance(letter, str) and letter.strip():
        letter_subset = [letter.strip()[0].upper()]
    else:
        letter_subset = None

    key, target_key = jax.random.split(key)
    target, meta = generate_target_shape(
        target_key, rotate=config["rotate"], letter_subset=letter_subset
    )
    obs = target.reshape(1, -1).astype(jnp.float32)

    senders, receivers, edge_feats, node_coords, boundary_mask = build_grid_graph()
    edge_feats = edge_feats.astype(jnp.float32)
    node_coords = node_coords.astype(jnp.float32)
    boundary_mask = boundary_mask.astype(jnp.float32)

    key, action_key = jax.random.split(key)
    action, value = _make_action(
        params,
        obs,
        config["max_force"],
        action_key,
        edge_feats,
        senders,
        receivers,
        node_coords,
        boundary_mask,
        config["use_policy_mean"],
    )
    action = action.squeeze(0)

    zeros = jnp.zeros((NUM_NODES, 2), dtype=jnp.float64)
    forces = jnp.concatenate([zeros, action[:, None].astype(jnp.float64)], axis=-1)

    problem, _ = initialize_empty_sim(continuous_forces=False)
    sol = single_simulation_step(problem, forces)
    mse = jnp.mean(jnp.square(target - sol))

    print(f"Target letter: {meta.get('letter')}")
    print(f"Target config: {meta}")
    print(f"Value estimate: {float(value.squeeze(0)):.6f}")
    print(f"MSE (policy): {float(mse):.6f}")

    if config["compare_baseline"]:
        baseline_problem, _ = initialize_empty_sim(continuous_forces=False)
        baseline_forces = get_zero_forces()
        baseline_sol = single_simulation_step(baseline_problem, baseline_forces)
        baseline_mse = jnp.mean(jnp.square(target - baseline_sol))
        print(f"MSE (zero forces): {float(baseline_mse):.6f}")
        if config["save_baseline_vtk"]:
            save_sol_to_vtk(
                baseline_sol,
                f"{config['save_prefix']}_baseline_sol.vtk",
                config["save_dir"],
            )

    if config["save_target_vtk"]:
        save_data_to_vtk(
            target,
            f"{config['save_prefix']}_target.vtk",
            config["save_dir"],
        )
    if config["save_sol_vtk"]:
        save_sol_to_vtk(
            sol,
            f"{config['save_prefix']}_sol.vtk",
            config["save_dir"],
        )
    if config["save_forces_vtk"]:
        save_data_to_vtk(
            forces,
            f"{config['save_prefix']}_forces.vtk",
            config["save_dir"],
        )

    print(f"Saved VTK files in: {config['save_dir']}")


if __name__ == "__main__":
    run_inference(copy.deepcopy(CONFIG))
