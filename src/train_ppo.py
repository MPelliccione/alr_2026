"""
Minimal PPO trainer for single-step plate deformation.

This treats the task as a contextual bandit: the observation is a target
shape and the action is a single force field applied once.
"""

from __future__ import annotations

import math
import pickle
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from init_pde import initialize_empty_sims
from simulation import parallel_simulation_step
from target_shapes import generate_n_target_shapes
from gym_env import NUM_NODES
from gnn import build_grid_graph, init_gnn_params, apply_gnn_batched


jax.config.update("jax_enable_x64", True)

OBS_DIM = NUM_NODES * 3
ACTION_DIM = NUM_NODES

CONFIG: dict[str, Any] = {
    "seed": 0,
    "updates": 200,
    "batch_size": 8,
    "ppo_epochs": 50,
    "lr": 3e-4,
    "max_force": 0.1,
    "clip_ratio": 0.2,
    "mini_batch_size": 4,
    "anneal_lr": True,
    "anneal_clip_ratio": True,
    "max_grad_norm": 0.5,
    "value_coef": 0.5,
    "entropy_coef": 1e-4,
    "init_log_std": -1.0,
    "gnn_hidden_dim": 64,
    "gnn_message_layers": 2,
    "log_every": 5,
    "save_path": None,
    "save_every": False,
}


def save_params(params: dict[str, Any], path: str | Path) -> None:
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cpu_params = jax.tree_util.tree_map(lambda x: np.array(x), params)
    with open(save_path, "wb") as f:
        pickle.dump(cpu_params, f)


def load_params(path: str | Path) -> dict[str, Any]:
    with open(path, "rb") as f:
        params = pickle.load(f)
    return jax.tree_util.tree_map(jnp.array, params)


def _linear_init(key: jax.Array, in_dim: int, out_dim: int) -> tuple[jax.Array, jax.Array]:
    limit = jnp.sqrt(6.0 / (in_dim + out_dim))
    w_key, _ = jax.random.split(key)
    w = jax.random.uniform(w_key, (in_dim, out_dim), minval=-limit, maxval=limit)
    b = jnp.zeros((out_dim,))
    return w, b


def _mlp_init(key: jax.Array, layer_sizes: list[int]) -> list[tuple[jax.Array, jax.Array]]:
    params: list[tuple[jax.Array, jax.Array]] = []
    keys = jax.random.split(key, len(layer_sizes) - 1)
    for k, in_dim, out_dim in zip(keys, layer_sizes[:-1], layer_sizes[1:], strict=True):
        params.append(_linear_init(k, in_dim, out_dim))
    return params


def _mlp_apply(params: list[tuple[jax.Array, jax.Array]], x: jax.Array) -> jax.Array:
    for w, b in params[:-1]:
        x = jnp.tanh(x @ w + b)
    w, b = params[-1]
    return x @ w + b


def init_actor_critic_params(
    key: jax.Array,
    action_dim: int,
    gnn_hidden_dim: int,
    gnn_message_layers: int,
    init_log_std: float = -1.0,
) -> dict[str, Any]:
    params = init_gnn_params(
        key,
        node_in_dim=7,
        edge_in_dim=3,
        hidden_dim=gnn_hidden_dim,
        n_message_layers=gnn_message_layers,
        num_nodes=action_dim,
    )
    params["log_std"] = jnp.full((action_dim,), init_log_std)
    return params


def _forward(
    params: dict[str, Any],
    obs: jax.Array,
    edge_feats: jax.Array,
    senders: jax.Array,
    receivers: jax.Array,
    node_coords: jax.Array,
    boundary_mask: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    # obs: (B, NUM_NODES * 3)
    targets = obs.reshape(-1, NUM_NODES, 3)
    coords = jnp.broadcast_to(node_coords[None, :, :], targets.shape)
    boundary = jnp.broadcast_to(boundary_mask[None, :, None], (targets.shape[0], NUM_NODES, 1))
    node_feats = jnp.concatenate([targets, coords, boundary], axis=-1)

    node_emb = apply_gnn_batched(params, node_feats, edge_feats, senders, receivers)
    mean = (node_emb @ params["mean_head"][0] + params["mean_head"][1]).squeeze(-1)

    pooled = jnp.mean(node_emb, axis=1)
    value = jnp.squeeze(_mlp_apply(params["value_mlp"], pooled), axis=-1)
    return mean, value


def _clip_log_std(log_std: jax.Array) -> jax.Array:
    return jnp.clip(log_std, -5.0, 2.0)


def _gaussian_log_prob(x: jax.Array, mean: jax.Array, log_std: jax.Array) -> jax.Array:
    log_std = _clip_log_std(log_std)
    inv_var = jnp.exp(-2.0 * log_std)
    quad = jnp.square(x - mean) * inv_var
    log_det = 2.0 * log_std + jnp.log(2.0 * jnp.pi)
    return -0.5 * jnp.sum(quad + log_det, axis=-1)


def _squash_correction(pre_tanh: jax.Array) -> jax.Array:
    return jnp.sum(jnp.log(1.0 - jnp.tanh(pre_tanh) ** 2 + 1e-6), axis=-1)


def _sample_actions(
    key: jax.Array,
    params: dict[str, Any],
    obs: jax.Array,
    max_force: float,
    edge_feats: jax.Array,
    senders: jax.Array,
    receivers: jax.Array,
    node_coords: jax.Array,
    boundary_mask: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    mean, value = _forward(params, obs, edge_feats, senders, receivers, node_coords, boundary_mask)
    log_std = _clip_log_std(params["log_std"])
    key, noise_key = jax.random.split(key)
    noise = jax.random.normal(noise_key, mean.shape)
    pre_tanh = mean + noise * jnp.exp(log_std)
    action = jnp.tanh(pre_tanh) * max_force
    log_prob = _gaussian_log_prob(pre_tanh, mean, log_std) - _squash_correction(pre_tanh)
    return action, log_prob, value, key


def _log_prob_from_action(
    action: jax.Array,
    mean: jax.Array,
    log_std: jax.Array,
    max_force: float,
) -> jax.Array:
    scaled = jnp.clip(action / max_force, -0.999, 0.999)
    pre_tanh = 0.5 * (jnp.log1p(scaled) - jnp.log1p(-scaled))
    return _gaussian_log_prob(pre_tanh, mean, log_std) - _squash_correction(pre_tanh)


def _gaussian_entropy(log_std: jax.Array) -> jax.Array:
    log_std = _clip_log_std(log_std)
    return jnp.sum(log_std + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e))


def _sample_targets(
    key: jax.Array, batch_size: int
) -> tuple[jax.Array, list[dict[str, Any]]]:
    targets = generate_n_target_shapes(key, n=batch_size)
    target_arrays, configs = zip(*targets, strict=True)
    return jnp.stack(target_arrays, axis=0), list(configs)


def _simulate_batch(
    problems: list[Any],
    targets: jax.Array,
    actions: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    batch_size = actions.shape[0]
    forces_z = actions.reshape(batch_size, NUM_NODES, 1).astype(jnp.float64)
    zeros = jnp.zeros((batch_size, NUM_NODES, 2), dtype=jnp.float64)
    loads = jnp.concatenate([zeros, forces_z], axis=-1)
    load_list = [loads[i] for i in range(batch_size)]
    sol_list = parallel_simulation_step(problems, load_list)
    sols = jnp.stack([sol[0] for sol in sol_list], axis=0)
    loss = jnp.mean(jnp.square(targets - sols), axis=(-1, -2))
    reward = -loss
    return reward.astype(jnp.float32), loss.astype(jnp.float32)


def _ppo_loss(
    params: dict[str, Any],
    batch: dict[str, jax.Array],
    clip_ratio: float,
    value_coef: float,
    entropy_coef: float,
    max_force: float,
    edge_feats: jax.Array,
    senders: jax.Array,
    receivers: jax.Array,
    node_coords: jax.Array,
    boundary_mask: jax.Array,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array]]:
    mean, value = _forward(params, batch["obs"], edge_feats, senders, receivers, node_coords, boundary_mask)
    log_std = _clip_log_std(params["log_std"])
    log_prob = _log_prob_from_action(batch["action"], mean, log_std, max_force)
    ratio = jnp.exp(log_prob - batch["log_prob"])
    unclipped = ratio * batch["adv"]
    clipped = jnp.clip(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * batch["adv"]
    policy_loss = -jnp.mean(jnp.minimum(unclipped, clipped))
    value_loss = 0.5 * jnp.mean(jnp.square(value - batch["returns"]))
    entropy = _gaussian_entropy(log_std)
    total_loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
    return total_loss, (policy_loss, value_loss, entropy)


def train(config: dict[str, Any]) -> None:
    key = jax.random.key(config["seed"])
    params = init_actor_critic_params(
        key,
        action_dim=ACTION_DIM,
        gnn_hidden_dim=config["gnn_hidden_dim"],
        gnn_message_layers=config["gnn_message_layers"],
        init_log_std=config["init_log_std"],
    )
    batch_size = int(config["batch_size"])
    mini_batch_size = int(config["mini_batch_size"])
    mini_batch_size = min(mini_batch_size, batch_size)
    if mini_batch_size <= 0:
        raise ValueError("mini_batch_size must be a positive integer")
    num_minibatches = math.ceil(batch_size / mini_batch_size)
    total_opt_steps = max(1, config["updates"] * config["ppo_epochs"] * num_minibatches)

    if config["anneal_lr"]:
        lr_schedule = optax.linear_schedule(
            init_value=config["lr"],
            end_value=0.0,
            transition_steps=total_opt_steps,
        )
    else:
        lr_schedule = config["lr"]

    if config["max_grad_norm"]:
        optimizer = optax.chain(
            optax.clip_by_global_norm(config["max_grad_norm"]),
            optax.adam(lr_schedule),
        )
    else:
        optimizer = optax.adam(lr_schedule)
    opt_state = optimizer.init(params)

    problems, _ = initialize_empty_sims(config["batch_size"], continuous_forces=False)
    senders, receivers, edge_feats, node_coords, boundary_mask = build_grid_graph()
    edge_feats = edge_feats.astype(jnp.float32)
    node_coords = node_coords.astype(jnp.float32)
    boundary_mask = boundary_mask.astype(jnp.float32)

    opt_step = 0
    save_every = config.get("save_every")
    periodic_save = (
        config["save_path"]
        and isinstance(save_every, int)
        and not isinstance(save_every, bool)
        and save_every > 0
    )

    for update in range(1, config["updates"] + 1):
        key, target_key = jax.random.split(key)
        targets, _ = _sample_targets(target_key, config["batch_size"])
        obs = targets.reshape(config["batch_size"], -1).astype(jnp.float32)

        action, log_prob, value, key = _sample_actions(
            key,
            params,
            obs,
            config["max_force"],
            edge_feats,
            senders,
            receivers,
            node_coords,
            boundary_mask,
        )
        reward, mse = _simulate_batch(problems, targets, action)

        returns = reward
        adv = reward - value
        adv = (adv - jnp.mean(adv)) / (jnp.std(adv) + 1e-8)

        batch = {
            "obs": obs,
            "action": action,
            "log_prob": log_prob,
            "adv": adv,
            "returns": returns,
        }

        total_loss = 0.0
        total_policy = 0.0
        total_value = 0.0
        total_entropy = 0.0
        total_batches = 0

        for _ in range(config["ppo_epochs"]):
            key, perm_key = jax.random.split(key)
            perm = jax.random.permutation(perm_key, batch_size)
            for start in range(0, batch_size, mini_batch_size):
                idx = perm[start : start + mini_batch_size]
                mini_batch = {k: v[idx] for k, v in batch.items()}

                if config["anneal_clip_ratio"]:
                    denom = max(1, total_opt_steps - 1)
                    current_clip = config["clip_ratio"] * (1.0 - (opt_step / denom))
                else:
                    current_clip = config["clip_ratio"]

                (loss, (policy_loss, value_loss, entropy)), grads = jax.value_and_grad(
                    _ppo_loss, has_aux=True
                )(
                    params,
                    mini_batch,
                    current_clip,
                    config["value_coef"],
                    config["entropy_coef"],
                    config["max_force"],
                    edge_feats,
                    senders,
                    receivers,
                    node_coords,
                    boundary_mask,
                )
                updates, opt_state = optimizer.update(grads, opt_state, params)
                params = optax.apply_updates(params, updates)

                total_loss += float(loss)
                total_policy += float(policy_loss)
                total_value += float(value_loss)
                total_entropy += float(entropy)
                total_batches += 1
                opt_step += 1

        loss = total_loss / max(1, total_batches)
        policy_loss = total_policy / max(1, total_batches)
        value_loss = total_value / max(1, total_batches)
        entropy = total_entropy / max(1, total_batches)

        if update % config["log_every"] == 0 or update == 1:
            print(
                f"update {update:04d} | reward {float(jnp.mean(reward)):.6f} | "
                f"mse {float(jnp.mean(mse)):.6f} | loss {float(loss):.6f} | "
                f"policy {float(policy_loss):.6f} | value {float(value_loss):.6f}"
            )

        if periodic_save and update % save_every == 0:
            save_params(params, config["save_path"])

    if config["save_path"]:
        save_params(params, config["save_path"])


if __name__ == "__main__":
    start = time.time()
    train(CONFIG)
    print(f"done in {time.time() - start:.2f}s")
