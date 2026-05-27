"""
Minimal PPO trainer for single-step plate deformation.

This treats the task as a contextual bandit: the observation is a target
shape and the action is a single force field applied once.
"""

from __future__ import annotations

import time
from typing import Any

import jax
import jax.numpy as jnp
import optax
import logging

# Reduce noisy debug output from the FEM backend during experiments.
# Set the `jax_fem` logger to INFO to suppress DEBUG messages.
logging.getLogger("jax_fem").setLevel(logging.INFO)

from init_pde import initialize_empty_sims
from simulation import parallel_simulation_step
from target_shapes import generate_n_target_shapes
from gym_env import NUM_NODES


jax.config.update("jax_enable_x64", True)

OBS_DIM = NUM_NODES * 3
ACTION_DIM = NUM_NODES

CONFIG: dict[str, Any] = {
    "seed": 0,
    "updates": 200,
    "batch_size": 8,
    "ppo_epochs": 20,
    "lr": 3e-4,
    "max_force": 0.1,
    "clip_ratio": 0.2,
    "value_coef": 0.5,
    "entropy_coef": 1e-4,
    "init_log_std": -1.0,
    "hidden_sizes": [256, 256],
    "log_every": 5,
    "save_path": None,
    "save_every": False,
}


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
    obs_dim: int,
    action_dim: int,
    hidden_sizes: list[int],
    init_log_std: float = -1.0,
) -> dict[str, Any]:
    trunk_key, policy_key, value_key, _ = jax.random.split(key, 4)
    trunk = _mlp_init(trunk_key, [obs_dim] + hidden_sizes)
    policy = _linear_init(policy_key, hidden_sizes[-1], action_dim)
    value = _linear_init(value_key, hidden_sizes[-1], 1)
    log_std = jnp.full((action_dim,), init_log_std)
    return {"trunk": trunk, "policy": policy, "value": value, "log_std": log_std}


def _forward(params: dict[str, Any], obs: jax.Array) -> tuple[jax.Array, jax.Array]:
    h = _mlp_apply(params["trunk"], obs)
    mean = h @ params["policy"][0] + params["policy"][1]
    value = jnp.squeeze(h @ params["value"][0] + params["value"][1], axis=-1)
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
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    mean, value = _forward(params, obs)
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
) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array]]:
    mean, value = _forward(params, batch["obs"])
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
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        hidden_sizes=config["hidden_sizes"],
        init_log_std=config["init_log_std"],
    )
    optimizer = optax.adam(config["lr"])
    opt_state = optimizer.init(params)

    problems, _ = initialize_empty_sims(config["batch_size"], continuous_forces=False)

    for update in range(1, config["updates"] + 1):
        key, target_key = jax.random.split(key)
        targets, _ = _sample_targets(target_key, config["batch_size"])
        obs = targets.reshape(config["batch_size"], -1).astype(jnp.float32)

        action, log_prob, value, key = _sample_actions(key, params, obs, config["max_force"])
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

        for _ in range(config["ppo_epochs"]):
            (loss, (policy_loss, value_loss, entropy)), grads = jax.value_and_grad(
                _ppo_loss, has_aux=True
            )(
                params,
                batch,
                config["clip_ratio"],
                config["value_coef"],
                config["entropy_coef"],
                config["max_force"],
            )
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)

        if update % config["log_every"] == 0 or update == 1:
            print(
                f"update {update:04d} | reward {float(jnp.mean(reward)):.6f} | "
                f"mse {float(jnp.mean(mse)):.6f} | loss {float(loss):.6f} | "
                f"policy {float(policy_loss):.6f} | value {float(value_loss):.6f}"
            )

        # Checkpointing block kept but disabled for testing.
        # To enable saving, replace the condition with: `if config["save_path"] and update % config["save_every"] == 0:`
        if False and config["save_path"] and update % config["save_every"] == 0:
            with open(config["save_path"], "wb") as f:
                import pickle

                pickle.dump(params, f)


if __name__ == "__main__":
    start = time.time()
    train(CONFIG)
    print(f"done in {time.time() - start:.2f}s")
