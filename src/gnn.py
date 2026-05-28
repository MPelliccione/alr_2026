"""
Simple, dependency-free GNN utilities implemented in pure JAX.

Provides:
- build_grid_graph: construct a regular-grid graph for the plate mesh
- init_gnn_params: initialize message-passing network parameters
- apply_gnn / apply_gnn_batched: run the message-passing network

This is intentionally small and self-contained so it can be integrated
without adding external GNN libraries.
"""
from __future__ import annotations

from typing import Tuple, List

import jax
import jax.numpy as jnp


def build_grid_graph(
    nx: int = 29, ny: int = 29, nz: int = 2, domain_x: float = 140.0, domain_y: float = 140.0, domain_z: float = 2.0
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Build a regular 3D grid graph and simple geometric edge features.

    Returns:
    - senders: (E,) int array of sender node indices
    - receivers: (E,) int array of receiver node indices
    - edge_feats: (E, 3) relative vector from sender -> receiver
    - node_coords: (N, 3) coordinates of each node
    - boundary_mask: (N,) boolean mask True for boundary nodes
    """
    # node coordinates
    x = jnp.linspace(0.0, domain_x, nx)
    y = jnp.linspace(0.0, domain_y, ny)
    z = jnp.linspace(0.0, domain_z, nz)
    xv, yv, zv = jnp.meshgrid(x, y, z, indexing="ij")
    pts = jnp.stack([xv, yv, zv], axis=-1)
    node_coords = jnp.reshape(pts, (-1, 3))

    def idx(i, j, k):
        return int(i * (ny * nz) + j * nz + k)

    edges: List[Tuple[int, int]] = []
    # only generate positive-offset undirected edges then add both directions
    offsets = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                u = idx(i, j, k)
                for di, dj, dk in offsets:
                    ni, nj, nk = i + di, j + dj, k + dk
                    if 0 <= ni < nx and 0 <= nj < ny and 0 <= nk < nz:
                        v = idx(ni, nj, nk)
                        edges.append((u, v))
                        edges.append((v, u))
    senders = jnp.array([e[0] for e in edges], dtype=jnp.int32)
    receivers = jnp.array([e[1] for e in edges], dtype=jnp.int32)

    edge_vecs = node_coords[receivers] - node_coords[senders]

    # boundary mask
    boundary_mask = jnp.zeros((nx * ny * nz,), dtype=jnp.bool_)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if i in (0, nx - 1) or j in (0, ny - 1) or k in (0, nz - 1):
                    boundary_mask = boundary_mask.at[idx(i, j, k)].set(True)

    return senders, receivers, edge_vecs, node_coords, boundary_mask


# ---- small MLP helpers (local copies) ----

def _linear_init(key: jax.Array, in_dim: int, out_dim: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    limit = jnp.sqrt(6.0 / (in_dim + out_dim))
    w_key, _ = jax.random.split(key)
    w = jax.random.uniform(w_key, (in_dim, out_dim), minval=-limit, maxval=limit)
    b = jnp.zeros((out_dim,))
    return w, b


def _mlp_init(key: jax.Array, layer_sizes: List[int]) -> List[Tuple[jnp.ndarray, jnp.ndarray]]:
    params: List[Tuple[jnp.ndarray, jnp.ndarray]] = []
    keys = jax.random.split(key, len(layer_sizes) - 1)
    for k, in_dim, out_dim in zip(keys, layer_sizes[:-1], layer_sizes[1:], strict=True):
        params.append(_linear_init(k, in_dim, out_dim))
    return params


def _mlp_apply(params: List[Tuple[jnp.ndarray, jnp.ndarray]], x: jnp.ndarray) -> jnp.ndarray:
    for w, b in params[:-1]:
        x = jnp.tanh(x @ w + b)
    w, b = params[-1]
    return x @ w + b


def init_gnn_params(
    key: jax.Array,
    node_in_dim: int,
    edge_in_dim: int,
    hidden_dim: int = 64,
    n_message_layers: int = 2,
    num_nodes: int = 1682,
) -> dict:
    """Initialize a small message-passing GNN's parameters.

    Returns a dict containing MLP parameter lists and a `log_std` for actions.
    """
    total_keys = 2 + 2 * n_message_layers + 2
    keys = jax.random.split(key, total_keys)
    k_node_enc = keys[0]
    k_edge_enc = keys[1]
    rest = list(keys[2:])

    params = {}
    params["node_enc"] = _mlp_init(k_node_enc, [node_in_dim, hidden_dim])
    params["edge_enc"] = _mlp_init(k_edge_enc, [edge_in_dim, hidden_dim])

    params["msg_mlps"] = []
    params["upd_mlps"] = []
    for i in range(n_message_layers):
        km = rest[2 * i]
        ku = rest[2 * i + 1]
        params["msg_mlps"].append(_mlp_init(km, [3 * hidden_dim, hidden_dim]))
        params["upd_mlps"].append(_mlp_init(ku, [2 * hidden_dim, hidden_dim]))

    # remaining keys: mean head and value mlp
    k_mean = rest[-2]
    k_value = rest[-1]

    params["mean_head"] = _linear_init(k_mean, hidden_dim, 1)
    params["value_mlp"] = _mlp_init(k_value, [hidden_dim, hidden_dim, 1])

    params["log_std"] = jnp.full((num_nodes,), -1.0)

    return params


def apply_gnn(params: dict, node_feats: jnp.ndarray, edge_feats: jnp.ndarray, senders: jnp.ndarray, receivers: jnp.ndarray) -> jnp.ndarray:
    """Run the message-passing GNN for one graph instance.

    node_feats: (N, node_in_dim)
    edge_feats: (E, edge_in_dim)
    returns node embeddings (N, hidden_dim)
    """
    N = node_feats.shape[0]

    # encode
    h = _mlp_apply(params["node_enc"], node_feats)
    e = _mlp_apply(params["edge_enc"], edge_feats)
    H = h.shape[1]

    for msg_mlp, upd_mlp in zip(params["msg_mlps"], params["upd_mlps"]):
        s = h[senders]
        r = h[receivers]
        m_in = jnp.concatenate([s, r, e], axis=-1)
        m = _mlp_apply(msg_mlp, m_in)
        # Sum messages per receiver node without relying on deprecated segment_sum.
        agg = jnp.zeros((N, H), dtype=m.dtype).at[receivers].add(m)
        upd_in = jnp.concatenate([h, agg], axis=-1)
        delta = _mlp_apply(upd_mlp, upd_in)
        h = jnp.tanh(h + delta)

    return h


def apply_gnn_batched(params: dict, batch_node_feats: jnp.ndarray, edge_feats: jnp.ndarray, senders: jnp.ndarray, receivers: jnp.ndarray) -> jnp.ndarray:
    """Batched wrapper over apply_gnn.

    batch_node_feats: (B, N, node_in_dim)
    returns: (B, N, hidden_dim)
    """
    fn = lambda nf: apply_gnn(params, nf, edge_feats, senders, receivers)
    return jax.vmap(fn)(batch_node_feats)
