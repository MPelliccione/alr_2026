"""
Quick test runner for PPO trainer.

Usage:
    python src/run_quick_test.py

This will run a very short training (few updates, small batch) to smoke-test the
training loop without saving checkpoints.
"""

from train_ppo import CONFIG, train
import copy

if __name__ == "__main__":
    cfg = copy.deepcopy(CONFIG)
    cfg.update(
        {
            "seed": 42,
            "updates": 2,
            "batch_size": 2,
            "ppo_epochs": 1,
            "lr": 1e-3,
            "mini_batch_size": 2,
            "anneal_lr": False,
            "anneal_clip_ratio": False,
            "max_grad_norm": 0.5,
            "gnn_hidden_dim": 32,
            "gnn_message_layers": 1,
            "log_every": 1,
            "save_path": None,
        }
    )
    train(cfg)
