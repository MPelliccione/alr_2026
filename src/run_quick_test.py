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
            "hidden_sizes": [64, 64],
            "log_every": 1,
            "save_path": None,
        }
    )
    train(cfg)
