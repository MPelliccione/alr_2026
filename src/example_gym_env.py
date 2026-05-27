"""
Minimal usage example for PlateDeformationEnv. This is only meant to run a single step. Not sure if
my pc could handle this.
"""

import jax
jax.config.update("jax_enable_x64", True)

from gym_env import PlateDeformationEnv

env = PlateDeformationEnv(max_force=0.1, seed=42)

obs, info = env.reset()
print(f"Target letter: {info['target_config']['letter']}")
# In this case observation space also contains x and y axis
print(f"Observation shape: {obs.shape}")
print(f"Action space shape: {env.action_space.shape}")

action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
print(f"Reward: {reward:.6f}  |  MSE loss: {info['loss']:.6f}")
