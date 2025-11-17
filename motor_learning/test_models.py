from agents import GaussianAgent
import numpy as np
from pathlib import Path
import gymnasium as gym

def test_with_4_gaussians(mode):
    targets_positions = np.asarray(
                    [[-0.2546,  0.2546],   # Target 1 (top-left)
                     [ 0.2546,  0.2546],    # Target 2 (top-right)
                     [-0.2546, -0.2546],  # Target 3 (bottom-left)
                     [ 0.2546, -0.2546]], np.float32)
    assert mode in ["interactive", "record"], "mode should be either \"record\" or \"interactive\""
    if mode == "interactive":
        env = gym.make('Targets-v0', targets_positions=targets_positions, training_area=(-0.50, 0.50), max_trials=30, render_mode='human')
    if mode == "record":
        env = gym.make('Targets-v0', targets_positions=targets_positions, training_area=(-0.50, 0.50), max_trials=30, render_mode='rgb_array')
        env = gym.wrappers.RecordVideo(env, Path('motor_learning')/"tests"/"videos", episode_trigger= lambda _: True, name_prefix="Targets_4_Gaussian_Agents_")
    agent = [
        GaussianAgent(env, mu=targets_positions[0,:], std=0.2),
        GaussianAgent(env, mu=targets_positions[1,:], std=0.2),
        GaussianAgent(env, mu=targets_positions[2,:], std=0.2),
        GaussianAgent(env, mu=targets_positions[3,:], std=0.2),
    ]
    _, _ = env.reset()
    done = truncated = False
    i = 0
    while not done and not truncated:
        step = agent[i%4].step(target=i % 4)
        print(f"{i}: {step.reward},target={i%4},action={agent[i%4].env.unwrapped.actions_history[-1]['position']}")
        i+=1
        done = step.done
    env.close()

if __name__ == "__main__":
    # test the environment with 4 Gaussian Agents
    # test_with_4_gaussians(mode="record")
    test_with_4_gaussians(mode="interactive")