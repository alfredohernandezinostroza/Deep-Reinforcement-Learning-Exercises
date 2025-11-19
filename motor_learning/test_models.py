from agents import GaussianAgent, ErrorBasedAgentNonRL, DataAgent, Step
import numpy as np
from pathlib import Path
import gymnasium as gym
from motor_environments import Renderer

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
    env.render() #render empty frame
    done = truncated = False
    i = 0
    while not done and not truncated:
        step = agent[i%4].step(target=i % 4)
        print(f"{i}: {step.reward},target={i%4},action={agent[i%4].env.unwrapped.actions_history[-1]['position']}")
        i+=1
        done = step.done
        env.render()
    env.close()

def test_ErrorBasedAgentNonRL(mode, max_trials=30):
    targets_positions = np.asarray(
                    [[-0.2546,  0.2546],   # Target 1 (top-left)
                     [ 0.2546,  0.2546],    # Target 2 (top-right)
                     [-0.2546, -0.2546],  # Target 3 (bottom-left)
                     [ 0.2546, -0.2546]], np.float32)
    assert mode in ["interactive", "record"], "mode should be either \"record\" or \"interactive\""
    if mode == "interactive":
        env = gym.make('Targets-v0', targets_positions=targets_positions, training_area=(-0.50, 0.50), max_trials=max_trials, render_mode='human')
    if mode == "record":
        env = gym.make('Targets-v0', targets_positions=targets_positions, training_area=(-0.50, 0.50), max_trials=max_trials, render_mode='rgb_array')
        env = gym.wrappers.RecordVideo(env, Path('motor_learning')/"tests"/"videos", episode_trigger= lambda _: True, name_prefix="Targets_1_Error-Based_agent_")
    agent = ErrorBasedAgentNonRL(env,  exploration_scale=0.01, exploration_threshold=0.05, motor_noise_std=0.01, learning_rate=0.05)
    _, _ = env.reset() #start recording
    history_window = 10
    done = truncated = False
    i = 0
    while not done and not truncated:
        step: Step = agent.step(target=0)
        if i % 50 == 0:
            print(f"{i}: {step.reward},action={agent.env.unwrapped.actions_history[-1]['position']}")
        expected_reward = np.mean(np.abs(agent.rewards_history[-history_window:]))
        prediction_error = expected_reward - np.abs(step.reward)
        if prediction_error > 0:
            update_direction = step.action["intended_position"] - agent.mu
            update_magnitude = agent.learning_rate * prediction_error
            agent.mu = np.clip(agent.mu + update_magnitude * update_direction, env.action_space["position"].low, env.action_space["position"].high)

        # step = agent[i%4].step(target=i % 4)
        # print(f"{i}: {step.reward},target={i%4},action={agent[i%4].env.unwrapped.actions_history[-1]['position']}")
        i+=1
        done = step.done
        beliefs = {"artist_id": "Agent's belief","position": agent.mu}
        env.unwrapped.renderer.animate(beliefs)
        env.render()
    env.close()

    def render(env):
        renderers = env.render()
        env.rederers


if __name__ == "__main__":
    # test the environment with 4 Gaussian Agents, first recording and then in human mode
    # test_with_4_gaussians(mode="record")
    # test_with_4_gaussians(mode="interactive")
    
    #test non RL Error-based Agent
    test_ErrorBasedAgentNonRL(mode="record", max_trials=300)
    test_ErrorBasedAgentNonRL(mode="interactive", max_trials=300)