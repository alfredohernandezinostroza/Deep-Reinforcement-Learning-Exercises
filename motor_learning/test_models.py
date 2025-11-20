from agents import GaussianAgent, ErrorBasedAgentNonRL, DataAgent, Step
import numpy as np
from pathlib import Path
import gymnasium as gym
from motor_environments import Renderer

def test_with_4_gaussians(mode, max_trials=30):
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
        env = gym.wrappers.RecordVideo(env, Path('motor_learning')/"tests"/"videos", episode_trigger= lambda _: True, name_prefix="Targets_4_Gaussian_Agents_")
    # env = gym.wrappers.ClipAction(env)
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
        beliefs = {"artist_id": f"Agent's belief {i%4}","position": agent[i%4].mu+np.random.normal([0,0],0.01)}
        env.unwrapped.renderer.animate(beliefs)
        env.render()
    env.close()

def test_ErrorBasedAgentNonRL(mode, max_trials=30):
    targets_positions = np.asarray(
                    [[-0.2546,  0.2546],   # Target 0 (top-left)
                     [ 0.2546,  0.2546],    # Target 1 (top-right)
                     [-0.2546, -0.2546],  # Target 2 (bottom-left)
                     [ 0.2546, -0.2546]], np.float32)
    assert mode in ["interactive", "record"], "mode should be either \"record\" or \"interactive\""
    if mode == "interactive":
        env = gym.make('Targets-v0', targets_positions=targets_positions, training_area=(-0.50, 0.50), max_trials=max_trials, render_mode='human')
        env.unwrapped.next_target_generator = lambda : 0
    if mode == "record":
        env = gym.make('Targets-v0', targets_positions=targets_positions, training_area=(-0.50, 0.50), max_trials=max_trials, render_mode='rgb_array')
        env.unwrapped.next_target_generator = lambda : 0
        env = gym.wrappers.RecordVideo(env, Path('motor_learning')/"tests"/"videos", episode_trigger= lambda _: True, name_prefix="Targets_1_Error-Based_agent_")
    agent = ErrorBasedAgentNonRL(env,  exploration_scale=0.01, motor_noise_std=0.01, learning_rate=1)
    _, _ = env.reset() #start recording
    history_window = 10
    done = truncated = False
    i = 0
    while not done and not truncated:
        step: Step = agent.step(target=0)
        if i % 1 == 0:
            print(f"{i}: {step.reward},action={agent.env.unwrapped.actions_history[-1]}")
        expected_reward = np.mean(np.abs(agent.rewards_history[-history_window:]))
        prediction_error = expected_reward - np.abs(step.reward)
        if prediction_error > 0:
            update_direction = step.info["agent_info"]["intended_action"] - agent.mu
            update_magnitude = agent.learning_rate * prediction_error
            agent.mu = np.clip(agent.mu + update_magnitude * update_direction, env.action_space.low, env.action_space.high)
        i+=1
        done = step.done
        beliefs = {"artist_id": "Agent's belief","position": agent.mu}
        env.unwrapped.renderer.animate(beliefs)
        env.render()
    env.close()


def test_ForagingAgentNonRL(mode, max_trials=30):
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
    agent = ForagingAgentNonRL(env,  exploration_scale=0.01, exploration_threshold=0.1, motor_noise_std=0.01, learning_rate=1)
    _, _ = env.reset() #start recording
    history_window = 10
    done = truncated = False
    i = 0
    while not done and not truncated:
        step: Step = agent.step(target=0)
        if i % 1 == 0:
            print(f"{i}: {step.reward},action={agent.env.unwrapped.actions_history[-1]['position']}")
        if agent.is_exploring:
            expected_reward = np.mean(np.abs(agent.rewards_history[-history_window:]))
            prediction_error = expected_reward - np.abs(step.reward)
            if prediction_error > 0:
                update_direction = step.action["intended_position"] - agent.mu
                update_magnitude = agent.learning_rate * prediction_error
                agent.mu = np.clip(agent.mu + update_magnitude * update_direction, env.action_space["position"].low, env.action_space["position"].high)
            beliefs = {"artist_id": "Agent's belief","position": agent.mu}
            env.unwrapped.renderer.animate(beliefs)
        done = step.done
        i+=1
        env.render()
    env.close()

    def render(env):
        renderers = env.render()
        env.rederers


if __name__ == "__main__":
    # test the environment with 4 Gaussian Agents, first recording and then in human mode
    # test_with_4_gaussians(mode="record")
    # test_with_4_gaussians(mode="interactive", max_trials=300)
    
    #test non RL Error-based Agent
    # test_ErrorBasedAgentNonRL(mode="record", max_trials=300)
    test_ErrorBasedAgentNonRL(mode="interactive", max_trials=800)

    #test non RL Foraging Agent
    # test_ForagingAgentNonRL(mode="record", max_trials=800)
    # test_ForagingAgentNonRL(mode="interactive", max_trials=800)