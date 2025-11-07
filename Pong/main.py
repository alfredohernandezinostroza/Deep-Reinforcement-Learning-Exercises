import ale_py
from wrappers import make_env
import torch
from typing import Tuple, List, Any, Deque
import gymnasium as gym
import numpy as np
from dataclasses import dataclass
from collections import deque
import time
from torch.utils.tensorboard import SummaryWriter

gym.register_envs(ale_py)  # unnecessary but helpful for IDEs
DEVICE = 'cuda:1'
# DEFAULT_ENV_NAME = "ALE/Pong-v5" 
DEFAULT_ENV_NAME = "PongNoFrameskip-v4" 
TRAINING_BUFFER_SIZE = 10000
GAMMA = 0.99
INITIAL_EPSILON = 1.0
EPSILON_DECAY_WINDOW = 150000
MINIMUM_EPSILON = 0.01
COPY_TO_TARGET_NETWORK = 1000
MEAN_REWARD_BOUND = 19
BATCH_SIZE = 32
LEARNING_RATE = 1e-4

def main():
    env = make_env(DEFAULT_ENV_NAME)
    agent = Agent(env, buffer_size=TRAINING_BUFFER_SIZE)
    epsilon = INITIAL_EPSILON
    learning_net = deepQNetwork(env.observation_space.shape, env.action_space.n).to(DEVICE)
    target_net   = deepQNetwork(env.observation_space.shape, env.action_space.n).to(DEVICE)
    optimizer = torch.optim.Adam(learning_net.parameters(), lr=LEARNING_RATE)
    current_step = 1
    ts_frame = 0
    ts = time.time()
    all_rewards = []
    best_mean_reward = 0
    writer = SummaryWriter(comment="-pong-dql")
    while True:
        done, reward = agent.step(epsilon, learning_net)
        if done:
            all_rewards.append(reward)
            speed = (current_step - ts_frame) / (time.time() - ts)
            ts_frame = current_step
            ts = time.time()
            m_reward = np.mean(all_rewards[-100:])
            print(f"{current_step}: done {len(all_rewards)} games, reward {m_reward:.3f}, "
                  f"eps {epsilon:.2f}, speed {speed:.2f} f/s")
            writer.add_scalar("epsilon", epsilon, current_step)
            writer.add_scalar("speed", speed, current_step)
            writer.add_scalar("reward_100", m_reward, current_step)
            writer.add_scalar("reward", reward, current_step)
            mean_reward = np.mean(all_rewards[-100:])
            if mean_reward > best_mean_reward:
                torch.save(learning_net.state_dict(), f"best_net_reward_{mean_reward:.0f}_iterations_{current_step}.dat")
            if mean_reward > MEAN_REWARD_BOUND:
                print("Solved in %d frames!" % current_step)
                break
        is_buffer_full = len(agent.buffer) == agent.buffer.maxlen
        if not is_buffer_full:
            continue
        optimizer.zero_grad()
        random_buffer_indices = np.random.choice(len(agent.buffer), BATCH_SIZE, replace=False)
        sample_from_buffer = [agent.buffer[idx] for idx in random_buffer_indices]
        loss = calculate_loss(sample_from_buffer, learning_net, target_net)
        loss.backward()
        optimizer.step()
        if current_step % COPY_TO_TARGET_NETWORK == 0:
            target_net.load_state_dict(learning_net.state_dict())
        epsilon = max(MINIMUM_EPSILON, INITIAL_EPSILON - current_step/EPSILON_DECAY_WINDOW)
        current_step += 1
def calculate_loss(buffer: List, learning_net, target_net):
    states_tensor, actions_tensor, next_states_tensor, rewards_tensor, dones_tensor, truncateds_tensor = get_tensors(buffer)
    state_action_values: torch.Tensor = learning_net(states_tensor)
    best_q_values = state_action_values.gather(dim=1,index=actions_tensor.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        next_q_values = target_net(next_states_tensor).max(1)[0]
        next_q_values[dones_tensor] = 0
        next_q_values = next_q_values.detach()
    expected_state_action_values = rewards_tensor + GAMMA*next_q_values
    mse_loss = torch.nn.MSELoss()
    loss =  mse_loss(best_q_values, expected_state_action_values)
    return loss

def get_tensors(buffer: List) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,]:
    states, actions, next_states, rewards, dones, truncateds = [],[],[],[],[],[]
    for step in buffer:
        states.append(step.state)
        actions.append(step.action)
        next_states.append(step.next_state)
        rewards.append(step.reward)
        dones.append(step.done)
        truncateds.append(step.truncated)
    states = torch.as_tensor(np.asarray(states)).to(DEVICE)
    actions = torch.LongTensor(np.asarray(actions)).to(DEVICE)
    next_states = torch.as_tensor(np.asarray(next_states)).to(DEVICE)
    rewards = torch.FloatTensor(np.asarray(rewards)).to(DEVICE)
    dones = torch.BoolTensor(np.asarray(dones)).to(DEVICE)
    truncateds = torch.BoolTensor(np.asarray(truncateds)).to(DEVICE)
    return states, actions, next_states, rewards, dones, truncateds

@dataclass
class Step:
    state: np.ndarray
    action: int
    next_state: np.ndarray
    reward: np.float32
    done: bool
    truncated: bool
    info: Any

class Agent():
    def __init__(self, env: gym.Env, buffer_size: int):
        self.env = env
        self.buffer: Deque[Step] = deque(maxlen=buffer_size)
        self.state, info = self.env.reset()
        self.total_reward = 0

    @torch.no_grad()
    def step(self, epsilon: float, learning_net_detached: torch.nn.Module) -> Tuple[bool, np.float32]:
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state_tensor = torch.as_tensor(self.state, device=DEVICE).unsqueeze_(0)
            state_action_values = learning_net_detached(state_tensor)
            _, action_tensor = torch.max(state_action_values, dim=1)
            action = action_tensor.item()
        new_state, reward, done, truncated, info = self.env.step(action)
        step = Step(self.state, action, new_state, reward, done, truncated, info)   
        self.buffer.append(step)
        self.state = new_state
        self.total_reward += reward
        if done or truncated:
            total_reward = self.total_reward
            self.state, info = self.env.reset()
            self.total_reward = 0
            return True, total_reward
        return False, None

class deepQNetwork(torch.nn.Module):
    def __init__(self, input_shape: Tuple[int, int, int], n_actions: int):
        super().__init__()
        self.convolution = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=8, stride=4),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Flatten(),
        )
        fully_connected_input_size = self.convolution(torch.zeros(1, *input_shape)).size()[-1] 
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(in_features=fully_connected_input_size, out_features=512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(in_features=512, out_features=n_actions)
        )
        self.net = torch.nn.Sequential(
            self.convolution,
            self.linear,
        )

    def forward(self, x: torch.ByteTensor):
        normalized = x / 255.0
        return self.net(normalized)


if __name__ == "__main__":
    main()