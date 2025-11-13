import torch
import gymnasium as gym
from pathlib import Path
from dataclasses import dataclass, field
import numpy as np
from typing import List, Any
from torch.utils.tensorboard import SummaryWriter

HIDDEN_LAYER_SIZE = 128
DEVICE='cuda:1'
LEARNING_RATE = 0.01
EPISODES_PER_TRAINING_SESSION=4
GAMMA=0.99
# ENTROPY_BETA=

def main():
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    env = gym.wrappers.RecordVideo(env, Path('.')/"videos", episode_trigger=lambda x: x % 400 == 0)
    net = TwoOutputNetwork(env.observation_space.shape[0], n_actions=2).to(DEVICE)
    optimizer = torch.optim.Adam(net.parameters(), lr= LEARNING_RATE)
    agent = PolicyAgent(env, policy=net)
    total_rewards = []
    current_iteration = 0
    done_episodes = 0
    writer = SummaryWriter(comment="-cartpole-policy")
    for episodes in agent.yield_n_episodes(n=EPISODES_PER_TRAINING_SESSION):
        episodes_total_rewards = [episode.total_reward for episode in episodes]
        total_rewards.extend(episodes_total_rewards)
        mean_rewards = float(np.mean(total_rewards[-100:]))
        print(f"{current_iteration}: this episodes reward mean: {np.mean(episodes_total_rewards):6.2f}, mean_100: {mean_rewards:6.2f}, "
                f"episodes: {done_episodes}")
        writer.add_scalar("this_episodes_reward_mean", np.mean(episodes_total_rewards), current_iteration)
        writer.add_scalar("reward_100", mean_rewards, current_iteration)
        writer.add_scalar("episodes", done_episodes, current_iteration)
        if mean_rewards > 450:
            print(f"Solved in {current_iteration} steps and {done_episodes} episodes!")
            break

        states_tensor,actions_tensor,_,_,_,_, q_values_tensor = episodes_to_tensors(episodes, DEVICE)
        logits: torch.Tensor = net(states_tensor)
        log_probs: torch.Tensor = torch.nn.functional.log_softmax(logits, dim=1)
        taken_actions_probs: torch.Tensor = log_probs[range(len(log_probs)), actions_tensor]
        loss = - (q_values_tensor * taken_actions_probs).mean()
        # probs: torch.Tensor = torch.nn.functional.softmax(logits, dim=1)
        # entropy_t = -(probs * log_probs).sum(dim=1).mean() 
        # entropy_loss_t = -ENTROPY_BETA * entropy_t 
        # loss_t = loss_policy_t + entropy_loss_t 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        current_iteration += 1
        done_episodes += EPISODES_PER_TRAINING_SESSION
    writer.close()
        

# def calculate_loss(q_values, log_probs_tensor, q_values_tensor):
#     objective = torch.nn.MSELoss()
    
#     loss = objective()

@dataclass
class Step():
    state: np.ndarray
    action: int
    next_state: np.ndarray
    reward: np.float32
    done: bool
    truncated: bool
    info: Any
    q_value: np.float32 | None = None

@dataclass
class EpisodeLists():
    states: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    next_states: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    dones: list = field(default_factory=list)
    truncateds: list = field(default_factory=list)
    q_values: list = field(default_factory=list)

class Episode():
    def __init__(self):
        self.steps: List[Step] = []
        self.lists = EpisodeLists()
        self.total_reward: np.float32 = 0.0

    def append(self, step):
        self.steps.append(step)
        self.total_reward += step.reward
        # populate sublists as we go
        self.lists.states.append(step.state)
        self.lists.actions.append(step.action)
        self.lists.next_states.append(step.next_state)
        self.lists.rewards.append(step.reward)
        self.lists.dones.append(step.done)
        self.lists.truncateds.append(step.truncated)

    def calculate_q_values(self):
        rewards = [step.reward for step in self.steps]
        q_vals_in_reverse = []
        sum_r = 0.0 
        for step, reward in zip(reversed(self.steps), reversed(rewards)):
        # for i, r in enumerate(reversed(rewards)): 
            sum_r = sum_r*GAMMA + reward
            q_vals_in_reverse.append(sum_r)
            step.q_value = sum_r
        self.lists.q_values = list(reversed(q_vals_in_reverse))
        return self.lists.q_values

class TwoOutputNetwork(torch.nn.Module):
    def __init__(self, n_inputs, n_actions=2):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(in_features=n_inputs, out_features=HIDDEN_LAYER_SIZE),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(in_features=HIDDEN_LAYER_SIZE, out_features=n_actions)
        )

    def forward(self, x):
        return self.layers(x)

class PolicyAgent():
    def __init__(self, env: gym.Env, policy: torch.nn.Module):
        self.env = env
        self.state, _info = env.reset()
        self.policy = policy
    @torch.no_grad()
    def step(self):
        state: torch.Tensor = torch.as_tensor(self.state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        logits = self.policy(state)
        action_probs = torch.nn.functional.softmax(logits, dim=1)
        action = torch.multinomial(action_probs, num_samples=1).item()
        next_state, reward, done, truncated, info = self.env.step(action)
        step = Step(self.state, action, next_state, reward, done, truncated, info)
        self.state = next_state
        return step

    def yield_n_episodes(self, n):
        while True:
            episodes: List[Episode] = []
            for i in range(n):
                self.reset()
                episode = Episode()
                done = truncated = False
                while not done and not truncated:
                    step = self.step()
                    episode.append(step)
                    done = step.done
                    truncated = step.truncated
                episode.calculate_q_values()
                episodes.append(episode)
            yield episodes

    def reset(self):
        self.state, _info = self.env.reset()

def episodes_to_tensors(episodes: List[Episode], device: str) -> tuple[torch.Tensor, torch.LongTensor, torch.LongTensor, torch.Tensor, torch.BoolTensor, torch.BoolTensor, torch.Tensor]:
    """"Gives tensors representing concatenated steps from an episode list"""
    all_states = []
    all_actions = []
    all_rewards = []
    all_next_states = []
    all_dones = []
    all_truncateds = []
    all_q_values = []
    for episode in episodes:
        all_states.extend(episode.lists.states)
        all_actions.extend(episode.lists.actions)
        all_next_states.extend(episode.lists.next_states)
        all_rewards.extend(episode.lists.rewards)
        all_dones.extend(episode.lists.dones)
        all_truncateds.extend(episode.lists.truncateds)
        all_q_values.extend(episode.lists.q_values)
    # all_states = np.vstack(all_states)
    # all_next_states = np.vstack(all_next_states)
    # all_q_values = np.vstack(all_q_values)

    states_tensor = torch.as_tensor(np.asarray(all_states), device=device, dtype=torch.float32)
    actions_tensor = torch.LongTensor(all_actions).to(DEVICE)
    next_states_tensor = torch.as_tensor(np.asarray(all_next_states), device=device, dtype=torch.float32)
    rewards_tensor = torch.LongTensor(all_rewards).to(DEVICE)
    dones_tensor = torch.BoolTensor(all_dones).to(DEVICE)
    truncateds_tensor = torch.BoolTensor(all_truncateds).to(DEVICE)
    q_values_tensor = torch.as_tensor(np.asarray(all_q_values), device=device, dtype=torch.float32)

    return states_tensor,actions_tensor,rewards_tensor,next_states_tensor,dones_tensor,truncateds_tensor, q_values_tensor
    
if __name__ == "__main__":
    main()