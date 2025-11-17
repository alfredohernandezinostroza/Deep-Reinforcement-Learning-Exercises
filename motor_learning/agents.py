import torch
import gymnasium as gym
import numpy as np
from dataclasses import dataclass, field
from typing import Any
from abc import ABC, abstractmethod
from motor_environments import Targets

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
        self.steps: list[Step] = []
        self.lists = EpisodeLists()
        self.total_reward: np.float32 = 0.0
    
    def append(self, step):
        self.steps.append(step)
        self.total_reward += step.reward
        self.lists.states.append(step.state)
        self.lists.actions.append(step.action)
        self.lists.next_states.append(step.next_state)
        self.lists.rewards.append(step.reward)
        self.lists.dones.append(step.done)
        self.lists.truncateds.append(step.truncated)

    def calculate_q_values(self, discount_factor):
        rewards = [step.reward for step in self.steps]
        q_vals_in_reverse = []
        sum_r = 0.0 
        for step, reward in zip(reversed(self.steps), reversed(rewards)):
            sum_r = sum_r*discount_factor + reward
            q_vals_in_reverse.append(sum_r)
            step.q_value = sum_r
        self.lists.q_values = list(reversed(q_vals_in_reverse))
        return self.lists.q_values

class GaussianAgent():
    def __init__(self, env: gym.Env, mu: np.float32, std: np.float32):
        self.env = env
        self.mu = mu
        self.std = std
        self.state = None
        # self.state, _info = env.reset()
        self.policy = np.random.normal

    def step(self, target):
        action = {"position": self.policy(self.mu, self.std),
                  "target": target}
        next_state, reward, done, truncated, info = self.env.step(action)
        step = Step(self.state, action, next_state, reward, done, truncated, info)
        self.state = next_state
        return step

    def yield_n_episodes(self, n):
        while True:
            episodes: list[Episode] = []
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

def episodes_to_tensors(episodes: list[Episode], device: str = 'cpu') -> tuple[torch.Tensor, torch.LongTensor, torch.LongTensor, torch.Tensor, torch.BoolTensor, torch.BoolTensor, torch.Tensor]:
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
    actions_tensor = torch.LongTensor(all_actions).to(device)
    next_states_tensor = torch.as_tensor(np.asarray(all_next_states), device=device, dtype=torch.float32)
    rewards_tensor = torch.LongTensor(all_rewards).to(device)
    dones_tensor = torch.BoolTensor(all_dones).to(device)
    truncateds_tensor = torch.BoolTensor(all_truncateds).to(device)
    q_values_tensor = torch.as_tensor(np.asarray(all_q_values), device=device, dtype=torch.float32)

    return states_tensor,actions_tensor,rewards_tensor,next_states_tensor,dones_tensor,truncateds_tensor, q_values_tensor
    