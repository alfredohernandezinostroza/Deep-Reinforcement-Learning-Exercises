import typing
import torch 
import torch.nn
import torch.optim
import gymnasium as gym
import numpy as np
from dataclasses import dataclass
from typing import List, Generator, Tuple
from torch.utils.tensorboard import SummaryWriter

HIDDEN_SIZE = 128 
PERCENTILE = 70
LEARNING_RATE = 0.01
DECAY_RATES = (0.9, 0.999)
NUMBER_OF_EPISODES_PER_BATCH = 16

@dataclass
class EpisodeStep:
    step_reward: float
    observation: np.ndarray
    action: int
@dataclass
class Episode:
    steps: List[EpisodeStep]
    total_reward: float

class OneLayerDecisionNet(torch.nn.Module):
    def __init__(self, observation_size: int, n_neurons: int, n_actions: int):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(in_features=observation_size,out_features=n_neurons),
            torch.nn.ReLU(inplace = True),
            # torch.nn.Sigmoid() we need logits, not immediatly a result
            torch.nn.Linear(in_features=n_neurons,out_features=n_actions)
        )
    def forward(self, x: torch.Tensor):
        return self.layers(x)
    

def iterate_batches(env: gym.Env, net: torch.nn.Module, n_episodes: int) -> Generator[Episode, any, any]:
    softmax = torch.nn.Softmax(dim=1) #dim=1 because we want to apply softmax to each row of the matrix, as each one is an independent observation
    while True:    
        batch: List[Episode] = []
        for i in range(n_episodes):
            observation, info = env.reset()
            episode = Episode(steps=[], total_reward=0.0)
            done = truncated = False
            while (not done and not truncated):
                observation_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(device)
                action_logits = net(observation_tensor) #we expecta a matrix with rows being an observation
                action_probs_tensor: torch.Tensor = softmax(action_logits)
                action_probs: np.ndarray = action_probs_tensor.squeeze().detach().cpu().numpy()
                action = np.random.choice(len(action_probs), p=action_probs)
                next_observation, reward, done, truncated, info = env.step(action)
                step = EpisodeStep(step_reward=reward, observation=observation, action=action)
                episode.steps.append(step)
                episode.total_reward += reward
                observation = next_observation
            batch.append(episode)
        yield batch
    
def filter_batch(batch: List[Episode], device: str) -> Tuple[torch.FloatTensor, torch.LongTensor, float, float]:
    best_observations = []
    best_actions = []
    rewards = [episode.total_reward for episode in batch]
    minimum_reward = np.percentile(rewards,q=70)
    rewards_mean = np.mean(rewards)
    for episode in batch:
        if episode.total_reward >= minimum_reward:
            best_observations.extend(step.observation for step in episode.steps)
            best_actions.extend(step.action for step in episode.steps)
    best_observations = np.vstack(best_observations)
    return torch.FloatTensor(best_observations).to(device), torch.LongTensor(best_actions).to(device), minimum_reward, rewards_mean

if __name__ == "__main__":
    device = 'cpu'
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    env = gym.wrappers.RecordVideo(env, video_folder="video", episode_trigger=lambda x: x % 200 == 0)
    net = OneLayerDecisionNet(observation_size=env.observation_space.shape[0], n_neurons=HIDDEN_SIZE, n_actions=env.action_space.n)
    net = net.to(device)
    optimizer = torch.optim.Adam(params=net.parameters(),lr=LEARNING_RATE, betas=DECAY_RATES)
    objective = torch.nn.CrossEntropyLoss()
    writer = SummaryWriter(comment="-cartpole")

    for iteration_number, batch in enumerate(iterate_batches(env, net, NUMBER_OF_EPISODES_PER_BATCH)):
        # if iteration_number % 48 == 0:
        #     env = gym.wrappers.RecordVideo(env, video_folder="video", episode_trigger=lambda x: True)
        best_observations, best_actions, minimum_reward, mean_reward = filter_batch(batch, device)
        best_action_logits = net(best_observations)
        loss = objective(best_action_logits, best_actions)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("%d: loss=%.3f, reward_mean=%.1f, rw_bound=%.1f" % (
            iteration_number, loss.item(), mean_reward, minimum_reward))
        writer.add_scalar("loss", loss.item(), iteration_number)
        writer.add_scalar("reward_bound", minimum_reward, iteration_number)
        writer.add_scalar("reward_mean", mean_reward, iteration_number)
        if minimum_reward > 475:
            print("Solved!")
            break
    writer.close()
            