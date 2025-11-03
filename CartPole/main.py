import typing
import torch 
import torch.nn
import torch.optim
import gymnasium as gym
import numpy as np
from dataclasses import dataclass
from types import List, Generator
from tensorboard import SummaryWriter

HIDDEN_SIZE = 128 
PERCENTILE = 70
LEARNING_RATE = 0.5
DECAY_RATES = (0.5, 0.999)
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

class OneLayerDecision(torch.nn.Module):
    def __init__(self, observation_size: int, n_neurons: int, n_actions: int):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(in_features=observation_size,out_features=n_neurons),
            torch.nn.ReLU(in_place = True),
            # torch.nn.Sigmoid() we need logits, not immediatly a result
            torch.nn.Linear(in_features=n_neurons,out_features=n_actions)
        )
    def forward(self, x: torch.Tensor):
        return self.layers(x)
    

def iterate_batches(env: gym.Env, net: torch.nn.Module, n_episodes: int) -> Generator[Episode, any, any]:
    episodes: List[Episode] = []
    for i in range(n_episodes):
        observation, info = env.reset()
        episode = Episode(steps=[], total_reward=0.0)
        done = False
        truncated = False
        softmax = torch.nn.Softmax(dim=1) #dim=1 because we want to apply softmax to each row of the matrix, as each one is an independent observation
        while (not done and not truncated):
            action_logits = net(torch.Tensor(observation).unsqueeze(0)) #we expecta a matrix with rows being an observation
            action_probs = softmax(action_logits)
            action = np.random.choice([0,1], p=action_probs)
            next_observation, reward, done, truncated, info = env.step(action)
            step = EpisodeStep(step_reward=reward, observation=observation,action=action)
            episode.steps.append(step)
            episode.reward += reward
            observation = next_observation
        episodes.append(episode)
    yield episodes 
    
def filter_batch(episode_list: List[Episode]) -> List[Episode]:
    best_observations = []
    best_actions = []
    minimum_reward = np.percentile([episode.reward for episode in episode_list],q=0.7)
    for episode in episode_list:
        if episode.total_reward > minimum_reward:
            best_observations.append(episode.steps)
            best_actions.append([step.action for step in episode.steps])
    return np.array(best_observations), np.array(best_actions)

if __name__ == "__main__":
    device = 'cpu'
    env = gym.make('CartPole-v1')
    env = gym.wrappers.RecordVideo(env, video_folder="video")
    net = OneLayerDecision(observation_size=env.observation_space.shape[0], n_neurons=HIDDEN_SIZE, n_actions=2)
    optimizer = torch.optim.Adam(params=net.parameters(),lr=LEARNING_RATE, betas=DECAY_RATES)
    objective = torch.nn.CrossEntropyLoss()
    for batch in iterate_batches(env, net, NUMBER_OF_EPISODES_PER_BATCH):
        best_observations, best_actions = filter_batch(batch)
        loss = objective(best_observations, best_actions)
        loss.backward()
        optimizer.step()
            