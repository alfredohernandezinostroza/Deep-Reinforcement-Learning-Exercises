import typing
import torch 
import torch.nn
import torch.optim
import gymnasium as gym
 
HIDDEN_SIZE = 128 
BATCH_SIZE = 16 
PERCENTILE = 70
LEARNING_RATE = 0.5
DECAY_RATES = (0.5, 0.999)
NUMBER_OF_EPISODES = 200
NUMBER_OF_AGENTS = 100

class OneLayerDecision(torch.nn.Module):
    def __init__(self, n_neurons):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(in_features=n_neurons,out_features=n_neurons),
            torch.nn.ReLU(in_place = True),
            torch.nn.Sigmoid()
        )
if __name__ == "__main__":
    device = 'cpu'
    env = gym.make('CartPole-v1')
    decision_makers = [OneLayerDecision(HIDDEN_SIZE) for i in range(NUMBER_OF_AGENTS)]
    optimizers = [torch.optim.Adam(params=decision_makers[i].parameters(),lr=LEARNING_RATE, betas=DECAY_RATES) for i in range(NUMBER_OF_AGENTS)]
    objective = torch.nn.BCELoss()
    for i in range(NUMBER_OF_AGENTS):
        decision_maker = decision_makers[i].to(device)
        optimizer = optimizers[i]
        env.reset()
        for j in range(NUMBER_OF_EPISODES):
            rewards = 0
            actions = []
            done = False
            truncated = False
            input_vector = torch.randn(HIDDEN_SIZE, device=device, dtype=torch.float16)
            action = decision_maker(input_vector)
            while (not done and not truncated):
                observation, reward, done, truncated, info = env.step(action)
                actions.append(action)
                rewards.append(reward)
                # cart_position = observation[0]
                # cart_velocity = observation[1]
                # pole_angle = observation[2]
                # pole_angular_velocity = observation[3]
            loss = objective()
            loss.backward()
            optimizer.step()
            