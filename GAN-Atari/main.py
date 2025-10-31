import torch
import gymnasium as gym
import cv2
import numpy as np
import types
from tensorboard import SummaryWriter
from random import random
import time
import torchvision 

IMAGE_SIZE = 64
LEARNING_RATE=0.1
BATCH_SIZE=100
LATENT_VECTOR_SIZE=64
REPORT_EVERY_ITER = 100
SAVE_IMAGE_EVERY_ITER = 100

class InputWrapper(gym.ObservationWrapper):

    def __init__(self, env: gym.Env):
        super().__init__(env)
        old_space: gym.spaces.Box = self.observation_space
        assert isinstance(old_space, gym.spaces.Box)
        self.observation_space = gym.spaces.Box(
            self.observation(old_space.low),
            self.observation(old_space.high),
            dtype=np.float32
        )

    def observation(self, observation: gym.core.ObsType) -> gym.core.ObsType:
        image_size = (IMAGE_SIZE, IMAGE_SIZE)
        new_observation = cv2.resize(observation, image_size)
        # transform image from (width, height, color) -> (color, width, height) 
        # new_observation = np.moveaxis(new_observation, source=0, destination=2) # I accidentaly swtiched them up, I wonder what would come out of the result if I kept it?
        new_observation = np.moveaxis(new_observation, source=2, destination=0)
        return new_observation.astype(np.float32)

class Discriminator64(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=33),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
            torch.nn.Conv2d(in_channels=3, out_channels=2, kernel_size=17),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
            torch.nn.Conv2d(in_channels=2, out_channels=2, kernel_size=9),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
            torch.nn.Conv2d(in_channels=2, out_channels=2, kernel_size=5),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
            torch.nn.Conv2d(in_channels=2, out_channels=1, kernel_size=4),
            torch.nn.Sigmoid()
        )
    def forward(self,x):
        return self.layers(x)

class Generator64(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=1, out_channels=2, kernel_size=4),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=5),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=9),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(in_channels=2, out_channels=3, kernel_size=17),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=33),
            torch.nn.Tanh()
        )
def iterate_batches(envs: types.List[gym.Env],
                     batch_size: int = BATCH_SIZE) -> types.Generator[torch.Tensor, None, None]: 
    batch = [e.reset()[0] for e in envs] 
    env_gen = iter(lambda: random.choice(envs), None)  
    while True: 
        e = next(env_gen) 
        action = e.action_space.sample() 
        obs, reward, is_done, is_trunc, _ = e.step(action) 
        if np.mean(obs) > 0.01: 
            batch.append(obs) 
        if len(batch) == batch_size: 
            batch_np = np.array(batch, dtype=np.float32) 
            # Normalising input to [-1..1] 
            yield torch.tensor(batch_np * 2.0 / 255.0 - 1.0) 
            batch.clear() 
        if is_done or is_trunc: 
            e.reset()

if __name__ == "__main__":
    envs = [
        InputWrapper(gym.make('Breakout-v4')),
        InputWrapper(gym.make('Pong-v4')),
        InputWrapper(gym.make('AirRaid-v4'))
        ]
    shape = envs[0].observation_space.shape
    device = 'cpu'
    
    generator: torch.nn.Module = Generator64().to(device)
    discriminator: torch.nn.Module = Discriminator64().to(device)
    objective = torch.nn.BCELoss()

    generator_optimizer = torch.optim.Adam(params=generator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    discriminator_optimizer = torch.optim.Adam(params=discriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

    writer = SummaryWriter()
    generator_losses = []
    discriminator_losses = []
    iter_no = 0

    fake_labels = torch.zeros(BATCH_SIZE, device=device)
    true_labels = torch.ones(BATCH_SIZE, device=device)

    for batch in iterate_batches(envs):
        batch = batch.to(device)
        latent_vector = torch.randn(BATCH_SIZE, LATENT_VECTOR_SIZE, 1, 1, device=device)  #conv2d expects 4D input
        #we don't normalize because torch.randn give normalized values already
        
        generator_output = generator(latent_vector)

        # train discriminator
        discriminator_optimizer.zero_grad()
        discriminator_fake_output = discriminator(generator_output.detach()) #we don't want to update the generator's gradients
        discriminator_true_output = discriminator(batch)
        discriminator_loss = objective(discriminator_fake_output, fake_labels) + objective(discriminator_true_output, true_labels)
        discriminator_loss.backward() #calculates gradients for the whole graph
        discriminator_optimizer.step() #update weights according to gradients

        discriminator_losses.append(discriminator_loss.item())

        # train generator
        generator_optimizer.zero_grad()
        discriminator_teaching_output = discriminator(generator_output)
        generator_loss = objective(discriminator_teaching_output, true_labels)
        generator_loss.backward()
        generator_optimizer.step()
        # we don't do discriminator_optimizer.step(), so even though the gradients have been calculated,
        # it doesn't matter, since we never updated tbe weights
        generator_losses.append(generator_loss.item())

        # logging
        iter_no += 1 
        if iter_no % REPORT_EVERY_ITER == 0: 
            dt = time.time() - ts_start 
            log.info("Iter %d in %.2fs: generator_loss=%.3e, discriminator_loss=%.3e", 
                     iter_no, dt, np.mean(generator_losses), np.mean(discriminator_losses)) 
            ts_start = time.time() 
            writer.add_scalar("generator_loss", np.mean(generator_losses), iter_no) 
            writer.add_scalar("discriminator_loss", np.mean(discriminator_losses), iter_no) 
            generator_losses = [] 
            discriminator_losses = [] 
        if iter_no % SAVE_IMAGE_EVERY_ITER == 0: 

            img = torchvision.vutils.make_grid(generator_output.detach()[:64], normalize=True) 
            writer.add_image("fake", img, iter_no) 
            img = torchvision.vutils.make_grid(batch.detach()[:64], normalize=True)

            writer.add_image("real", img, iter_no)
