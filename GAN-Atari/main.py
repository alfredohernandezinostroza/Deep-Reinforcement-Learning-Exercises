from torch import Discriminator
import gymnasium as gym
import cv2
import numpy as np

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
        image_size = (64, 64)
        new_observation = cv2.resize(observation, image_size)
        # transform image from (width, height, color) -> (color, width, height) 
        # new_observation = np.moveaxis(new_observation, source=0, destination=2) # I accidentaly swtiched them up, I wonder what would come out of the result if I kept it?
        new_observation = np.moveaxis(new_observation, source=2, destination=0)
        return new_observation.astype(np.float32)


        
if __name__ == "__main__":

    envs = [InputWrapper('Breakout-v4'), InputWrapper('Pong-v4'), InputWrapper('AirRaid-v4')]
    shape = envs[0].observation_space.shape