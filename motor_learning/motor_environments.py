import gymnasium as gym
from gymnasium.envs.registration import EnvSpec
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass
from typing import TypedDict
import matplotlib as plt
from agents import GaussianAgent

# from gymnasium.envs.registration import register

# register(
#     id="gymnasium_env/GridWorld-v0",
#     entry_point="gymnasium_env.envs:GridWorldEnv",
# )

def main():
    targets_positions = np.asarray(
                    [[-0.2546,  0.2546],   # Target 1 (top-left)
                     [ 0.2546,  0.2546],    # Target 2 (top-right)
                     [-0.2546, -0.2546],  # Target 3 (bottom-left)
                     [ 0.2546, -0.2546]], np.float32)
    env = Targets(targets_positions, training_area=(-5.0, 5.0), max_trials=800, render_mode='human')
    agent = GaussianAgent(env, targets_positions[0,:])
    done = truncated = False
    while not done and not truncated:
        agent.step()
        

class Action(TypedDict):
    position: np.ndarray
    target: int

class Targets(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 1}
    spec = EnvSpec("Targets-v0")
    def __init__(self,
                 targets_positions: List[np.ndarray],
                 training_area: Tuple,
                 max_trials: int,
                 *,
                 render_mode=None 
                 ):
        super().__init__()
        self.render_mode = render_mode
        self.fig = None
        self.img = None
        self.n_targets = len(targets_positions)
        self.action_space = gym.spaces.Dict(
            {
                "position": gym.spaces.Box(low=training_area[0], high=training_area[1], shape=(2,), dtype=np.float32),
                "target": gym.spaces.Discrete(self.n_targets)
            }
        )
        self.observation_space = gym.spaces.Discrete(self.n_targets)
        assert all(self.action_space["position"].contains(position) for position in targets_positions)
        self.targets_positions = targets_positions
        self.max_trials = max_trials
        self.actions_history = []
        self.state = None

    def reward_function(self, distance):
        """Could be edited to make more complicated reward functions"""
        return -distance

    def step(self, action: Action, next_target: int) -> tuple[int, float, bool, bool, dict]:
        done = False
        self.actions_history.append(action)
        if len(self.actions_history) >= self.max_trials:
            done = True
        truncated = False
        info = None
        distance_to_target = np.linalg.norm(action["position"] - self.targets_positions[action["intention"]])
        reward = self.reward_function(distance_to_target)
        return next_target, reward, done, truncated, info
    
    def reset(self, next_target: int | None = None):
        self.actions_history = []
        self.state = next_target
        return next_target, None
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "human":
            frame = self._render_frame()
            if self.img is None:
                plt.ion()  # Interactive mode
                fig, ax = plt.subplots(figsize=(6, 6))
                self.img = ax.imshow(frame)
                self.fig = fig
            else:
                self.img.set_data(frame)
            plt.pause(0.01)

    def _render_frame(self):
        fig = plt.figure(figsize=(5, 5), dpi=80)
        ax = fig.add_subplot(111)
        
        # Set plotting limits for your training are
        x_min = y_min = self.observation_space.low[0] 
        x_max = y_max = self.observation_space.high[0]
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal", adjustable="box")

        # Draw targets as large colored dots
        for i, pos in enumerate(self.targets_positions):
            ax.plot(pos[0], pos[1], 'o', markersize=20, label=f"Target {i}", color=plt.cm.tab10(i))
        
        # Draw actions as small dots; action["position"] is expected to be (x, y)
        if self.actions_history:
            actions = np.array([a["position"] for a in self.actions_history])
            ax.plot(actions[:,0], actions[:,1], '.', color="black", markersize=8)
        
        # Optionally, add a legend and grid
        # ax.legend()
        ax.grid(True)
        
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return image

if __name__ == "__main__":
    main()