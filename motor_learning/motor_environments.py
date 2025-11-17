import gymnasium as gym
from gymnasium.envs.registration import EnvSpec
import numpy as np
from typing import List, Tuple
from typing import TypedDict
import matplotlib.pyplot as plt
import matplotlib
from gymnasium.envs.registration import register

register(
    id="Targets-v0",
    entry_point=f"{__name__}:Targets",
)
        
class Action(TypedDict):
    position: np.ndarray
    target: int

class Targets(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 6}
    spec = EnvSpec("Targets-v0",
                   entry_point=None,
                   max_episode_steps=None,
                   reward_threshold=None,
                   nondeterministic=True)
    def __init__(self,
                 targets_positions: List[np.ndarray],
                 training_area: Tuple,
                 max_trials: int,
                 *,
                 render_mode=None 
                 ):
        # super().__init__()
        assert self.render_mode is None or self.render_mode in ["human", "rgb_array"]
        self.render_mode = render_mode
        if self.render_mode == "rgb_array":
            matplotlib.use('Agg')
        else:
            matplotlib.use('TkAgg')            
            plt.ion()  # Interactive mode
        self.fig = None
        self.ax = None
        self.img = None
        self.n_targets = len(targets_positions)
        self.next_target_generator = lambda : np.random.randint(0,self.n_targets)
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
        self.actions_history_by_target = [[]] * self.n_targets
        self.state = None
        self.fig = None

    def reward_function(self, distance):
        """Could be edited to make more complicated reward functions"""
        return -distance

    def step(self, action: Action) -> tuple[int, float, bool, bool, dict]:
        done = False
        self.actions_history.append(action)
        self.actions_history_by_target[action["target"]].append(action["position"])
        if len(self.actions_history) >= self.max_trials:
            done = True
        truncated = False
        info = {}
        distance_to_target = np.linalg.norm(action["position"] - self.targets_positions[action["target"]])
        reward = self.reward_function(distance_to_target)
        next_target = self.next_target_generator()
        if self.render_mode == "human":
            self.render()
        return next_target, reward, done, truncated, info
    
    def render(self):
        if self.fig is None:
            if self.render_mode == "human":
                self.fig = plt.figure(figsize=(5, 5), dpi=90)
            if self.render_mode == "rgb_array":
                self.fig = plt.figure(figsize=(4.5, 4.5), dpi=200)
            self.ax = self.fig.add_subplot(111)
            self.ax.grid(True)
            
            # Set plotting limits for your training are
            x_min = y_min = self.action_space["position"].low[0] 
            x_max = y_max = self.action_space["position"].high[0]
            self.ax.set_xlim(x_min, x_max)
            self.ax.set_ylim(y_min, y_max)
            self.ax.set_aspect("equal", adjustable="box")

            # Draw targets as large colored dots
            for i, pos in enumerate(self.targets_positions):
                self.ax.plot(pos[0], pos[1], 'o', markersize=20, label=f"Target {i}", color=plt.cm.tab10(i))
        if len(self.actions_history) > 0:
            last_action = np.array(self.actions_history[-1]["position"])
            last_target = np.array(self.actions_history[-1]["target"])
            self.ax.plot(last_action[0], last_action[1], '.', color=plt.cm.tab10(last_target), markersize=8)
        if self.render_mode == "human":
            plt.pause(0.1)
        
        if self.render_mode == "rgb_array":
            self.fig.canvas.draw()
            image = np.frombuffer(self.fig.canvas.buffer_rgba(), dtype=np.uint8)
            width, height = self.fig.canvas.get_width_height()
            image = image.reshape((height, width, 4))  # Use actual dimensions
            image = image[:, :, :3].copy()  # Drop alpha channel for RGB, and copy to avoid using the same memory adress for all frames
            return image

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.actions_history = []
        self.state  = self.next_target_generator()
        next_target = self.next_target_generator()
        info = {}
        if self.render_mode == "human":
            self.render()
        return next_target, info
    
    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax  = None
            self.img = None

