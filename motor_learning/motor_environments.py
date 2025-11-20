import gymnasium as gym
from gymnasium.envs.registration import EnvSpec
import numpy as np
from typing import List, Tuple, TypedDict, Dict
import matplotlib.pyplot as plt
import matplotlib
from gymnasium.envs.registration import register
from dataclasses import dataclass

register(
    id="Targets-v0",
    entry_point=f"{__name__}:Targets",
)

@dataclass
class ActionWithInfo():
    action: np.array
    info: Dict

class Renderer():
    def __init__(self, initial_points: np.ndarray, render_mode: str):
        self.render_mode = render_mode
        if self.render_mode == "rgb_array":
            matplotlib.use('Agg')
        else:
            matplotlib.use('TkAgg')            
            plt.ion()  # Interactive mode
        if self.render_mode == "human":
            self.fig = plt.figure(figsize=(5, 5), dpi=90)
        if self.render_mode == "rgb_array":
            self.fig = plt.figure(figsize=(4.5, 4.5), dpi=200)
        self.ax = self.fig.add_subplot(111)
        self.ax.grid(True)
        self.ax.set_aspect("equal", adjustable="box")
        self.animated_artists = {}
        # Draw targets as large colored dots
        for i, pos in enumerate(initial_points):
            self.ax.plot(pos[0], pos[1], 'o', markersize=20, label=f"Target {i}", color=plt.cm.tab10(i), alpha=0.9, linewidth=0, zorder=0)

    def set_limits(self, x_min, x_max, y_min=None, y_max=None):
        self.ax.set_xlim(x_min, x_max)
        # y_min/y_max may be omitted to use the same limits for both axes
        if (y_min is None) ^ (y_max is None):
            raise TypeError("Either provide both y_min and y_max, or neither.")
        if y_min is not None and y_max is not None:
            self.ax.set_ylim(y_min, y_max)
        else:
            self.ax.set_ylim(x_min, x_max)

    def animate(self, animated_points: Dict):
        if animated_points["position"].shape == (2,):
            animated_points["position"] = np.expand_dims(animated_points["position"], axis=0)
        if self.animated_artists.get(animated_points["artist_id"]):
            self.animated_artists[animated_points["artist_id"]].set_offsets(animated_points["position"])
            self.animated_artists[animated_points["artist_id"]].set_color('black')
        else:
            new_artist = self.ax.scatter(x=animated_points["position"][:,0],y=animated_points["position"][:,1], marker='o', color='black', s=30, zorder=2)
            self.animated_artists[animated_points["artist_id"]] = new_artist

    def render(self, point, color):
        if color is not None:
            self.ax.plot(point[0], point[1], '.', color=plt.cm.tab10(color), markersize=8, zorder=1)

        if self.render_mode == "human":
            plt.pause(0.1)
        
        if self.render_mode == "rgb_array":
            self.fig.canvas.draw()
            image = np.frombuffer(self.fig.canvas.buffer_rgba(), dtype=np.uint8)
            width, height = self.fig.canvas.get_width_height()
            image = image.reshape((height, width, 4))  # Use actual dimensions
            image = image[:, :, :3].copy()  # Drop alpha channel for RGB, and copy to avoid using the same memory adress for all frames
            return image
        
    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax  = None
            self.img = None

class Targets(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 6}
    spec = EnvSpec("Targets-v0",
                   entry_point=None,
                   max_episode_steps=None,
                   reward_threshold=None,
                   nondeterministic=True)
    def __init__(self,
                 targets_positions: List[np.ndarray] = np.asarray(
                    [[-0.2546,  0.2546],   # Target 1 (top-left)
                     [ 0.2546,  0.2546],    # Target 2 (top-right)
                     [-0.2546, -0.2546],  # Target 3 (bottom-left)
                     [ 0.2546, -0.2546]], np.float32),
                 training_area: Tuple = (-0.50, 0.50),
                 max_trials: int = 1000,
                 *,
                 render_mode=None,
                 ):
        # super().__init__()
        self.n_targets = len(targets_positions)
        self.next_target_generator = lambda : np.random.randint(0,self.n_targets)
        # Actions consist only of a 2D position. The current target is provided
        # via the observation (`self.state`) rather than being part of the action.
        self.action_space = gym.spaces.Box(low=training_area[0], high=training_area[1], shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Discrete(self.n_targets)
        assert all(self.action_space.contains(position) for position in targets_positions)
        self.targets_positions = targets_positions
        self.max_trials = max_trials
        self.actions_history = []
        self.states_history = []
        # Make independent lists per target (avoid repeating the same list)
        self.actions_history_by_target = [[] for _ in range(self.n_targets)]
        self.state = None
        assert render_mode is None or render_mode in ["human", "rgb_array"]
        self.render_mode = render_mode
        if render_mode:
            self.renderer: Renderer = Renderer(initial_points=targets_positions, render_mode=render_mode)
            self.renderer.set_limits(self.action_space.low[0], self.action_space.high[0])

    def reward_function(self, distance):
        """Could be edited to make more complicated reward functions"""
        return -distance

    def step(self, action) -> tuple[int, float, bool, bool, dict]:
        """
        Accepts either a raw 2-vector (from `Box`) or a dict with key 'position'.

        The current target is taken from `self.state` (observation). After applying
        the action we advance `self.state` to a newly sampled target and return it
        as the next observation.
        """
        info = {}
        if isinstance(action, ActionWithInfo):
            info =  {
                    "agent_info": action.info
                    }
            action = action.action

        # Record actions and states
        self.actions_history.append(action)
        current_target = int(self.state)
        self.states_history.append(current_target)
        self.actions_history_by_target[current_target].append(action)

        distance_to_target = np.linalg.norm(action - self.targets_positions[current_target])
        reward = self.reward_function(distance_to_target)

        # advance to a new target and return it as the next observation
        next_target = self.next_target_generator()
        self.state = next_target

        # if self.render_mode == "human":
        #     self.render()
        done = False
        if len(self.actions_history) >= self.max_trials:
            done = True
        truncated = False
        return next_target, reward, done, truncated, info
    
    def render(self):
        point = color = None
        if len(self.actions_history) > 0:
            point = np.array(self.actions_history[-1])
            color = np.array(self.states_history[-1])
        return self.renderer.render(point, color)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.actions_history = []
        # choose an initial target and return it as the initial observation
        init_target = self.next_target_generator()
        self.state = init_target
        self.states_history.append(init_target)
        info = {}
        return init_target, info
    
    def close(self):
        self.renderer.close()

