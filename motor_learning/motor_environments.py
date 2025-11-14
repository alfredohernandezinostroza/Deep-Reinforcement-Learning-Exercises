import gymnasium as gym
from gymnasium.envs.registration import EnvSpec
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass
from typing import TypedDict
import matplotlib.pyplot as plt
from agents import GaussianAgent
from pathlib import Path
import matplotlib
from PIL import Image
from gymnasium.envs.registration import register

register(
    id="Targets-v0",
    entry_point=f"{__name__}:Targets",
)

def main():
    targets_positions = np.asarray(
                    [[-0.2546,  0.2546],   # Target 1 (top-left)
                     [ 0.2546,  0.2546],    # Target 2 (top-right)
                     [-0.2546, -0.2546],  # Target 3 (bottom-left)
                     [ 0.2546, -0.2546]], np.float32)
    env = gym.make('Targets-v0', targets_positions=targets_positions, training_area=(-0.50, 0.50), max_trials=30, render_mode='rgb_array')
    # env = gym.make('Targets-v0', targets_positions=targets_positions, training_area=(-0.50, 0.50), max_trials=30, render_mode='human')
    # env = Targets(targets_positions, training_area=(-0.50, 0.50), max_trials=30, render_mode='rgb_array')
    # env = Targets(targets_positions, training_area=(-0.50, 0.50), max_trials=20, render_mode='human')
    env = gym.wrappers.RecordVideo(env, Path('motor_learning')/"videos", episode_trigger= lambda _: True)
    agent = [
    GaussianAgent(env, mu=targets_positions[0,:], std=0.2),
    GaussianAgent(env, mu=targets_positions[1,:], std=0.2),
    GaussianAgent(env, mu=targets_positions[2,:], std=0.2),
    GaussianAgent(env, mu=targets_positions[3,:], std=0.2),
    ]
    done = truncated = False
    i = 0
    while not done and not truncated:
        step = agent[i%4].step(target=i % 4)
        print(f"{i}: {step.reward},target={i%4},action={agent[i%4].env.unwrapped.actions_history[-1]['position']}")
        i+=1
        done = step.done
    env.close()
        
class Action(TypedDict):
    position: np.ndarray
    target: int

class Targets(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 1}
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
        self.state = None
        self.fig = None

    def reward_function(self, distance):
        """Could be edited to make more complicated reward functions"""
        return -distance

    def step(self, action: Action) -> tuple[int, float, bool, bool, dict]:
        done = False
        self.actions_history.append(action)
        if len(self.actions_history) >= self.max_trials:
            done = True
        truncated = False
        info = {}
        distance_to_target = np.linalg.norm(action["position"] - self.targets_positions[action["target"]])
        reward = self.reward_function(distance_to_target)
        next_target = self.next_target_generator()
        # if self.render_mode == "human":
        #     self.render()
        return next_target, reward, done, truncated, info
    
    def render(self):
        if self.fig == None:
            self.fig = plt.figure(figsize=(5, 5), dpi=80)
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
            print("add action!")
        if self.render_mode == "human":
            plt.ion()  # Interactive mode
            plt.pause(0.1)
        
        if self.render_mode == "rgb_array":
            print("converting ro image!")
            self.fig.canvas.draw()
            image = np.frombuffer(self.fig.canvas.buffer_rgba(), dtype=np.uint8)
            width, height = self.fig.canvas.get_width_height()
            image = image.reshape((height, width, 4))  # Use actual dimensions
            image = image[:, :, :3]  # Drop alpha channel for RGB
            # img = Image.fromarray(image)
            # img.save("your_file.png")

            # plt.close(fig)
            return image

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.actions_history = []
        self.state  = self.next_target_generator()
        next_target = self.next_target_generator()
        info = {}
        # if self.render_mode == "human":
        #     self.render()
        return next_target, info
    
    # def render(self):
    #     if self.render_mode == "rgb_array":
    #         return self._render_frame()
    #     elif self.render_mode == "human":
    #         if self.fig is None:
    #             plt.ion()  # Enable interactive mode
    #             self.fig, self.ax = plt.subplots(figsize=(6, 6))
    #             plt.show(block=False)
            
    #         # Clear and redraw
    #         self.ax.clear()
            
    #         # Use action_space bounds
    #         x_min = y_min = self.action_space["position"].low[0] 
    #         x_max = y_max = self.action_space["position"].high[0]
    #         self.ax.set_xlim(x_min, x_max)
    #         self.ax.set_ylim(y_min, y_max)
    #         self.ax.set_aspect("equal", adjustable="box")

    #         # Draw targets as large colored dots
    #         for i, pos in enumerate(self.targets_positions):
    #             self.ax.plot(pos[0], pos[1], 'o', markersize=20, label=f"Target {i}", color=plt.cm.tab10(i))
            
    #         # Draw actions as small dots with colors based on target
    #         if self.actions_history:
    #             actions = np.array([a["position"] for a in self.actions_history])
    #             targets = np.array([a["target"] for a in self.actions_history])
    #             self.ax.scatter(actions[:,0], actions[:,1], c=targets, cmap='tab10', s=64)
            
    #         self.ax.grid(True)
            
    #         self.fig.canvas.draw()
    #         self.fig.canvas.flush_events()
    #         plt.pause(0.1)   

    # def render(self):
    #     # print("rendering")
    #     if self.render_mode == "rgb_array":
    #         return self._render_frame()
    #     elif self.render_mode == "human":
    #         plt.ion()  # Enable interactive mode
    #         frame = self._render_frame()
    #         if self.fig is None:
    #             self.fig, ax = plt.subplots(figsize=(6, 6))
    #             self.img = ax.imshow(frame)
    #             ax.axis('off')  # Optional: hide axes for cleaner look
    #             plt.show(block=False)
    #         else:
    #             self.img.set_data(frame)
    #             self.fig.canvas.draw()
    #             self.fig.canvas.flush_events()
    #         plt.pause(0.01)



    # def render(self):
    #     if self.render_mode == "rgb_array":
    #         return self._render_frame()
    #     elif self.render_mode == "human":
    #         frame = self._render_frame()
    #         plt.ion()  # Interactive mode
    #         if self.img is None:
    #             fig, ax = plt.subplots(figsize=(6, 6))
    #             self.img = ax.imshow(frame)
    #             self.fig = fig
    #         else:
    #             self.img.set_data(frame)
    #         plt.pause(0.1)


    # def _render_frame(self):
    #     if self.fig == None:
    #         self.fig = plt.figure(figsize=(5, 5), dpi=80)
    #         self.ax = self.fig.add_subplot(111)
            
    #         # Set plotting limits for your training are
    #         x_min = y_min = self.action_space["position"].low[0] 
    #         x_max = y_max = self.action_space["position"].high[0]
    #         self.ax.set_xlim(x_min, x_max)
    #         self.ax.set_ylim(y_min, y_max)
    #         self.ax.set_aspect("equal", adjustable="box")

    #     # Draw targets as large colored dots
    #     for i, pos in enumerate(self.targets_positions):
    #         self.ax.plot(pos[0], pos[1], 'o', markersize=20, label=f"Target {i}", color=plt.cm.tab10(i))
        
    #     # Draw actions as small dots; action["position"] is expected to be (x, y)
    #     if self.actions_history:
    #         action = np.array(self.actions_history[-1]["position"])
    #         target = np.array(self.actions_history[-1]["target"])
            
    #         self.ax.plot(action[0], action[1], '.', color=plt.cm.tab10(target), markersize=8)
    #         # self.ax.scatter(actions[:,0], actions[:,1], c=targets, cmap='tab10', s=64)
        
    #     # Optionally, add a legend and grid
    #     # ax.legend()
    #     self.ax.grid(True)
        
    #     self.fig.canvas.draw()
    #     image = np.frombuffer(self.fig.canvas.buffer_rgba(), dtype=np.uint8)
    #     width, height = self.fig.canvas.get_width_height()
    #     image = image.reshape((height, width, 4))  # Use actual dimensions
    #     image = image[:, :, :3]  # Drop alpha channel for RGB
    #     # plt.close(fig)
    #     return image
    
    def close(self):
        print("Closing!")
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax  = None
            self.img = None


if __name__ == "__main__":
    main()