import os
from stable_baselines3 import A2C
import gymnasium as gym
import motor_environments #for registering the Targets gym env
"""sb3 is made to trainf first with .learning(n_steps) (with n_steps usually high, on the order of 10^4), and then to show the agent's performance with a loop in a new environment.
We don't care about that: we want to visualize and get the data DURING the training. Thus, in this code we try to bypass this limitation
by running .learning(1) inside a loop so we can render the environment."""


# 1. Define the logging directory
TENSORBOARD_LOG_DIR = "./a2c_targets_log/"
VIDEOS = "./targets_sb3_videos/"
# 2. Create the model and pass the tensorboard_log argument
# The 'learn' call will now save results to the specified directory
model = A2C(
    "MlpPolicy", 
    "Targets-v0", 
    verbose=0, # Set to 1 to see basic terminal output during training
    tensorboard_log=TENSORBOARD_LOG_DIR,
device='cpu' #according to the docs it's better to train on cpu
)

# Create the environment
eval_env = gym.make("Targets-v0", render_mode="rgb_array")
wrapped_env = gym.wrappers.RecordVideo(
    eval_env, 
    VIDEOS,
    # Here, we record the very first episode (id 0) we run
    # episode_trigger=lambda episode_id: episode_id == 0 
)

# --- 3. Run the trained model in the wrapped environment ---
obs, info = wrapped_env.reset()
done = False    
i = 0
while not done:
    # Use deterministic=True for a consistent, final policy
    action, _ = model.predict(obs, deterministic=True) 
    obs, reward, terminated, truncated, info = wrapped_env.step(action)
    done = terminated or truncated
    wrapped_env.render()
    print(i)
    i += 1

# Close the wrapper and underlying environment
wrapped_env.close() 

print(f"Video saved successfully in the '{VIDEOS}' directory!")