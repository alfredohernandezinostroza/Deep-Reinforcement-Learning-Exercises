import os
from stable_baselines3 import A2C
import gymnasium as gym
import motor_environments #for registering the Targets gym env

# 1. Define the logging directory
TENSORBOARD_LOG_DIR = "./a2c_targets_log/"
VIDEOS = "./targets_sb3_videos/"
# 2. Create the model and pass the tensorboard_log argument
# The 'learn' call will now save results to the specified directory
env = gym.make("Targets-v0", render_mode="human")
env = motor_environments.RenderOnStepWrapper(env)
model = A2C(
    "MlpPolicy", 
    env,
    verbose=0, # Set to 1 to see basic terminal output during training
    tensorboard_log=TENSORBOARD_LOG_DIR,
device='cpu' #according to the docs it's better to train on cpu
).learn(30_000)

print(f"Training finished. TensorBoard logs saved in: {os.path.abspath(TENSORBOARD_LOG_DIR)}")

# Create the environment
env = gym.make("Targets-v0", render_mode="human")
# env = gym.wrappers.RecordVideo(
#     env, 
#     VIDEOS,
#     # Here, we record the very first episode (id 0) we run
#     # episode_trigger=lambda episode_id: episode_id == 0 
# )

# --- 3. Run the trained model in the wrapped environment ---
obs, info = env.reset()
done = False
i = 0
while not done:
    # Use deterministic=True for a consistent, final policy
    action, _ = model.predict(obs, deterministic=False) 
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    env.render()
    print(i)
    i += 1

# Close the wrapper and underlying environment
env.close() 

print(f"Video saved successfully in the '{VIDEOS}' directory!")