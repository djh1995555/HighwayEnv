import gymnasium as gym
import highway_env
from stable_baselines3 import DQN

env = gym.make("highway-fast-v0", render_mode="rgb_array")
output_dir = "highway_dqn/"
model_dir = f"{output_dir}/model"
model = DQN('MlpPolicy', env,
              policy_kwargs=dict(net_arch=[256, 256]),
              learning_rate=5e-4,
              buffer_size=15000,
              learning_starts=200,
              batch_size=32,
              gamma=0.8,
              train_freq=1,
              gradient_steps=1,
              target_update_interval=50,
              verbose=1,
              tensorboard_log=output_dir)
model.learn(int(500))
model.save(model_dir)

# Load and test saved model
model = DQN.load(model_dir)
print("Start to test")
while True:
  done = truncated = False
  obs, info = env.reset()
  while not (done or truncated):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    env.render()