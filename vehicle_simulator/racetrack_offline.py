import gymnasium as gym
import highway_env
import datetime
from stable_baselines3 import DDPG
from stable_baselines3 import SAC
from stable_baselines3 import TD3

env_name = "racetrack-v0"
model = "sac"
TRAIN = True

env = gym.make(env_name, render_mode="rgb_array")
output_dir = f"output/model/{env_name}_{model}/{datetime.datetime.now()}"
model_filepath = f"{output_dir}/model"
if(TRAIN):
  if(model == "ddpg"):
    model = DDPG('MlpPolicy', env,
                  policy_kwargs=dict(net_arch=[256, 256]),
                  learning_rate=5e-4,
                  buffer_size=15000,
                  learning_starts=200,
                  batch_size=32,
                  gamma=0.8,
                  train_freq=1,
                  gradient_steps=1,
                  verbose=1,
                  tensorboard_log=output_dir)
    model.learn(int(5000))
  elif(model == "sac"):
    model = SAC('MlpPolicy', env,
                  policy_kwargs=dict(net_arch=[256, 256]),
                  learning_rate=5e-4,
                  buffer_size=15000,
                  learning_starts=200,
                  batch_size=32,
                  gamma=0.8,
                  train_freq=1,
                  gradient_steps=1,
                  verbose=1,
                  tensorboard_log=output_dir)
    model.learn(int(200))
  elif(model == "td3"):
    model = TD3('MlpPolicy', env,
                  policy_kwargs=dict(net_arch=[256, 256]),
                  learning_rate=5e-4,
                  buffer_size=15000,
                  learning_starts=200,
                  batch_size=32,
                  gamma=0.8,
                  train_freq=1,
                  gradient_steps=1,
                  verbose=1,
                  tensorboard_log=output_dir)
    model.learn(int(20000))
  model.save(model_filepath)

# Load and test saved model
if(model == "ddpg"):
  model = DDPG.load(model_filepath)
elif(model == "sac"):
  model = SAC.load(model_filepath)
elif(model == "td3"):
  model = TD3.load(model_filepath)
  
print("Start to test")
while True:
  done = truncated = False
  obs, info = env.reset()
  while not (done or truncated):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    env.render()