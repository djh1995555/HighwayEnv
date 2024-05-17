import gymnasium as gym
from collections import OrderedDict
from controller.pid_controller import PIDController

env_name = "racetrack-v0"
env = gym.make(env_name, render_mode="rgb_array")
controller = PIDController()

debug_info = OrderedDict()
while True:
  done = truncated = False
  obs, info = env.reset()
  i = 0
  while not (done or truncated):
    action = controller.compute_control_cmd(info['lat_err'])
    obs, reward, done, truncated, info = env.step(action)

    env.render()
    i += 1