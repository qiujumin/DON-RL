from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import CUSTOM_MOVEMENT

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
env = JoypadSpace(env, CUSTOM_MOVEMENT)

model = PPO("CnnPolicy", env, verbose=1)
model.learn(total_timesteps=int(2e5), progress_bar=True) 
model.save("data/ppo_supermariobros")
del model

model = PPO.load("data/ppo_supermariobros", env=env)
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render()
