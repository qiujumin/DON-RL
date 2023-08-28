import gym

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

env = gym.make("CarRacing-v0")

model = PPO("CnnPolicy", env, verbose=1)
model.learn(total_timesteps=int(2e5), progress_bar=True)
model.save("data/ppo_carracing")
del model

model = PPO.load("data/ppo_carracing", env=env)

mean_reward, std_reward = evaluate_policy(
    model, model.get_env(), n_eval_episodes=10)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render()
