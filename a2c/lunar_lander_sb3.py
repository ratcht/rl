import gymnasium as gym
from aim.sb3 import AimCallback
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy


def train():
  env = make_vec_env("LunarLander-v3", n_envs=8)

  model = A2C(
    "MlpPolicy",
    env,
    gamma=0.995,
    n_steps=5,
    learning_rate=7e-4,
    ent_coef=1e-4,
    verbose=1,
  )

  model.learn(
    total_timesteps=200_000,
    progress_bar=True
  )
  model.save("a2c_lunarlander")

  eval_env = make_vec_env("LunarLander-v3", n_envs=1)
  mean, std = evaluate_policy(model, eval_env, n_eval_episodes=20)
  print(f"mean reward: {mean:.1f} +/- {std:.1f}")
  return model


def watch(model, episodes=3):
  env = gym.make("LunarLander-v3", render_mode="human")
  for _ in range(episodes):
    obs, _ = env.reset()
    done = False
    while not done:
      action, _ = model.predict(obs, deterministic=True)
      obs, _, terminated, truncated, _ = env.step(action)
      done = terminated or truncated
  env.close()


if __name__ == "__main__":
  model = train()
  watch(model)