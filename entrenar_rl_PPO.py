from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from entorno_sumo import SumoEnv
import os

env = make_vec_env(lambda: SumoEnv(), n_envs=1)

model_path = "ppo_model_bajo.zip"

if os.path.exists(model_path):
    print("Modelo encontrado, cargando...")
    model = PPO.load(model_path, env=env)
else:
    print("No hay modelo previo, creando uno nuevo...")
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.001, gamma=0.9875)

model.learn(total_timesteps=360000)

model.save(model_path)
print("Modelo guardado en:", model_path)




