from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from entorno_sumo import SumoEnv
import os

env = make_vec_env(SumoEnv, n_envs=1)

model_path = "a2c_model_bajo.zip"

if os.path.exists(model_path):
    print("Modelo A2C encontrado, cargando...")
    model = A2C.load(model_path, env=env)
else:
    print("No hay modelo A2C previo, creando uno nuevo...")
    model = A2C("MlpPolicy", env, verbose=1, learning_rate=0.001, gamma=0.9875)

model.learn(total_timesteps=360000)

model.save(model_path)
print("Modelo A2C guardado en:", model_path)
