from stable_baselines3 import DQN
from entorno_sumo import SumoEnv
import os

env = SumoEnv()

model_path = "dqn_model_alto.zip"

if os.path.exists(model_path):
    print("Modelo DQN encontrado, cargando...")
    model = DQN.load(model_path, env=env)
else:
    print("No hay modelo DQN previo, creando uno nuevo...")
    model = DQN("MlpPolicy", env, verbose=1, learning_rate=0.001, gamma=0.9875)

model.learn(total_timesteps=1800000)

model.save(model_path)
print("Modelo DQN guardado:", model_path)
