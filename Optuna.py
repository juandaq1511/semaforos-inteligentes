import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from entorno_sumo import SumoEnv
import os
import numpy as np

BEST_MODEL_PATH = "best_ppo_optuna_model_alto.zip"

# Función de evaluación personalizada
def evaluate_model(model, n_episodes=20):
    rewards = []
    for _ in range(n_episodes):
        env = SumoEnv(modo_evaluacion=True) 
        obs, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated

        rewards.append(total_reward)
        env.close()

    return np.mean(rewards)

# Función objetivo para Optuna
def objective(trial):
    # Sugerir hiperparámetros
    learning_rate = 0.001
    gamma = trial.suggest_float("gamma", 0.90, 0.99)

    # Crear entorno vectorizado
    env = make_vec_env(lambda: SumoEnv(modo_evaluacion=True), n_envs=1)

    # Crear y entrenar el modelo
    model = PPO("MlpPolicy", env, learning_rate=learning_rate, gamma=gamma, verbose=0)
    model.learn(total_timesteps=360000)
    env.close()

    # Evaluar el modelo
    mean_reward = evaluate_model(model, n_episodes=5)

    # Guardar el mejor modelo seguro (evita error en trial 0)
    if trial.number == 0 or mean_reward > trial.study.best_value:
        model.save(BEST_MODEL_PATH)

    return mean_reward

if __name__ == "__main__":
    # Activar pruning para evitar malos trials
    pruner = optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=1)
    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(objective, n_trials=5)

    print("\n📈 Mejor resultado encontrado:")
    print("Recompensa promedio:", study.best_value)
    print("Hiperparámetros óptimos:", study.best_params)
    print("Modelo guardado en:", BEST_MODEL_PATH)
