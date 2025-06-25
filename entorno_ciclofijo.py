import gymnasium as gym
from gymnasium import spaces
import numpy as np
import traci
import os
import xml.etree.ElementTree as ET
import random
import shutil
import time

class SumoEnv(gym.Env):
    def __init__(self, sumocfg_path="Simulación/osm.sumocfg", rutas_dir="Simulación/rutas", max_steps=1800):
        super().__init__()
        self.max_steps = max_steps
        self.step_count = 0
        self.tls_id = "cluster_10263379069_10263379072_10263379088_10263379094_#1more"

        self.time_in_sn = 0
        self.time_in_oo = 0
        self.current_phase = 0

        self.main_edges = {
            "sur": "-958496038#1",
            "occidente": "-958496039#1",
            "norte": "89987002#2",
            "oriente": "958496040#1"
        }
        self.extra_edges = {
            "sur": "-1013171321#1",
            "occidente": "-89986459#0",
            "norte": None,
            "oriente": "25444693#3"
        }

        self.edge_ids = list(self.main_edges.values())
        self.observation_edges = self.edge_ids + [e for e in self.extra_edges.values() if e]

        self.sumocfg_original = sumocfg_path
        self.rutas_dir = rutas_dir
        self.temp_cfg = "Simulación/temp.sumocfg"

        self.observation_space = spaces.Box(low=0, high=max_steps, shape=(2,), dtype=np.float32)

        self.log_file = "estadisticas_ciclofijo_alto.csv"
        self.episodio = 1
        if os.path.exists(self.log_file):
            with open(self.log_file, "r") as f:
                self.episodio = sum(1 for _ in f)
        else:
            with open(self.log_file, "w") as f:
                f.write("episodio,espera_total,cola_total,vehiculos,paradas,tiempo_prom_sistema,longitud_prom_cola,paradas_prom_vehiculo,throughput\n")
        print(f"Archivo CSV listo en: {os.path.abspath(self.log_file)}")

        self.total_waiting_time = 0
        self.total_queue_length = 0
        self.total_stops = 0
        self.vehiculos_salidos = 0
        self.prev_edge_map = {}
        self.prev_speeds = {}
        self.vehiculos_unicos = set()

    def reset(self, seed=None, options=None):
        self.step_count = 0
        self.time_in_sn = 0
        self.time_in_oo = 0
        self.current_phase = 0

        self.total_waiting_time = 0
        self.total_queue_length = 0
        self.total_stops = 0
        self.vehiculos_salidos = 0
        self.prev_edge_map = {}
        self.prev_speeds = {}
        self.vehiculos_unicos = set()

        rutas = [f for f in os.listdir(self.rutas_dir) if f.endswith(".rou.xml")]
        sel = random.choice(rutas)
        ruta_full = os.path.abspath(os.path.join(self.rutas_dir, sel)).replace("\\", "/")
        shutil.copy(self.sumocfg_original, self.temp_cfg)
        tree = ET.parse(self.temp_cfg)
        root = tree.getroot()
        for inp in root.findall("input"):
            rf = inp.find("route-files")
            if rf is not None:
                rf.set("value", ruta_full)
        tree.write(self.temp_cfg)

        if traci.isLoaded():
            traci.close()
        traci.start(["sumo", "-c", self.temp_cfg, "--no-step-log", "true"])

        traci.trafficlight.setPhase(self.tls_id, 0)
        print(f"Episodio {self.episodio} iniciado")
        return self._get_obs(), {}

    def step(self, _action=None):
        traci.simulationStep()
        self.step_count += 1

        if self.current_phase == 0:
            self.time_in_sn += 1
        else:
            self.time_in_oo += 1

        self.total_queue_length += sum(traci.edge.getLastStepHaltingNumber(e) for e in self.edge_ids)
        self.total_waiting_time += sum(traci.edge.getWaitingTime(e) for e in self.edge_ids)

        for veh in traci.vehicle.getIDList():
            self.vehiculos_unicos.add(veh)
            speed = traci.vehicle.getSpeed(veh)
            if veh in self.prev_speeds and self.prev_speeds[veh] > 0.1 and speed < 0.1:
                self.total_stops += 1
            self.prev_speeds[veh] = speed

        self._remove_crossed_vehicles()

        if self.current_phase == 0 and self.time_in_sn >= 50:
            self._switch_phase(1)
        elif self.current_phase == 1 and self.time_in_oo >= 35:
            self._switch_phase(0)

        obs = self._get_obs()
        reward = -sum(traci.edge.getLastStepVehicleNumber(e) for e in self.edge_ids)
        done = self.step_count >= self.max_steps or traci.simulation.getMinExpectedNumber() <= 0

        if done:
            veh_total = len(self.vehiculos_unicos)
            tiempo_prom_sistema = self.total_waiting_time / veh_total if veh_total > 0 else 0
            longitud_prom_cola = self.total_queue_length / self.step_count if self.step_count > 0 else 0
            paradas_prom_vehiculo = self.total_stops / veh_total if veh_total > 0 else 0
            throughput = (self.vehiculos_salidos * 3600) / self.step_count if self.step_count > 0 else 0

            with open(self.log_file, "a") as f:
                f.write(f"{self.episodio},{self.total_waiting_time},{self.total_queue_length},{veh_total},{self.total_stops},{tiempo_prom_sistema},{longitud_prom_cola},{paradas_prom_vehiculo},{throughput}\n")

            print(f"Episodio {self.episodio} terminado - CSV actualizado")
            self.episodio += 1

        return obs, reward, done, False, {}

    def _remove_crossed_vehicles(self):
        veh_ids = traci.vehicle.getIDList()
        for veh_id in veh_ids:
            current_edge = traci.vehicle.getRoadID(veh_id)
            prev_edge = self.prev_edge_map.get(veh_id)
            if prev_edge in self.edge_ids and current_edge != prev_edge:
                try:
                    traci.vehicle.remove(veh_id)
                    self.vehiculos_salidos += 1
                except:
                    pass
            self.prev_edge_map[veh_id] = current_edge

    def _switch_phase(self, phase):
        traci.trafficlight.setPhase(self.tls_id, phase)
        self.current_phase = phase
        self.time_in_sn = 0
        self.time_in_oo = 0

    def _get_obs(self):
        return np.array([self.time_in_sn, self.time_in_oo], dtype=np.float32)

    def close(self):
        if traci.isLoaded():
            traci.close()

NUM_EPISODIOS = 100
env = SumoEnv()

for i in range(NUM_EPISODIOS):
    print(f"\nSimulando Episodio {i+1}/{NUM_EPISODIOS}")
    obs, _ = env.reset()
    done = False
    while not done:
        obs, reward, done, _, _ = env.step()
        time.sleep(0.001)

env.close()
print("\nTodas las simulaciones completadas.")
