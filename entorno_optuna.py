import gymnasium as gym
from gymnasium import spaces
import numpy as np
import traci
import os
import xml.etree.ElementTree as ET
import random
import shutil

class SumoEnv(gym.Env):
    def __init__(self, sumocfg_path="Simulación/osm.sumocfg", rutas_dir="Simulación/rutas", max_steps=1800, modo_evaluacion=False):
        super().__init__()
        self.max_steps = max_steps
        self.step_count = 0
        self.modo_evaluacion = modo_evaluacion

        self.tls_id = "cluster_10263379069_10263379072_10263379088_10263379094_#1more"

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
        self.observation_edges = list(self.main_edges.values()) + [e for e in self.extra_edges.values() if e]

        self.sumocfg_original = sumocfg_path
        self.rutas_dir = rutas_dir
        self.temp_cfg = "Simulación/temp.sumocfg"

        self.observation_space = spaces.Box(low=0, high=100, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)

        self.prev_edge_map = {}
        self.prev_speeds = {}

        self.total_waiting_time = 0
        self.total_queue_length = 0
        self.total_stops_real = 0
        self.vehiculos_salidos = 0
        self.vehiculos_unicos = set()

    def reset(self, seed=None, options=None):
        self.step_count = 0
        self.prev_edge_map = {}
        self.prev_speeds = {}

        self.total_waiting_time = 0
        self.total_queue_length = 0
        self.total_stops_real = 0
        self.vehiculos_salidos = 0
        self.vehiculos_unicos = set()

        self._generar_configuracion_temporal()

        if traci.isLoaded():
            traci.close()

        traci.start(["sumo", "-c", self.temp_cfg, "--no-step-log", "true"])
        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1

        if action == 1:
            actual = traci.trafficlight.getPhase(self.tls_id)
            traci.trafficlight.setPhase(self.tls_id, (actual + 1) % 2)

        self._remove_crossed_vehicles()
        traci.simulationStep()

        self.total_queue_length += sum(traci.edge.getLastStepHaltingNumber(e) for e in self.edge_ids)
        step_waiting_time = sum(traci.edge.getWaitingTime(e) for e in self.edge_ids)
        self.total_waiting_time += step_waiting_time

        for veh in traci.vehicle.getIDList():
            speed = traci.vehicle.getSpeed(veh)
            self.vehiculos_unicos.add(veh)

            if veh in self.prev_speeds and self.prev_speeds[veh] > 0.1 and speed < 0.1:
                self.total_stops_real += 1

            self.prev_speeds[veh] = speed

        obs = self._get_obs()
        reward = -step_waiting_time
        terminated = self.step_count >= self.max_steps or traci.simulation.getMinExpectedNumber() <= 0

        return obs, reward, terminated, False, {}

    def _get_obs(self):
        occ_or = ["-958496039#1", "958496040#1"]
        nor_sur = ["89987002#2", "-958496038#1"]

        occ_or_total = sum(traci.edge.getLastStepVehicleNumber(e) for e in occ_or)
        nor_sur_total = sum(traci.edge.getLastStepVehicleNumber(e) for e in nor_sur)

        return np.array([occ_or_total, nor_sur_total], dtype=np.float32)

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

    def close(self):
        if traci.isLoaded():
            traci.close()

    def _generar_configuracion_temporal(self):
        archivos = [f for f in os.listdir(self.rutas_dir) if f.endswith(".rou.xml")]
        seleccionado = random.choice(archivos)
        ruta_ruta = os.path.abspath(os.path.join(self.rutas_dir, seleccionado)).replace("\\", "/")

        shutil.copy(self.sumocfg_original, self.temp_cfg)
        tree = ET.parse(self.temp_cfg)
        root = tree.getroot()
        for nodo_input in root.findall("input"):
            rutas_tag = nodo_input.find("route-files")
            if rutas_tag is not None:
                rutas_tag.set("value", ruta_ruta)
        tree.write(self.temp_cfg)
