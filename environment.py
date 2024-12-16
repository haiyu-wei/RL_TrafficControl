import os
import traci
import gym
from gym import spaces
import numpy as np


class SumoEnvironment(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, sumo_config, net_file, route_file, gui=True, max_steps=1000):
        super(SumoEnvironment, self).__init__()
        self.sumo_config_path = sumo_config
        self.net_file_path = net_file
        self.route_file_path = route_file
        self.gui = gui
        self.sumo_binary = "sumo-gui" if gui else "sumo"

        self.max_steps = max_steps
        self.current_step = 0

        self.action_space = spaces.Discrete(4)  # Assuming 4 possible phases for the traffic light
        # State: Let's assume we use the first 4 queue lengths as the observation
        self.observation_space = spaces.Box(low=0, high=100, shape=(20,), dtype=np.float32)
        self.ts_ids = None

        self.sumo = None
        self.conn = None

    def start_sumo(self):
        """
        Start Sumo Simulator
        """
        if traci.isLoaded():
            traci.close()

        sumo_launch_args = [
            self.sumo_binary,
            "-c", self.sumo_config_path,
            "-n", self.net_file_path,
            "-r", self.route_file_path
        ]
        print(f"Starting SUMO with command: {' '.join(sumo_launch_args)}")
        self.sumo = traci.start(sumo_launch_args)
        self.conn = traci.getConnection()
        self.current_step = 0
        self.ts_ids = list(self.conn.trafficlight.getIDList())

    def reset(self):
        self.sumo.close()
        self.start_sumo()
        state = self.get_state_array()
        return state

    def step(self, action):
        # Advance the simulation by 10 steps (for example)
        for _ in range(10):
            for ts in self.ts_ids:
                self.conn.trafficlight.setPhase(ts, action)
                # print(f"set traffic light: {ts} to {action}")
            self.conn.simulationStep()
            self.current_step += 1

        state = self.get_state_array()
        reward = self.calculate_reward()
        done = self.current_step >= self.max_steps
        info = {}

        return state, reward, done, info

    def get_state(self):
        vehicles = traci.vehicle.getIDList()
        num_vehicles =[traci.edge.getLastStepVehicleNumber(edge) for edge in traci.edge.getIDList()]
        queue_lengths = [traci.edge.getLastStepHaltingNumber(edge) for edge in traci.edge.getIDList()]
        avg_speed = [traci.edge.getLastStepMeanSpeed(edge) for edge in traci.edge.getIDList()]

        return num_vehicles, avg_speed, queue_lengths

    def get_state_array(self):
        num_vehicles ,avg_speed, queue_lengths = self.get_state()
        q = queue_lengths[:20]
        if len(q) < 4:
            q += [0] * (4 - len(q))
        return np.array(q, dtype=np.float32)

    def calculate_reward(self):
        num_vehicles, avg_speed, queue_lengths = self.get_state()
        # print(f"numvehicle: {num_vehicles}")

        avg_speed = [-1 if speed == 0 else 0 for speed in avg_speed]

        queue_imbalance_penalty = max(queue_lengths) - min(queue_lengths)
        # print(f"avgspeed: {avg_speed}")

        reward = -sum(queue_lengths) + 0.05*sum(a * b for a, b in zip(num_vehicles, avg_speed)) - 1 * queue_imbalance_penalty
        return reward

    def close(self):
        traci.close()

    def set_traffic_light(self, light_id, phase):
        traci.trafficlight.setPhase(light_id, phase)
