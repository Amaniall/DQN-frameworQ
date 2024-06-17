# """CHANGE CUSTOM ENV IMPORT HERE""" ##################################################################################
from .custom_env import RES
from gym import spaces
import numpy as np
########################################################################################################################


from gym import spaces
import numpy as np


from gym import spaces
import numpy as np

class DqnEnv:
    def __init__(self):
        self.n_vehicles = 9  # Initial number of vehicles
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_vehicles + 1, 4), dtype=np.float32)
        self.action_space = spaces.Discrete(4)  # 0: Maintain, 1: Accelerate, 2: Decelerate, 3: Lane Change

        self.vehicles = self._initialize_vehicles()

    def _initialize_vehicles(self):
        vehicles = [
            {'position': 0, 'speed': 0, 'destination': 400 },
            {'position': 1, 'speed': 0, 'destination': 410 },
            {'position': 2, 'speed': 0, 'destination': 430 },
            {'position': 3, 'speed': 0, 'destination': 450 },
            {'position': 4, 'speed': 0, 'destination': 460 },
            {'position': 5, 'speed': 0, 'destination': 550 },
            {'position': 6, 'speed': 0, 'destination': 600 },
            {'position': 7, 'speed': 0, 'destination': 650 },
            {'position': 8, 'speed': 0, 'destination': 700 },
        ]
        return vehicles

    def _get_obs(self):
        observations = []
        for vehicle in self.vehicles:
            position = vehicle['position']
            speed = vehicle['speed']
            destination = vehicle['destination']
            #is_tail = self._is_tail(vehicle)
            observations.append([position, speed, destination])
        return np.array(observations)

    def _apply_action(self, vehicle, action):
        if action == 1:  # Accelerate
            vehicle['speed'] += 1
        elif action == 2:  # Decelerate
            vehicle['speed'] -= 1
        elif action == 3:  # Lane Change
            self._lane_change(vehicle)

    def _lane_change(self, vehicle):
        pass  # Implement lane change logic

    def _simulate_step(self):
        pass  # Update environment state

    def _calculate_reward(self):
        reward = 0
        for vehicle in self.vehicles:
            if self._is_tail(vehicle):
                reward += self._tail_reward(vehicle)
            else:
                reward += self._head_reward(vehicle)
        return reward

    def _check_done(self):
        return False  # Determine if episode is done

    def _is_tail(self, vehicle):
        return vehicle['position'] == max(v['position'] for v in self.vehicles)

    def _tail_reward(self, vehicle):
        return 1  # Reward for being at tail

    def _head_reward(self, vehicle):
        return 0.5  # Reward for being at head

    def _add_vehicle(self, vehicle):
        min_destination = min(v['destination'] for v in self.vehicles)
        max_destination = max(v['destination'] for v in self.vehicles)

        if vehicle['destination'] <= max_destination:
            self._join_head(vehicle)
        elif vehicle['destination'] >= min_destination:
            self._join_tail(vehicle)
        else:
            print("vehicle can't join, waiting for another platoon")

    def _join_head(self, vehicle):
        vehicle['position'] = min(v['position'] for v in self.vehicles) - 1
        #vehicle['is_tail'] = False
        vehicle['speed'] = self._determine_speed_head(vehicle)
        self.vehicles.insert(0, vehicle)  # Add to the front

    def _join_tail(self, vehicle):
        vehicle['position'] = max(v['position'] for v in self.vehicles) + 1
        vehicle['speed'] = self._determine_speed_tail(vehicle)
        self.vehicles.append(vehicle)  # Add to the back

    def _join_middle(self, vehicle):
        # Insert the vehicle in the middle based on its destination
        index = next(i for i, v in enumerate(self.vehicles) if v['destination'] < vehicle['destination'])
        vehicle['position'] = self.vehicles[index - 1]['position'] + 1
        vehicle['speed'] = self._determine_speed_middle(vehicle)
        self.vehicles.insert(index, vehicle)

    def _determine_speed_head(self, vehicle):
        # Accelerate when joining at the head
        return vehicle['speed'] + 2

    def _determine_speed_tail(self, vehicle):
        # Decelerate when joining at the tail
        return vehicle['speed'] - 2

    def _determine_speed_middle(self, vehicle):
        # Maintain speed when joining in the middle
        return vehicle['speed']

    def step(self, action):
        for i, vehicle in enumerate(self.vehicles):
            self._apply_action(vehicle, action[i])
        
        # Example of adding a new vehicle based on some condition
        new_vehicle = {'position': 0, 'speed': 0, 'destination': np.random.randint(50, 500) }
        self._add_vehicle(new_vehicle)
        
        self._simulate_step()
        obs = self._get_obs()
        reward = self._calculate_reward()
        done = self._check_done()
        
        return obs, reward, done, {}

    def reset(self):
        self.vehicles = self._initialize_vehicles()
        return self._get_obs()




"""class DqnEnv:
    def __init__(self):
        self.n_vehicles = 9  # Initial number of vehicles
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_vehicles + 1, 4), dtype=np.float32)
        self.action_space = spaces.Discrete(4)  # 0: Maintain, 1: Accelerate, 2: Decelerate, 3: Lane Change

        self.vehicles = self._initialize_vehicles()

    def _initialize_vehicles(self):
        vehicles = [
            {'position': 0, 'speed': 0, 'destination': 480, 'is_tail': False},
            {'position': 1, 'speed': 0, 'destination': 450, 'is_tail': False},
            {'position': 2, 'speed': 0, 'destination': 430, 'is_tail': False},
            {'position': 3, 'speed': 0, 'destination': 410, 'is_tail': False},
            {'position': 4, 'speed': 0, 'destination': 400, 'is_tail': False},
            {'position': 5, 'speed': 0, 'destination': 350, 'is_tail': False},
            {'position': 6, 'speed': 0, 'destination': 300, 'is_tail': False},
            {'position': 7, 'speed': 0, 'destination': 250, 'is_tail': False},
            {'position': 8, 'speed': 0, 'destination': 200, 'is_tail': True},
        ]
        return vehicles

    def _get_obs(self):
        observations = []
        for vehicle in self.vehicles:
            position = vehicle['position']
            speed = vehicle['speed']
            destination = vehicle['destination']
            is_tail = self._is_tail(vehicle)
            observations.append([position, speed, destination, is_tail])
        return np.array(observations)

    def _apply_action(self, vehicle, action):
        if action == 1:  # Accelerate
            vehicle['speed'] += 1
        elif action == 2:  # Decelerate
            vehicle['speed'] -= 1
        elif action == 3:  # Lane Change
            self._lane_change(vehicle)

    def _lane_change(self, vehicle):
        pass  # Implement lane change logic

    def _simulate_step(self):
        pass  # Update environment state

    def _calculate_reward(self):
        reward = 0
        for vehicle in self.vehicles:
            if self._is_tail(vehicle):
                reward += self._tail_reward(vehicle)
            else:
                reward += self._head_reward(vehicle)
        return reward

    def _check_done(self):
        return False  # Determine if episode is done

    def _is_tail(self, vehicle):
        return vehicle['position'] == max(v['position'] for v in self.vehicles)

    def _tail_reward(self, vehicle):
        return 1  # Reward for being at tail

    def _head_reward(self, vehicle):
        return 0.5  # Reward for being at head

    def _add_vehicle(self, vehicle):
        head_destination = self.vehicles[0]['destination']
        tail_destination = self.vehicles[-1]['destination']

        if vehicle['destination'] > head_destination:
            self._join_head(vehicle)
        else:
            self._join_tail(vehicle)

    def _join_head(self, vehicle):
        vehicle['position'] = min(v['destination'] for v in self.vehicles) - 1
        vehicle['is_tail'] = False
        vehicle['speed'] = self._determine_speed_head(vehicle)
        self.vehicles.insert(0, vehicle)  # Add to the front

    def _join_tail(self, vehicle):
        vehicle['position'] = max(v['destination'] for v in self.vehicles) + 1
        vehicle['is_tail'] = True
        vehicle['speed'] = self._determine_speed_tail(vehicle)
        self.vehicles.append(vehicle)  # Add to the back

    def _determine_speed_head(self, vehicle):
        # Accelerate when joining at the head
        return vehicle['speed'] + 2

    def _determine_speed_tail(self, vehicle):
        # Decelerate when joining at the tail
        return vehicle['speed'] - 2

    def step(self, action):
        for i, vehicle in enumerate(self.vehicles):
            self._apply_action(vehicle, action[i])
        
        # Example of adding a new vehicle based on some condition
        if np.random.rand() < 0.1:  # 10% chance to add a new vehicle
            new_vehicle = {'position': 0, 'speed': 0, 'destination': np.random.randint(50, 500), 'is_tail': False}
            self._add_vehicle(new_vehicle)
        
        self._simulate_step()
        obs = self._get_obs()
        reward = self._calculate_reward()
        done = self._check_done()
        
        return obs, reward, done, {}

    def reset(self):
        self.vehicles = self._initialize_vehicles()
        return self._get_obs()
"""

"""class DqnEnv:
    def __init__(self):
        self.n_vehicles = 10  # Initial number of vehicles
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_vehicles, 4), dtype=np.float32)
        self.action_space = spaces.Discrete(4)  # 0: Maintain, 1: Accelerate, 2: Decelerate, 3: Lane Change

        self.vehicles = self._initialize_vehicles()

    def _initialize_vehicles(self):
        vehicles = []
        for _ in range(self.n_vehicles):
            vehicle = {'position': 0, 'speed': 0, 'destination': np.random.randint(50, 500), 'is_tail': False}
            vehicles.append(vehicle)
        # Sort vehicles by destination in descending order
        vehicles.sort(key=lambda v: v['destination'], reverse=True)
        return vehicles

    def _get_obs(self):
        observations = []
        for vehicle in self.vehicles:
            position = vehicle['position']
            speed = vehicle['speed']
            destination = vehicle['destination']
            is_tail = self._is_tail(vehicle)
            observations.append([position, speed, destination, is_tail])
        return np.array(observations)

    def _apply_action(self, vehicle, action):
        if action == 1:  # Accelerate
            vehicle['speed'] += 1
        elif action == 2:  # Decelerate
            vehicle['speed'] -= 1
        elif action == 3:  # Lane Change
            self._lane_change(vehicle)

    def _lane_change(self, vehicle):
        pass  # Implement lane change logic

    def _simulate_step(self):
        pass  # Update environment state

    def _calculate_reward(self):
        reward = 0
        for vehicle in self.vehicles:
            if self._is_tail(vehicle):
                reward += self._tail_reward(vehicle)
            else:
                reward += self._head_reward(vehicle)
        return reward

    def _check_done(self):
        return False  # Determine if episode is done

    def _is_furthest(self, vehicle):
        return vehicle['destination'] == max(v['destination'] for v in self.vehicles)

    def _is_closest(self, vehicle):
        return vehicle['destination'] == min(v['destination'] for v in self.vehicles)

    def _is_safe_to_join_head(self, vehicle):
        return True  # Logic to check if safe to join head

    def _is_safe_to_join_tail(self, vehicle):
        return True  # Logic to check if safe to join tail

    def _is_tail(self, vehicle):
        return vehicle['position'] == max(v['position'] for v in self.vehicles)

    def _tail_reward(self, vehicle):
        return 1  # Reward for being at tail

    def _head_reward(self, vehicle):
        return 0.5  # Reward for being at head

    def _add_vehicle(self, destination):
        vehicle = {'position': 0, 'speed': 0, 'destination': destination, 'is_tail': False}
        # Insert vehicle into the sorted list
        index = next((i for i, v in enumerate(self.vehicles) if v['destination'] < vehicle['destination']), len(self.vehicles))
        self.vehicles.insert(index, vehicle)
        # Update positions and speed based on insertion point
        if index == 0:
            self._join_head(vehicle)
        else:
            self._join_tail(vehicle)

    def _join_head(self, vehicle):
        if self._is_safe_to_join_head(vehicle):
            vehicle['position'] = min(v['position'] for v in self.vehicles) - 1
            vehicle['is_tail'] = False
            vehicle['speed'] = self._determine_speed_head(vehicle)

    def _join_tail(self, vehicle):
        if self._is_safe_to_join_tail(vehicle):
            vehicle['position'] = max(v['position'] for v in self.vehicles) + 1
            vehicle['is_tail'] = True
            vehicle['speed'] = self._determine_speed_tail(vehicle)

    def _determine_speed_head(self, vehicle):
        # Determine speed for a vehicle joining at the head
        return vehicle['speed'] + 2  # Example logic to accelerate

    def _determine_speed_tail(self, vehicle):
        # Determine speed for a vehicle joining at the tail
        return vehicle['speed'] - 2  # Example logic to decelerate

    def step(self, action):
        for i, vehicle in enumerate(self.vehicles):
            self._apply_action(vehicle, action[i])
        
        # Example of adding a new vehicle based on some condition
        if np.random.rand() < 0.1:  # 10% chance to add a new vehicle
            new_destination = np.random.randint(50, 500)
            self._add_vehicle(new_destination)
        
        self._simulate_step()
        obs = self._get_obs()
        reward = self._calculate_reward()
        done = self._check_done()
        
        return obs, reward, done, {}

    def reset(self):
        self.vehicles = self._initialize_vehicles()
        return self._get_obs()
"""

"""class DqnEnv:

    def __init__(self):
        self.n_vehicles = 10  # Initial number of vehicles
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_vehicles, 4), dtype=np.float32)
        self.action_space = spaces.Discrete(4)  # 0: Maintain, 1: Accelerate, 2: Decelerate, 3: Lane Change

        self.vehicles = self._initialize_vehicles()

    def _initialize_vehicles(self):
        vehicles = []
        for _ in range(self.n_vehicles):
            vehicle = {'position': 0, 'speed': 0, 'destination': np.random.randint(50, 500), 'is_tail': False}
            vehicles.append(vehicle)
        return vehicles

    def _get_obs(self):
        observations = []
        for vehicle in self.vehicles:
            position = vehicle['position']
            speed = vehicle['speed']
            destination = vehicle['destination']
            is_tail = self._is_tail(vehicle)
            observations.append([position, speed, destination, is_tail])
        return np.array(observations)

    def _apply_action(self, vehicle, action):
        if action == 1:  # Accelerate
            vehicle['speed'] += 1
        elif action == 2:  # Decelerate
            vehicle['speed'] -= 1
        elif action == 3:  # Lane Change
            self._lane_change(vehicle)

    def _lane_change(self, vehicle):
        pass  # Implement lane change logic

    def _simulate_step(self):
        pass  # Update environment state

    def _calculate_reward(self):
        reward = 0
        for vehicle in self.vehicles:
            if self._is_tail(vehicle):
                reward += self._tail_reward(vehicle)
            else:
                reward += self._head_reward(vehicle)
        return reward

    def _check_done(self):
        return False  # Determine if episode is done

    def _is_furthest(self, vehicle):
        return vehicle['destination'] == max(v['destination'] for v in self.vehicles)

    def _is_closest(self, vehicle):
        return vehicle['destination'] == min(v['destination'] for v in self.vehicles)

    def _is_safe_to_join_head(self, vehicle):
        return True  # Logic to check if safe to join head

    def _is_safe_to_join_tail(self, vehicle):
        return True  # Logic to check if safe to join tail

    def _is_tail(self, vehicle):
        return vehicle['position'] == max(v['position'] for v in self.vehicles)

    def _tail_reward(self, vehicle):
        return 1  # Reward for being at tail

    def _head_reward(self, vehicle):
        return 0.5  # Reward for being at head

    def _add_vehicle(self, destination):
        vehicle = {'position': 0, 'speed': 0, 'destination': destination, 'is_tail': False}
        if self._is_furthest(vehicle):
            self._join_head(vehicle)
        else:
            self._join_tail(vehicle)
        self.vehicles.append(vehicle)

    def _join_head(self, vehicle):
        if self._is_safe_to_join_head(vehicle):
            vehicle['position'] = min(v['position'] for v in self.vehicles) - 1
            vehicle['is_tail'] = False
            vehicle['speed'] = self._determine_speed_head(vehicle)

    def _join_tail(self, vehicle):
        if self._is_safe_to_join_tail(vehicle):
            vehicle['position'] = max(v['position'] for v in self.vehicles) + 1
            vehicle['is_tail'] = True
            vehicle['speed'] = self._determine_speed_tail(vehicle)

    def _determine_speed_head(self, vehicle):
        # Determine speed for a vehicle joining at the head
        return vehicle['speed'] + 2  # Example logic to accelerate

    def _determine_speed_tail(self, vehicle):
        # Determine speed for a vehicle joining at the tail
        return vehicle['speed'] - 2  # Example logic to decelerate

    def step(self, action):
        for i, vehicle in enumerate(self.vehicles):
            self._apply_action(vehicle, action[i])
        
        # Example of adding a new vehicle based on some condition
        if np.random.rand() < 0.1:  # 10% chance to add a new vehicle
            new_destination = np.random.randint(50, 500)
            self._add_vehicle(new_destination)
        
        self._simulate_step()
        obs = self._get_obs()
        reward = self._calculate_reward()
        done = self._check_done()
        
        return obs, reward, done, {}

    def reset(self):
        self.vehicles = self._initialize_vehicles()
        return self._get_obs()
"""

"""class DqnEnv:
    def __init__(self):
        self.n_vehicles = 10  # Define a constant for the number of vehicles
        self.mode = {"train": False, "observe": False, "play": False}
        self.player = p if self.mode["play"] else None
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_vehicles, 4), dtype=np.float32)
        self.action_space = spaces.Discrete(4)  # Example: 0: Maintain, 1: Accelerate, 2: Decelerate, 3: Lane Change
        
        self.min_max = {
            "obs": (0., 1.),
            "rew": (0., 1.)
        }
        ################################################################################################################

        
        # CHANGE ACTION AND OBSERVATION SPACE SIZES HERE #########################################################
        
        self.action_space_n = 1
        self.observation_space_n = 1
        ###############################
        self.vehicles = self._initialize_vehicles()
        

    def _initialize_vehicles(self):
        vehicles = []
        for _ in range(self.n_vehicles):
            vehicle = {'position': 0, 'speed': 0, 'destination': np.random.randint(50, 500), 'is_tail': False}
            vehicles.append(vehicle)
        return vehicles

    def obs(self):
        observations = []
        for vehicle in self.vehicles:
            position = vehicle['position']
            speed = vehicle['speed']
            destination = vehicle['destination']
            is_tail = self._is_tail(vehicle)
            observations.append([position, speed, destination, is_tail])
        return np.array(observations)

    def _apply_action(self, vehicle, action):
        if action == 1:  # Accelerate
            vehicle['speed'] += 1
        elif action == 2:  # Decelerate
            vehicle['speed'] -= 1
        elif action == 3:  # Lane Change
            self._lane_change(vehicle)

    def _lane_change(self, vehicle):
        pass  # Implement lane change logic

    def _simulate_step(self):
        pass  # Update environment state

    def _calculate_reward(self):
        reward = 0
        for vehicle in self.vehicles:
            if self._is_tail(vehicle):
                reward += self._tail_reward(vehicle)
            else:
                reward += self._head_reward(vehicle)
        return reward

    def _check_done(self):
        return False  # Determine if episode is done

    def _is_furthest(self, vehicle):
        return vehicle['destination'] == max(v['destination'] for v in self.vehicles)

    def _is_closest(self, vehicle):
        return vehicle['destination'] == min(v['destination'] for v in self.vehicles)

    def _is_safe_to_join_head(self, vehicle):
        return True  # Logic to check if safe to join head

    def _is_safe_to_join_tail(self, vehicle):
        return True  # Logic to check if safe to join tail

    def _is_tail(self, vehicle):
        return vehicle['position'] == max(v['position'] for v in self.vehicles)

    def _tail_reward(self, vehicle):
        return 1  # Reward for being at tail

    def _head_reward(self, vehicle):
        return 0.5  # Reward for being at head

    
    def step(self, action):
    # Ensure action is a list
     if isinstance(action, int):
        action = [action] * len(self.vehicles)
    
     for i, vehicle in enumerate(self.vehicles):
        self._apply_action(vehicle, action[i])
    
     for vehicle in self.vehicles:
        if self._is_furthest(vehicle):
            self._join_head(vehicle)
        else:
            self._join_tail(vehicle)
    
     self._simulate_step()
     obs = self.obs()
     reward = self._calculate_reward()
     done = self._check_done()
    
     return obs, reward, done, {}

    
    
    
    def step(self, action):
        for i, vehicle in enumerate(self.vehicles):
            self._apply_action(vehicle, action[i])
        
        for vehicle in self.vehicles:
            if self._is_furthest(vehicle):
                self._join_head(vehicle)
            else:
                self._join_tail(vehicle)
        
        self._simulate_step()
        obs = self._get_obs()
        reward = self._calculate_reward()
        done = self._check_done()
        
        return obs, reward, done, {}

    def _join_head(self, vehicle):
        if self._is_safe_to_join_head(vehicle):
            vehicle['position'] = min(v['position'] for v in self.vehicles) - 1
            vehicle['is_tail'] = False

    def _join_tail(self, vehicle):
        if self._is_safe_to_join_tail(vehicle):
            vehicle['position'] = max(v['position'] for v in self.vehicles) + 1
            vehicle['is_tail'] = True

    def reset(self):
        self.vehicles = self._initialize_vehicles()
        return self.obs()
"""


"""class DqnEnv:

    def min_max_scale(self, x, feature):
        return (x - self.min_max[feature][0]) / (self.min_max[feature][1] - self.min_max[feature][0])

    def __init__(self):
        
        self.n_vehicles = 10
        self.vehicles= []
        self.mode = {"train": False, "observe": False, "play": False, "mode": True}
        self.player = p if self.mode["play"] else None
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_vehicles, 5), dtype=np.float32)
        self.action_space = spaces.Discrete(4)  # 0: Maintain, 1: Accelerate, 2: Decelerate, 3: Lane Change

        #### CHANGE ENV CONSTRUCT HERE##############################################################################
        ################################################################################################################

        # CHANGE FEATURE SCALING HERe ############################################################################

        self.min_max = {
            "obs": (0., 1.),
            "rew": (0., 1.)
        }
        ################################################################################################################

        
        # CHANGE ACTION AND OBSERVATION SPACE SIZES HERE #########################################################
        
        self.action_space_n = 1
        self.observation_space_n = 1
        ################################################################################################################


    def _initialize_vehicles(self):
        # Initialize vehicle states
        vehicles = []
        for _ in range(self.n_vehicles):
            vehicle = {'position': 0, 'speed': 0, 'destination': np.random.randint(50, 500), 'is_tail': False}
            vehicles.append(vehicle)
        return vehicles
    
    
    def obs(self):
        
        # CHANGE OBSERVATION HERE ################################################################################
        #obs = [self.min_max_scale(x, "obs") for x in [1.]]
        observations = []
        for vehicle in vehicles:
            position = self._get_position(vehicle)
            speed = self._get_speed(vehicle)
            destination = self._get_destination(vehicle)
            is_tail = self._is_tail(vehicle)
            observations.append([position, speed, destination, is_tail])
        return np.array(observations)
        
        ################################################################################################################
        return obs

    def rew(self):
        # CHANGE REWARD HERE#####################################################################################
         rew = 0
         for vehicle in self.vehicles:
          if self._is_tail(vehicle):
            rew += self._tail_reward(vehicle)
          else:
            rew += self._head_reward(vehicle)
         return rew
        
        
        #rew = self.min_max_scale(0, "rew")
        ################################################################################################################

    def done(self):
        # CHANGE DONE HERE #######################################################################################
        done = False
        ################################################################################################################
        return done

    def info(self):
        # CHANGE INFO HERE#######################################################################################
        info = {}
        ################################################################################################################
        return info

    def reset(self):
        # CHANGE RESET HERE ######################################################################################
        pass
        ################################################################################################################

     #def step(self, action):
        # CHANGE STEP HERE #######################################################################################
    def step(self, action):
    # Ensure action is a list
     if isinstance(action, int):
        action = [action] * len(self.vehicles)
    
     for i, vehicle in enumerate(self.vehicles):
        self._apply_action(vehicle, action[i])
    
     for vehicle in self.vehicles:
        if self._is_furthest(vehicle):
            self._join_head(vehicle)
        else:
            self._join_tail(vehicle)
    
     self._simulate_step()
     obs = self.obs()
     reward = self._calculate_reward()
     done = self._check_done()
    
     return obs, reward, done, {}
    
      

    def _is_furthest(self, vehicle):
     return vehicle['destination'] == max(v['destination'] for v in self.vehicles)

    def _is_closest(self, vehicle):
     return vehicle['destination'] == min(v['destination'] for v in self.vehicles)




    def _join_head(self, vehicle):
    # Logic to join the head of the platoon
     if self._is_safe_to_join_head(vehicle):
        vehicle['position'] = min(v['position'] for v in self.vehicles) - 1
        vehicle['is_tail'] = False

    def _join_tail(self, vehicle):
    # Logic to join the tail of the platoon
     if self._is_safe_to_join_tail(vehicle):
        vehicle['position'] = max(v['position'] for v in self.vehicles) + 1
        vehicle['is_tail'] = True
        
        
    
        ################################################################################################################

    def reset_render(self):
        # CHANGE RESET RENDER HERE ###############################################################################
        pass
        ################################################################################################################

    def step_render(self):
        # CHANGE STEP RENDER HERE ################################################################################
        pass
        ################################################################################################################
"""