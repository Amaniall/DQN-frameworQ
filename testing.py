SUMO_HOME = "/usr/share/sumo/"
import sys
sys.path.append(SUMO_HOME + 'tools')

try:
    sys.path.append("/usr/share/sumo/tools")
    from sumolib import checkBinary
except ImportError:
    sys.exit("please declare environment variable 'SUMO_HOME' as the root directory of your sumo installation (it should contain folders 'bin', 'tools' and 'docs')")


from env import HYPER_PARAMS, network_config, CustomEnv
from dqn import CustomEnvWrapper, make_env, Agents

import os
import time
import argparse
import itertools
from datetime import timedelta

if __name__ == "__main__":
    env = CustomEnv()  # Initialize with 9 vehicles
    obs = env.reset()
    print("Initial Observation:", obs)
    
    # Add the 10th vehicle
    new_vehicle = {'position': 10, 'speed': 80, 'destination': 900}
    env._add_vehicle(new_vehicle)
    
    # Print the updated observations
    obs = env._get_obs()
    print("Updated Observation with 10th vehicle:", obs)
    
    # Apply actions to all vehicles
    action = [1] * 10  # Example action: all vehicles accelerate
    obs, reward, done, _ = env.step(action)