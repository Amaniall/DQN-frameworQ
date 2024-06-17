import os
import sys
import traci
import sumolib

# Set SUMO_HOME to your SUMO installation path
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

def run():
    """Execute the TraCI control loop"""
    step = 0
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        step += 1
        print(f"Simulation step: {step}")

    traci.close()

if __name__ == "__main__":
    # Path to your SUMO configuration file
    sumo_cfg = "C:/Users/ELITEBOOK/romain_code\DQN-frameworQ-3\env\custom_env\data\platoon\platoon.sumocfg"

    # SUMO binary path
    sumoBinary = sumolib.checkBinary('sumo')

    # Start TraCI
    traci.start([sumoBinary, "-c", sumo_cfg])
    run()
