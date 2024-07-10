import traci
import sumolib

def run_simulation():
    # Path to SUMO and the configuration file
    sumoBinary = sumolib.checkBinary('sumo')  # use 'sumo-gui' if you want to visualize
    sumoConfig = "gui-settings.cfg"

    # Start SUMO as a subprocess and connect to it with TraCI
    traci.start([sumoBinary, "-c", sumoConfig])
    
    # Run the simulation for a set number of steps
    step = 0
    while step < 100:
        traci.simulationStep()  # Advance the simulation by one step

        # Get the list of vehicle IDs currently in the simulation
        vehicle_ids = traci.vehicle.getIDList()
        print(f"Step {step}:")
        for vehicle_id in vehicle_ids:
            # Get the position (x, y) of the vehicle
            position = traci.vehicle.getPosition(vehicle_id)
            print(f"  Vehicle {vehicle_id} is at position {position}")

        step += 1

    # Close the TraCI connection
    traci.close()

if __name__ == "__main__":
    run_simulation()
