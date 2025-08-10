############################################################################################################################
##### BASIC IDEA ###########################################################################################################
############################################################################################################################

# User chooses input file
# Number of bodies, total time and time step are handed over automatically with the file path
# User chooses integrator 
# Simulation starts using the provided data and integrator
# Conserved quantities are calculated after every step (+ specific angular momentum, runge-lenz vector and semi-major axis)
# Conserved quantities and current body data are stored in a list
# GUI opens and shows current step/time
# Simulation ends
# History of conserved quantities and body values is written to .txt file with dynamic file name
# Plot shows final positions of bodies
# User is asked if there is any need for additional plots
# Plot functions for task 2 or close program
# This is repeated until the program is closed -> option to create all plots at once


############################################################################################################################
##### LIBRARIES ############################################################################################################
############################################################################################################################

# Import all neccesary libraries
# numpy -----------------> Numerical calculations, arrays and matrices
# pandas ----------------> Data processing, reading in data
# matplotlib.pyplot -----> Plot creation
# tkinter ---------------> Implementing graphical user interfaces (GUI)
# threading -------------> Enables threads (separate execution unit) -> parallelize tasks
# time ------------------> Needed for any time dependant operation
# copy ------------------> Used to create and manipulate shallow/deep copies of objects
# mpl_toolkits.mplot3d --> 3D Plots

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import messagebox, ttk
import threading
import time
import copy
from mpl_toolkits.mplot3d import Axes3D

############################################################################################################################
##### BODY CLASS ###########################################################################################################
############################################################################################################################

# Defines a class Body
# Every body has position [x,y,z], velocity [vx,vy,vz] and mass m

class Body:
    def __init__(self, position, velocity, mass):
        self.position = np.array(position)
        self.velocity = np.array(velocity)
        self.mass = mass

############################################################################################################################
##### INTEGRATOR CLASS #####################################################################################################
############################################################################################################################

class Integrator:
    def __init__(self, dt: float):
        self.dt = dt

    def integrate(self, bodies: list, accelerations: list):
        raise NotImplementedError("This should be overwritten!")

############################################################################################################################
##### IMPLEMENTATION OF THE DIFFERENT INTEGRATORS ##########################################################################
############################################################################################################################

# Using classes instead of functions simplifies the code and allows simple extension
# The Formula "x_{n+1} = x_n + v_{n+1} * dt" can be expressed as "body.position += body.veloctiy * self.dt" in python

class EulerIntegrator(Integrator): # Adjusted implementation:  Position is calculated with "old velocity", the position is updated afterwards
    def integrate(self, bodies, accelerations, calculate_accelerations):
        for i, body in enumerate(bodies):
            body.position += body.velocity * self.dt
            body.velocity += accelerations[i] * self.dt

class EulerCromerIntegrator(Integrator): # Adjusted implementation: Update velocity with acceleration first and then caluclate updated position
    def integrate(self, bodies, accelerations, calculate_accelerations):
        for i, body in enumerate(bodies):
            body.velocity += accelerations[i] * self.dt
            body.position += body.velocity * self.dt

class VelocityVerletIntegrator(Integrator): # Adjusted implementation: Now uses the averaged acceleration to update the velocity
    def integrate(self, bodies, accelerations, calculate_accelerations):
        # Caculate position with current velocity and acceleration
        for i, body in enumerate(bodies):
            body.position += body.velocity * self.dt + 0.5 * accelerations[i] * self.dt**2
        
        # Calculate updated acceleration (a_{n+1}) 
        accelerations_new = calculate_accelerations(bodies) 

        # Calculate new velocity using the averaged acceleration
        for i, body in enumerate(bodies):
            body.velocity += 0.5 * (accelerations[i] + accelerations_new[i]) * self.dt

class HermiteIntegrator(Integrator):
    def integrate(self, bodies, accelerations, calculate_accelerations):
        for i , body in enumerate(bodies):
            body.position += body.velocity * self.dt + 0.5 * accelerations[i] * self.dt**2
            body.velocity += accelerations[i] * self.dt

class IteratedHermiteIntegrator(Integrator):
    def integrate(self, bodies, accelerations, calculate_accelerations):
        for i, body in enumerate(bodies):
            body.position += body.velocity * self.dt + 0.5 * accelerations[i]  * self.dt**2
            body.velocity += 0.5 * (accelerations[i] + accelerations[i]) * self.dt

class HeunIntegrator(Integrator):
    def integrate(self, bodies, accelerations, calculate_accelerations):
        for i, body in enumerate(bodies):
            body.position += body.velocity * self.dt + 0.5 * accelerations[i] * self.dt**2
            body.velocity += accelerations[i] * self.dt

        new_accelerations = calculate_accelerations(bodies)

        for i, body in enumerate(bodies):
            body.velocity += 0.5 * (new_accelerations[i] + accelerations[i]) * self.dt
            body.position += 0.5 * (body.velocity * self.dt + 0.5 * new_accelerations[i] * self.dt**2)

import numpy as np

class RK4Integrator(Integrator): # Updated implementation to fit the equations in the script
    def integrate(self, bodies, accelerations, calculate_accelerations):
        # Iterate over all bodies in the list
        for i, body in enumerate(bodies):
            
            # Initial values (K1)
            x0 = body.position # Get the initial position of the body (x0)
            v0 = body.velocity # Get the initial velocity of the body (v0)
            a0 = accelerations[i] # Get the initial acceleration (a0)

            # K2: Compute intermediate values using the initial conditions
            # x1 = x0 + 0.5 * v0 * dt (Position at half time step)
            # v1 = v0 + 0.5 * a0 * dt (Velocity at half time step)
            x1 = x0 + 0.5 * v0 * self.dt
            v1 = v0 + 0.5 * a0 * self.dt
            body.position = x1 # Update the position of the body to calculate acceleration at this step
            a1 = calculate_accelerations(bodies)[i]  # Calculate new acceleration (a1) at the intermediate step

            # K3: Compute intermediate values using the updated values from K2
            # x2 = x0 + 0.5 * v1 * dt (Position at another half time step)
            # v2 = v0 + 0.5 * a1 * dt (Velocity at new time step)
            x2 = x0 + 0.5 * v1 * self.dt
            v2 = v0 + 0.5 * a1 * self.dt
            body.position = x2 # Update the position of the body to calculate acceleration at this step
            a2 = calculate_accelerations(bodies)[i]  # Calculate new acceleration (a2) at this intermediate step

            # K4: Compute final intermediate values
            # x3 = x0 + v2 * dt (Position at full time step)
            # v3 = v0 + a2 * dt (Velocity at full time step)
            x3 = x0 + v2 * self.dt
            v3 = v0 + a2 * self.dt
            body.position = x3  # Update the position of the body to calculate acceleration at this final step
            a3 = calculate_accelerations(bodies)[i]  # Calculate new acceleration (a3) at the final step

            # Final update step (RK4 formula)
            # Position update: Combine all intermediate velocities using weighted averages according to the equations from the script.
            body.position += (self.dt / 6) * (v0 + 2 * (v1 + v2) + v3)
            # Velocity update: Combine all accelerations using weighted averages according to the equations from the script.
            body.velocity += (self.dt / 6) * (a0 + 2 * (a1 + a2) + a3)


############################################################################################################################
##### SIMULATION PARAMETERS ################################################################################################
############################################################################################################################

# Defines a global variable "simulation parameters"
# Instead of hard coding a return value for every choice and deviation, this allows to use a simple "if" statement

simulation_parameters = {
    "a": {"file": "C://Users//svens//OneDrive//Desktop//Python//2_Computational_Physics//N_Body_Simulation//2body.txt", "num_bodies": 2, "total_time": 2.0, "dt": 0.001},
    "b": {"file": "C://Users//svens//OneDrive//Desktop//Python//2_Computational_Physics//N_Body_Simulation//3body.txt", "num_bodies": 3, "total_time": 220.0, "dt": 0.001},
    "c": {"file": "C://Users//svens//OneDrive//Desktop//Python//2_Computational_Physics//N_Body_Simulation//100body.txt", "num_bodies": 100, "total_time": 2.0, "dt": 0.01},
    "d": {"file": "C://Users//svens//OneDrive//Desktop//Python//2_Computational_Physics//N_Body_Simulation//1kbody.txt", "num_bodies": 1000, "total_time": 2.0, "dt": 0.01}
}

# Defines function that lets the user choose the input file

def choose_number_of_bodies():
    while True:
        print("Please choose the number of bodies you want to simulate.")
        print("Type the corresponding letter!")
        print("A. 2")
        print("B. 3")
        print("C. 100")
        print("D. 1000")
        print("E. Exit the program")

        choice = input("Please enter your choice: ").strip().lower()

        if choice in simulation_parameters:
            return simulation_parameters[choice]
        elif choice == "e":
            print("Program closed")
            exit()
        else:
            print("Invalid choice, please try again!")

# Defines a function that lets the user choose the desired integrator
# With ".strip().lower()" leadings/endings are removed and all characters are set to lower case -> no need to account for all that manually -> saves MANY lines of code

def choose_integrator():
    while True:
        print("Please choose the integrator:")
        print("A. Euler")
        print("B. Euler-Cromer")
        print("C. Velocity Verlet")
        print("D. Hermite")
        print("E. Iterated Hermite")
        print("F. Heun Integrator")
        print("G. RK4 Integrator")
        print("H. Exit the program")

        choice = input("Please enter your choice: ").strip().lower()

        if choice == "a":
            return EulerIntegrator
        elif choice == "b":
            return EulerCromerIntegrator
        elif choice == "c":
            return VelocityVerletIntegrator
        elif choice == "d":
                return HermiteIntegrator
        elif choice == "e":
            return IteratedHermiteIntegrator
        elif choice == "f":
            return HeunIntegrator
        elif choice == "g":
            return RK4Integrator
        elif choice == "h":
            print("Program closed")
            exit()
        else:
            print("Invalid choice, please try again!")

############################################################################################################################
##### READ INPUT FILE ######################################################################################################
############################################################################################################################

# Takes in previously chosen file path and reads the corresponding file
# The option "delim_whitespace" accepts spaces as delimeter of the file
# The "names" option allows to assign names to coloumns

def parse_data(file_path):
    try:
        data = pd.read_csv(file_path, delim_whitespace=True, header=None, names=["x", "y", "z", "vx", "vy", "vz", "m"], engine="python")
        # Creates objects in the Body class and stores data from the input file
        bodies = [Body([row["x"], row["y"], row["z"]], [row["vx"], row["vy"], row["vz"]], row["m"]) for _, row in data.iterrows()]
        # Calculates the total mass of all bodies in the file
        total_mass = sum(body.mass for body in bodies)
        

        # Calculates the standardized mass for every body in the file
        for body in bodies:
            body.mass /= total_mass  

        return bodies
    # Shows error if file is unavailable or cannot be accessed  and returns empty list   
    except FileNotFoundError:
        print("The file you chose is not available!")
        return []

############################################################################################################################
##### CALCULATE ACCELERATION ###############################################################################################
############################################################################################################################

# Adjusted to fit RK4 calculation, new positions and velocities are only used if needed
def calculate_accelerations(bodies, new_positions=None, new_velocities=None, G=1.0):
    # Initialize acceleration array
    accelerations = []
    
    for i, body in enumerate(bodies):
        net_force = np.zeros(3)
        
        # Use new positions and velocities if available
        if new_positions is not None and new_velocities is not None:
            if i >= len(new_positions) or i >= len(new_velocities):
                raise IndexError(f"Index {i} is out of bounds for new_positions or new_velocities")
            current_position = new_positions[i]
            current_velocity = new_velocities[i]
        else:
            current_position = body.position
            current_velocity = body.velocity
        
        for j, other_body in enumerate(bodies):
            if i != j:
                r_vector = other_body.position - current_position
                distance = np.linalg.norm(r_vector)
                if distance > 0:
                    force_magnitude = G * (body.mass * other_body.mass) / distance**2
                    force_direction = r_vector / distance
                    net_force += force_magnitude * force_direction
        
        if body.mass != 0: 
            acceleration = net_force / body.mass
        else:
            acceleration = np.zeros(3) 
        
        accelerations.append(acceleration)
    
    return accelerations
############################################################################################################################
##### CALCULATE CONSERVED QUANTITIES #######################################################################################
############################################################################################################################

def calculate_energy(bodies):
    total_energy = 0.0
    for body in bodies:
        kinetic_energy = 0.5 * body.mass * np.dot(body.velocity, body.velocity)
        total_energy += kinetic_energy
    for i, body in enumerate(bodies):
        for j, other_body in enumerate(bodies):
            if i < j:
                r_vector = other_body.position - body.position
                distance = np.linalg.norm(r_vector)
                if distance > 0:
                    potential_energy = - (body.mass * other_body.mass) / distance
                    total_energy += potential_energy
    return total_energy

def calculate_momentum(bodies):
    total_momentum = np.zeros(3)
    for body in bodies:
        total_momentum += body.mass * body.velocity
    return total_momentum

def calculate_angular_momentum(bodies):
    total_angular_momentum = np.zeros(3)
    for body in bodies:
        angular_momentum = np.cross(body.position, body.mass * body.velocity)
        total_angular_momentum += angular_momentum
    return total_angular_momentum

############################################################################################################################
##### CALCULATE SPECIFIC ANGULAR MOMENTUM, RUNGE-LENZ VECTOR AND SEMI-MAJOR AXIS ###########################################
############################################################################################################################

# These functions and the corresponding plot functions cause major problems!
# Try different approaches regarding logarithms, also avoid any division by 0
# Added tests to calculate_XYZ, plot_XYZ and also in the main simulation loop

def calculate_specific_angular_momentum(bodies):
    # Start with empty list
    specific_angular_momentum = []
    # Iterate over all bodies in the file
    for body in bodies:
        # Formula to calculate specific angular momentum
        if body.position is not None and body.velocity is not None: 
            j = np.cross(body.position, body.velocity)
            # Add result to list
            specific_angular_momentum.append(j)
        else:
            # Add placeholder should data be missing
            specific_angular_momentum.append(np.zeros(3))
    return specific_angular_momentum

def calculate_runge_lenz_vector(bodies, total_mass, G=1):
    runge_lenz_vector = []
    for body in bodies:
        r = body.position
        v = body.velocity
        r_magnitude = np.linalg.norm(r) if np.linalg.norm(r) > 0 else np.nan # Added to avoid division by 0
        angular_momentum = np.cross(r, body.mass * v)

        # Calculate Runge-Lenz vector
        # BUT only if there is no division by 0 in r_magnitude and/or total_mass
        if r_magnitude != 0 and total_mass != 0: # Added to avoid division by 0
            e = (np.cross(v, angular_momentum) / (G * total_mass)) - (r / r_magnitude)
            runge_lenz_vector.append(e)
        else:
            e = np.full(3, np.nan) # Added placeholder should calculation be invalid
    return runge_lenz_vector

def calculate_semi_major_axis(specific_angular_momentum, runge_lenz_vector, total_mass, G=1):
    semi_major_axis = []
    for j, e in zip(specific_angular_momentum, runge_lenz_vector):
        e_magnitude = np.linalg.norm(e) if e is not None else 0
        if e_magnitude < 1 and total_mass !=0:
            a = (np.dot(j, j)) / (G * total_mass * (1 - e_magnitude**2))
            semi_major_axis.append(a)
        else:
            semi_major_axis.append(np.nan) # Added placeholder should calculation be invalid

    return semi_major_axis

############################################################################################################################
##### SPECIFIC PLOTS #######################################################################################################
############################################################################################################################

# Autoscaling axes should not be the reason there is no change in relative energy displayed?

# ---> Replaced the plotting functions with a general function that takes the data for the relative values calculated in the simulation and plots them

# def plot_energy(relative_energy):
#     simulation_steps = np.arange(len(relative_energy)) # Fixed error, time_steps is now simulation_steps and now correctly denotes the number of steps in the simulation
#     plt.figure()
#     plt.plot(simulation_steps, [safe_log10(e) if e > 0 else np.nan for e in relative_energy], label="Relative Change in Energy", color="blue") # Added test
#     plt.xlabel("Time")
#     plt.ylabel("log |(E - E_start) / E_start|")
#     plt.title("Change in Energy over time")
#     plt.grid()
#     plt.legend()
#     plt.show()

# def plot_runge_lenz_vector(relative_runge_lenz_vector):
#     simulation_steps = np.arange(len(relative_runge_lenz_vector)) # Fixed error, time_steps is now simulation_steps and now correctly denotes the number of steps in the simulation
#     plt.figure()
#     plt.plot(simulation_steps, [safe_log10(e) if e > 0 else np.nan for e in relative_runge_lenz_vector], label="Relative Change in Runge-Lenz vector", color="green") # Added test
#     plt.xlabel("Time")
#     plt.ylabel("log ||(e - e_start)| / |e_start|")
#     plt.title("Change in Runge-Lenz vector over time")
#     plt.grid()
#     plt.legend()
#     plt.show()

# def plot_semi_major_axis(relative_semi_major_axis):
#     simulation_steps = np.arange(len(relative_semi_major_axis)) # Fixed error, time_steps is now simulation_steps and now correctly denotes the number of steps in the simulation
#     plt.figure()
#     plt.plot(simulation_steps, [safe_log10(a) if a > 0 else np.nan for a in relative_semi_major_axis], label="Relative Change in Semi-major Axis", color="green") # Added test
#     plt.xlabel("Time")
#     plt.ylabel("log |(a_e - a_e_start) / a_e_start|")
#     plt.title("Change in Semi-major Axis over time")
#     plt.grid()
#     plt.legend()
#     plt.show()

# Introduce general function that takes data, title, xlabel and y label as arguments
# ---> These arguments are assigned when a plot at the end of the simulation is chosen

# ---> New problem: There is only 1 value for energy calculated every step but N (number of bodies in the system) for runge-lenz vector and semi-major axis
# How do i plot every body seperatly in one single plot?

def plot_relative_values(data, title, xlabel, ylabel):
  
    # Add check if data is accessible
    if not data:
        print("No data to plot.")
        return
    
    # Create time array (basically x-axis)
    time = range(len(data))

    plt.figure(figsize=(10, 6))
    
    plt.plot(time, data)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.show()

############################################################################################################################
##### SAFE ENVIRONMENT FOR LOGARITHMS ######################################################################################
############################################################################################################################

# Try to create a function that ensures safe handling any scenario where there is a division by 0
# Seems to work just fine now -> hooray!
# I honestly don't know what exactly went wrong before and i don't know what is different now with that function instead of making the same check manually?!

def safe_log10(value, threshold=1e-10):
    if value > threshold:
        return np.log10(value)
    else:
        return threshold # Testing if that results in any significant change to the plots

############################################################################################################################
##### CHECK FOR NAN VALUES #################################################################################################
############################################################################################################################

# NaN values might be a problem when saving the data later -> check if there are any and if so notify user

def check_nan_values(relative_energy, relative_runge_lenz, relative_semi_major_axis):
    if np.any(np.isnan(relative_energy)):
        print("Warning: NaN values found in relative energy.")
    if np.any(np.isnan(relative_runge_lenz)):
        print("Warning: NaN values found in Runge-Lenz vector.")
    if np.any(np.isnan(relative_semi_major_axis)):
        print("Warning: NaN values found in Semi-major axis.")

############################################################################################################################
##### SAVE DATA TO FILE ####################################################################################################
############################################################################################################################

def save_data_to_file(filename, position_history, energy_history, momentum_history, angular_momentum_history, 
                       specific_angular_momentum_history, runge_lenz_vector_history, semi_major_axis_history):
    with open(filename, 'w') as f:
        # Added header line for clarity
        print(150 * "-")
        print(f"Saving data to {filename}...")
        f.write("Step\tPositions (x, y, z)\tEnergy\tMomentum (px, py, pz)\tAngular Momentum (Lx, Ly, Lz)\t"
                "Specific Angular Momentum (j_x, j_y, j_z)\tRunge-Lenz Vector (e_x, e_y, e_z)\tSemi-Major Axis\n")
        # Add print statement here to see where the process fails
        print("Header written successfully!")

        # Loop through all steps and save data
        for step in range(len(position_history)):
            positions = position_history[step]
            energy = energy_history[step]
            momentum = momentum_history[step]
            angular_momentum = angular_momentum_history[step]
            specific_angular_momentum = specific_angular_momentum_history[step]
            runge_lenz_vector = runge_lenz_vector_history[step]
            semi_major_axis = semi_major_axis_history[step]

            # Convert position data to string
            position_str = ', '.join([f"({body[0]:.6f}, {body[1]:.6f}, {body[2]:.6f})" for body in positions])

            # Write data to file, changed output structure
            f.write(f"{step}\t{position_str}\t{energy:.6f}\t"
                    f"{momentum[0]:.6f}, {momentum[1]:.6f}, {momentum[2]:.6f}\t"
                    f"{angular_momentum[0]:.6f}, {angular_momentum[1]:.6f}, {angular_momentum[2]:.6f}\t"
                    f"{specific_angular_momentum[0][0]:.6f}, {specific_angular_momentum[0][1]:.6f}, {specific_angular_momentum[0][2]:.6f}\t"
                    f"{runge_lenz_vector[0][0]:.6f}, {runge_lenz_vector[0][1]:.6f}, {runge_lenz_vector[0][2]:.6f}\t"
                    f"{float(semi_major_axis[0]):.6f}\n")

        # Add print statement for debugging
        print("Data saved successfully!")
        print(150 * "-")
    
############################################################################################################################
##### SIMULATION ###########################################################################################################
############################################################################################################################
start_time = time.time()
def run_simulation(bodies, dt, total_time, integrator_class):
    # Set gravitational constant to 1 according to to task
    G = 1.0 
    integrator = integrator_class(dt)

    # Initialize mass and positions
    total_mass = sum(body.mass for body in bodies)
    center_of_mass = sum(body.position * body.mass for body in bodies) / total_mass
    total_velocity = sum(body.velocity * body.mass for body in bodies) / total_mass

    for body in bodies:
        body.position -= center_of_mass
        body.velocity -= total_velocity

    num_steps = int(total_time / dt)
    energy_history = []
    momentum_history = []
    angular_momentum_history = []
    position_history = []
    specific_angular_momentum_history = []
    runge_lenz_vector_history = []
    semi_major_axis_history = []

    # Initialize the relative values
    relative_energy = []
    relative_runge_lenz = []
    relative_semi_major_axis = []

    # Store initial values for specific plots
    initial_energy = calculate_energy(bodies)
    initial_runge_lenz_vector = calculate_runge_lenz_vector(bodies, total_mass, G)
    initial_semi_major_axis = calculate_semi_major_axis(calculate_specific_angular_momentum(bodies), initial_runge_lenz_vector, total_mass, G)

    # Print initial values for debugging (if necessary)
    # ---> Currently not used for better clarity in terminal
    # print(150 * "-")
    # print(f"Initial Energy: {initial_energy}")
    # print(f"Initial Runge-Lenz Vector: {initial_runge_lenz_vector}")
    # print(f"Initial Semi-major Axis: {initial_semi_major_axis}")

    ###############
    ##### GUI #####
    ###############

    # Tkinter-setup for GUI
    root = tk.Tk()
    root.title("Simulation in Progress")
    
    progress_label = tk.Label(root, text="Current Step: 0")
    progress_label.pack(pady=5)
    
    timer_label = tk.Label(root, text="Time: 0.00 s")
    timer_label.pack(pady=5)

    progress = ttk.Progressbar(root, orient="horizontal", length=300, mode="determinate")
    progress.pack(pady=20)

    start_time = time.time()  # Track the starting time

    def update_progress(step):
        elapsed_time = time.time() - start_time
        progress['value'] = (step + 1) / num_steps * 100
        progress_label.config(text=f"Current Step: {step + 1}")
        timer_label.config(text=f"Time: {elapsed_time:.2f} s")
        root.update_idletasks()
    
    def run():
        # Initialize the runtime values for the calculation of the relative values of energy, runge-lenz vector and semi-major axis
        relative_energy_time = 0
        relative_rlv_time = 0
        relative_sma_time = 0
        total_elapsed_time = 0

        for step in range(num_steps):
            # Calculate accelerations and integrate forward
            accelerations = calculate_accelerations(bodies)
            integrator.integrate(bodies, accelerations, calculate_accelerations)

            # Debugging output: Check the positions and velocities after integration -> RK4 gives vastly different orbits
            # if step < 50:  # Only output the first 50 steps
            #     print(f"Step {step}:")
            #     for i, body in enumerate(bodies):
            #         print(f"  Body {i} - Position: {body.position}, Velocity: {body.velocity}")

            # Calculates total energy, total momentum and total angular momentum
            total_energy = calculate_energy(bodies)
            total_momentum = calculate_momentum(bodies)
            total_angular_momentum = calculate_angular_momentum(bodies)

            # Calculates the specific angular momentum, runge-lenz vector and semi-major axis
            specific_angular_momentum = calculate_specific_angular_momentum(bodies)
            runge_lenz_vector = calculate_runge_lenz_vector(bodies, total_mass, G)
            semi_major_axis = calculate_semi_major_axis(specific_angular_momentum, runge_lenz_vector, total_mass, G)

            # Add data to history
            # ---> Adjusted order, now data for current step is saved before calculating changes
            position_history.append([np.copy(body.position) for body in bodies])
            energy_history.append(total_energy)
            momentum_history.append(total_momentum)
            angular_momentum_history.append(total_angular_momentum)
            specific_angular_momentum_history.append(specific_angular_momentum)
            runge_lenz_vector_history.append(runge_lenz_vector)
            semi_major_axis_history.append(semi_major_axis)

            # Update relative energy, runge-lenz vector, and semi-major axis
            # ---> Changes are only calculated if an initial value is set
            start_time_energy = time.time()
            if initial_energy != 0:
                relative_energy.append(safe_log10(np.abs((total_energy - initial_energy) / initial_energy)))
            else:
                relative_energy.append(np.nan)
            relative_energy_time += time.time() - start_time_energy

            start_time_rlv = time.time()
            if initial_runge_lenz_vector:
                for e, e_start in zip(runge_lenz_vector, initial_runge_lenz_vector):
                    if np.linalg.norm(e_start) != 0:
                        relative_runge_lenz.append(safe_log10(np.abs(np.linalg.norm(e) - np.linalg.norm(e_start)) / np.linalg.norm(e_start)))
                    else:
                        relative_runge_lenz.append(np.nan)
            relative_rlv_time += time.time() - start_time_rlv

            start_time_sma = time.time()
            if initial_semi_major_axis:
                for a, a_start in zip(semi_major_axis, initial_semi_major_axis):
                    if a_start != 0:
                        relative_semi_major_axis.append(safe_log10(np.abs(a - a_start) / a_start))
                    else:
                        relative_semi_major_axis.append(np.nan)
            relative_sma_time += time.time() - start_time_sma

            # Check for NaN values
            check_nan_values(relative_energy, relative_runge_lenz, relative_semi_major_axis)

            # Stop simulation if NaN occurs
            if np.any(np.isnan(relative_energy)) or np.any(np.isnan(relative_runge_lenz)) or np.any(np.isnan(relative_semi_major_axis)):
                print("Simulation stopped due to NaN values.")
                break

            # Update progress
            update_progress(step)
            time.sleep(0.01)

        # Print seperate calculation times
        # Remember to use f-strings so that variables can be formatted correctly!
        print(125 * "-")
        print("Task 3:")
        print(f"{relative_energy_time:.2f} seconds (relative Energy)")
        print(f"{relative_rlv_time:.2f} seconds (Runge-Lenz vector)")
        print(f"{relative_sma_time:.2f} seconds (Semi-major axis)")
        end_time = time.time()
        total_elapsed_time = end_time - start_time
        print(f"{total_elapsed_time:.2f} seconds (Total elapsed time)")
        # After the simulation loop ends
        root.destroy()
    # Start simulation in a separate thread to keep the GUI responsive
    threading.Thread(target=run).start()
    root.mainloop()

    # Save data to file
    num_bodies = len(bodies)
    integrator_name = integrator_class.__name__.replace('Integrator', '')
    filename = f'C://Users//svens//OneDrive//Desktop//Python//2_Computational_Physics//N_Body_Simulation//simulation_results_{num_bodies}_bodies_{integrator_name}.txt'
    
    save_data_to_file(filename, position_history, energy_history, momentum_history, angular_momentum_history, specific_angular_momentum_history, runge_lenz_vector_history, semi_major_axis_history)

    # Visualize body positions after simulation
    # ---> modified so that now the orbit is shown instead of the final positions of the bodies
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the orbits of all bodies
    # "zip(*position_history)" is an argument unpacking operator 
    # ---> In this case hands over every element of the list position_history (every row) as separate list [zip()]
    for body_positions in zip(*position_history): 
        body_positions = np.array(body_positions)
        ax.plot(body_positions[:, 0], body_positions[:, 1], body_positions[:, 2])

    ax.set_title('Orbits of Bodies')
    ax.set_xlabel('x-axis')
    ax.set_ylabel('y-axis')
    ax.set_zlabel('z-axis')
    ax.grid()
    plt.show()

    # Handle user input for plotting
    while True:
        print("Do you want to plot the relative changes over time in...")
        print("A. Energy")
        print("B. Runge-Lenz Vector")
        print("C. Semi-major Axis")
        print("D. No plot")
        
        user_input = input("Enter A, B, C, or D: ").strip().upper()
        
        if user_input == "A":
            plot_relative_values(relative_energy, 'Change in Energy over Time', 'Time [s]', 'log |(E - E_start) / E_start|')
        elif user_input == "B":
            # Plot all calculated Runge-Lenz vector values  in one plot
            # -> use history
            all_runge_lenz = [np.linalg.norm(runge_lenz) for runge_lenz in runge_lenz_vector_history]
            plot_relative_values(all_runge_lenz,
                                'Change in Runge-Lenz Vector (all bodies)',
                                'Time [s]',
                                'log |(e - e_start) / e_start|')
        elif user_input == "C":
            # Plot all semi-major axis values in one plot
            # -> use history
            all_semi_major_axes = [semi_major_axis for semi_major_axis in semi_major_axis_history]
            plot_relative_values(all_semi_major_axes,
                                'Change in Semi-major Axis (all bodies)',
                                'Time [s]',
                                'log |(a_e - a_e_start) / a_e_start|')
        elif user_input == "D":
            break
        else:
            print("Invalid input. Please try again.")

############################################################################################################################
##### MAIN FUNCTION ########################################################################################################
############################################################################################################################

def main():
    params = choose_number_of_bodies()
    bodies = parse_data(params["file"])
    if bodies:
        print(f"Loaded {len(bodies)} bodies.")

        dt = params["dt"]
        total_time = params["total_time"]
        integrator_class = choose_integrator()
        run_simulation(bodies, dt, total_time, integrator_class)
    else:
        print("No body data in this list, please check the file and try again.")

if __name__ == "__main__":
    main()