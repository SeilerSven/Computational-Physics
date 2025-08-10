############################################################################################################################
##### LATEST CHANGES #######################################################################################################
############################################################################################################################

# Installed the "Greek Alphabet Snippets" extension so i can finally type Greek letters!
# ---> Had to deactivate unicode highlighting due to annoying yellow boxes around the characters

############################################################################################################################
##### BASIC IDEA ###########################################################################################################
############################################################################################################################

# Write code that solves the linear advaction equation (2.1) using the given procedure (2.13-2.16)
# First define hydrodynamic system with parameters

############################################################################################################################
##### LIBRARIES ############################################################################################################
############################################################################################################################

# Import all necessary libraries
# numpy -----------------> Numerical calculations, arrays, and matrices
# matplotlib.pyplot -----> Plot creation
# tkinter ---------------> Implementing graphical user interfaces (GUI)
# time ------------------> Needed for any time-dependent operation

import numpy as np
import matplotlib.pyplot as plt
import time
import tkinter as tk
from tkinter import ttk
import os

############################################################################################################################
##### DEFINE HYDRODYNAMIC SYSTEM ###########################################################################################
############################################################################################################################

# Create class for hydrodynamic system with all parameters

class HydrodynamicSystem:
    def __init__(self, ρ_j, Δ_x, u_j, boundary_condition, ε_j, γ, p_j):
        self.ρ_j = np.array(ρ_j)                        # Density
        self.Δ_x = np.array(Δ_x)                        # Grid spacing
        self.u_j = np.array(u_j)                        # Velocity
        self.boundary_condition = boundary_condition    # Type of boundary condition
        self.ε_j = np.array(ε_j)                        # Energy
        self.γ = np.array(γ)                            # 
        self.p_j = np.array(p_j)                        # Pressure

# Function that creates the starting system

def create_starting_system(x_j, N, Δ_x, γ):
    # Initialize arrays
    ρ_j = np.zeros(N)
    ε_j = np.ones(N)
    p_j = np.ones(N)
    ### Changed velocity array size to match density array
    u_j = np.ones(N) * -1                               # For periodic boundary condition

    # Set density values with respect to position

    for i in range(N):
        if abs(x_j[i]) <= 1/3:                          # Set density according to task, either 1.0 if absolute <= 1/3 or 0.0 else
            ρ_j[i] = 1.0
        else:
            ρ_j[i] = 0.0

    # Return the initialized system
    return HydrodynamicSystem(ρ_j, Δ_x, u_j, 'periodic', ε_j, γ, p_j)

############################################################################################################################
##### SOLVE ADVECTION EQUATION #############################################################################################
############################################################################################################################

# Function that solves the linear advection equation using ghost cells

def solve_linear_advection(sys, σ, a, t_end):
    # Fixed to properly handle array sizes
    N = len(sys.ρ_j)                                   # Number of physical cells
    Δ_t = (σ * sys.Δ_x) / a                            # Time step calculated from courant number
    steps = int(t_end / Δ_t)                           # Number of time steps

    ρ_j = np.copy(sys.ρ_j)
    u_j = np.copy(sys.u_j)

    # Changed to create array with ghost cells outside the loop
    # Preallocate for time evolution
    ρ_j_evolved = np.copy(ρ_j)
    
    # Function to calculate Δρ (2.16)
    # Improved delta_ρ function to handle periodic boundaries properly
    def delta_ρ(ρ_j, j):
        j_plus = (j + 1) % N
        j_minus = (j - 1) % N
        
        numer = (ρ_j[j_plus] - ρ_j[j]) * (ρ_j[j] - ρ_j[j_minus])
        denom = ρ_j[j_plus] - ρ_j[j_minus]
        
        if numer > 0:
            if abs(denom) < 1e-10:  
                return 0
            else:
                return 2 * numer / denom
        else:
            return 0

    # Function to calculate ρ_adv (2.15)
    # Corrected ρ_adv function to handle periodic boundaries
    def ρ_adv(ρ_j, u_j, j, Δ_t, Δ_x):
        if u_j[j] > 0:
            j_upwind = (j - 1) % N
            return ρ_j[j_upwind] + (1 / 2) * (1 - u_j[j] * Δ_t / Δ_x) * delta_ρ(ρ_j, j_upwind)
        else:
            return ρ_j[j] - (1 / 2) * (1 + u_j[j] * Δ_t / Δ_x) * delta_ρ(ρ_j, j)

    # Function to calculate Fm (2.14)
    def Fm(ρ_j, u_j, j, Δ_t, Δ_x):
        return ρ_adv(ρ_j, u_j, j, Δ_t, Δ_x) * u_j[j]

    # Function to update ρ (2.13)
    # Corrected ρ_new function to handle periodic boundaries
    def ρ_new(ρ_j, u_j, j, Δ_t, Δ_x):
        j_plus = (j + 1) % N
        return ρ_j[j] - Δ_t / Δ_x * (Fm(ρ_j, u_j, j_plus, Δ_t, Δ_x) - Fm(ρ_j, u_j, j, Δ_t, Δ_x))

    # Main loop to evolve the system over time
    for step in range(steps):
        # Create a copy of the current density state for calculations
        ρ_j_current = np.copy(ρ_j_evolved)
        
        # Update density for each cell using formula 2.13
        for i in range(N):
            ρ_j_evolved[i] = ρ_new(ρ_j_current, u_j, i, Δ_t, sys.Δ_x)

    # Return the updated system
    return HydrodynamicSystem(ρ_j_evolved, sys.Δ_x, u_j, sys.boundary_condition, sys.ε_j, sys.γ, sys.p_j)

############################################################################################################################
##### PLOTS ################################################################################################################
############################################################################################################################

def plot_results(x_j, start_system, evolved_system, ρ_analytical, filename, N, t):
    plt.figure()
    
    # Plot analytical solution
    plt.plot(x_j, ρ_analytical, label="Analytische Lsg.", linewidth=2, alpha=0.4, color='darkblue')
    # Plot numerical solution as scatter plot
    plt.scatter(x_j, evolved_system.ρ_j, label="Numerische Lsg.", marker='x', color='orange', s=10)
    plt.plot(x_j, evolved_system.ρ_j, linewidth=1, alpha=0.2, color='black')
    
    plt.xlabel("x")
    plt.ylabel(r"$\rho$")
    plt.legend()
    plt.grid()
    plt.title(f"Advection for N={N} at t={t}", fontsize=14)
    
    filepath = "C:\\Users\\svens\\OneDrive\\Desktop\\Python\\2_Computational_Physics\\Numerical_Hydrodynamics"
    full_path = os.path.join(filepath, f"A1_N{N}_t{t}.png")
    
    # Save before plotting otherwise the png is empty!
    plt.savefig(full_path)
    
    plt.show()
    
    plt.close()

############################################################################################################################
##### MAIN #################################################################################################################
############################################################################################################################

def main():
    # Set parameters for the simulation
    N40 = 40  # Number of grid points for the first system
    N400 = 400  # Number of grid points for the second system
    σ = 0.8  # Courant number (used for stability)
    a = 1.0  # Speed in the advection equation
    Δ_x40 = 2 / (N40 - 1)  # Grid spacing for N40
    Δ_x400 = 2 / (N400 - 1)  # Grid spacing for N400
    t_start = 0.0  # Start time
    t_end40 = 4.0  # End time for the first system (N=40)
    t_end400 = 40.0  # End time for the second system (N=400)
    x_j40 = np.linspace(-1, 1, N40)  # x-coordinates for N40
    x_j400 = np.linspace(-1, 1, N400)  # x-coordinates for N400

    # Create initial system for both N40 and N400
    start_system40 = create_starting_system(x_j40, N40, Δ_x40, 1.4) 
    start_system400 = create_starting_system(x_j400, N400, Δ_x400, 1.4) 

    # Corrected analytical solution for advected profile after t=4
    ρ_analytical40_t0 = np.copy(start_system40.ρ_j)
    ρ_analytical40_t4 = np.zeros(N40)
    for i in range(N40):
        # Calculate shifted position based on advection velocity and time
        shifted_pos = x_j40[i] + a * t_end40
        # Apply periodic boundary conditions
        while shifted_pos > 1:
            shifted_pos -= 2
        while shifted_pos < -1:
            shifted_pos += 2
        # Set density based on shifted position
        if abs(shifted_pos) <= 1/3:
            ρ_analytical40_t4[i] = 1.0

    # Corrected analytical solution for advected profile after t=40
    ρ_analytical400_t0 = np.copy(start_system400.ρ_j)
    ρ_analytical400_t40 = np.zeros(N400)
    for i in range(N400):
        # Calculate shifted position based on advection velocity and time
        shifted_pos = x_j400[i] + a * t_end400
        # Apply periodic boundary conditions
        while shifted_pos > 1:
            shifted_pos -= 2
        while shifted_pos < -1:
            shifted_pos += 2
        # Set density based on shifted position
        if abs(shifted_pos) <= 1/3:
            ρ_analytical400_t40[i] = 1.0

    # Solve advection for both systems at different times
    # N=40, t=0 (Initial state)
    evolved_system40_t0 = solve_linear_advection(start_system40, σ, a, t_start)

    # N=40, t=4 (After 4 seconds)
    start_time = time.time()  # Start time measurement
    evolved_system40_t4 = solve_linear_advection(start_system40, σ, a, t_end40) 
    print(f"Time for N=40, t=4: {time.time() - start_time} seconds")  

    # N=400, t=0 (Initial state for N400)
    evolved_system400_t0 = solve_linear_advection(start_system400, σ, a, t_start)

    # N=400, t=40 (After 40 seconds)
    start_time = time.time()  # Start time measurement
    evolved_system400_t40 = solve_linear_advection(start_system400, σ, a, t_end400)
    print(f"Time for N=400, t=40: {time.time() - start_time} seconds") 

    # Plot results for all cases, including titles with N and t in the title
    plot_results(x_j40, start_system40, evolved_system40_t0, ρ_analytical40_t0, 
                 f"A1_N{N40}_t{t_start}.png", 
                 N40, t_start)  # Plot for N=40, t=0

    plot_results(x_j40, start_system40, evolved_system40_t4, ρ_analytical40_t4, 
                 f"A1_N{N40}_t{t_end40}.png", 
                 N40, t_end40)  # Plot for N=40, t=4

    plot_results(x_j400, start_system400, evolved_system400_t0, ρ_analytical400_t0, 
                 f"A1_N{N400}_t{t_start}.png", 
                 N400, t_start)  # Plot for N=400, t=0

    plot_results(x_j400, start_system400, evolved_system400_t40, ρ_analytical400_t40, 
                 f"A1_N{N400}_t{t_end400}.png", 
                 N400, t_end400)  # Plot for N=400, t=40

# Call the main function to run the program
if __name__ == "__main__":
    main()