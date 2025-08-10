############################################################################################################################
##### BASIC IDEA ###########################################################################################################
############################################################################################################################

# This program solves the 1D Euler equations for a shock tube problem
# Euler equations are a system of conservation laws describing:
# - Conservation of mass (continuity equation)
# - Conservation of momentum
# - Conservation of energy

############################################################################################################################
##### LIBRARIES ############################################################################################################
############################################################################################################################

# Import all necessary libraries
# numpy -----------------> Numerical calculations, arrays and matrices
# pandas ----------------> Data processing, reading in data
# matplotlib.pyplot -----> Plot creation
# os --------------------> Path handling

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

############################################################################################################################
##### EULER SOLVER #########################################################################################################
############################################################################################################################

# Write code that solves the 1D Euler equations using (2.4) - (2.6)

def solve_euler(N, γ):

    ##########################
    ##### Advection Step #####
    ##########################

    ###################
    ##### Density #####
    ###################

    def ρ_advection(Δt, Δx, ρ, u, mass_flux, Δρ, ρ_adv, N):
        # Compute Δρ using (2.16)
        # Added boundary handling to prevent index out of bounds
        for i in range(1, N-1):
            cond = (ρ[i+1] - ρ[i]) * (ρ[i] - ρ[i-1])
            if cond > 0:
                Δρ[i] = 2 * cond / (ρ[i+1] - ρ[i-1])
            else:
                Δρ[i] = 0

        # (2.15) - Compute ρ_adv
        # Corrected indexing and boundary handling
        for i in range(1, N-1):
            if u[i] > 0:
                ρ_adv[i] = ρ[max(0, i-1)] + 0.5 * (1 - u[i] * Δt / Δx) * Δρ[max(0, i-1)]
            else:
                ρ_adv[i] = ρ[i] + 0.5 * (1 + u[i] * Δt / Δx) * Δρ[i]

        # (2.14) - Compute mass flux
        # Corrected indexing 
        for i in range(1, N-1):
            mass_flux[i] = ρ_adv[i] * u[i]

        # (2.13) - Update ρ
        # Corrected indexing and boundary handling
        for i in range(1, N-1):
            ρ[i] = ρ[i] - Δt / Δx * (mass_flux[i+1] - mass_flux[i])

    #############################
    ##### Momentum/Velocity #####
    #############################

    def u_advection(Δt, Δx, ρ, ρ0, u, ε, mass_flux, momentum_flux, Δu, u_adv, N):
        # (2.22) - Compute Δu
        # Corrected indexing 
        for i in range(1, N):
            cond = (u[min(i+1, N-1)] - u[i]) * (u[i] - u[max(0, i-1)])
            if cond > 0:
                Δu[i] = 2 * cond / (u[min(i+1, N-1)] - u[max(0, i-1)])
            else:
                Δu[i] = 0

        # (2.20) - Compute u_adv
        # Corrected indexing and boundary handling
        for i in range(1, N):
            u_bar = 0.5 * (u[i] + u[min(i+1, N-1)])
            if u_bar > 0:
                u_adv[i] = u[i] + 0.5 * (1 - u_bar * Δt / Δx) * Δu[i]
            else:
                u_adv[i] = u[min(i+1, N-1)] - 0.5 * (1 + u_bar * Δt / Δx) * Δu[min(i+1, N-1)]

        # (2.19) - Compute momentum flux
        # Corrected indexing 
        for i in range(1, N):
            momentum_flux[i] = 0.5 * (mass_flux[i] + mass_flux[min(i+1, N-1)]) * u_adv[i]

        # (2.23) - Update u
        # Corrected indexing and boundary handling
        for i in range(1, N):
            ρ_bar0 = 0.5 * (ρ0[max(0, i-1)] + ρ0[i])
            ρ_bar  = 0.5 * (ρ[max(0, i-1)] + ρ[i])
            u[i] = 1 / ρ_bar * (u[i] * ρ_bar0 - Δt / Δx * (momentum_flux[i] - momentum_flux[max(0, i-1)]))

    ##################
    ##### Energy #####
    ##################

    def ε_advection(Δt, Δx, ρ, ρ0, u, ε, mass_flux, energy_flux, Δε, ε_adv, N):
        # (2.29) - Compute Δε
        # Corrected indexing 
        for i in range(1, N-1):
            cond = (ε[i+1] - ε[i]) * (ε[i] - ε[i-1])
            if cond > 0:
                Δε[i] = 2 * cond / (ε[i+1] - ε[i-1])
            else:
                Δε[i] = 0

        # (2.28) - Compute ε_adv
        # Corrected indexing and boundary handling
        for i in range(1, N-1):
            if u[i] > 0:
                ε_adv[i] = ε[max(0, i-1)] + 0.5 * (1 - u[i] * Δt / Δx) * Δε[max(0, i-1)]
            else:
                ε_adv[i] = ε[i] - 0.5 * (1 + u[i] * Δt / Δx) * Δε[i]

        # (2.27) - Compute energy flux
        # Corrected indexing 
        for i in range(1, N-1):
            energy_flux[i] = mass_flux[i] * ε_adv[i]

        # (2.30) - Update ε
        # Corrected indexing and boundary handling
        for i in range(1, N-1):
            ε[i] = 1 / ρ[i] * (ε[i] * ρ0[i] - Δt / Δx * (energy_flux[i+1] - energy_flux[i]))

    def calculate_pressure(p, ρ, ε, N, γ):
        # (2.36) - Compute pressure
        # Corrected indexing 
        for i in range(1, N-1):
            p[i] = (γ - 1) * ρ[i] * ε[i]

    def calculate_temperature(temperature, ε, N, γ):
        # Compute temperature
        # Corrected indexing 
        for i in range(1, N-1):
            temperature[i] = ε[i] * (γ - 1)

    def update_u(Δt, Δx, ρ, u, p, N):
        # (2.34) - Update u
        # Corrected indexing and boundary handling
        for i in range(1, N):
            ρ_bar = 0.5 * (ρ[max(0, i-1)] + ρ[i])
            u[i] = u[i] - Δt / Δx * (p[i] - p[max(0, i-1)]) / ρ_bar

    def update_ε(Δt, Δx, ρ, u_old, ε, p, N):
        # (2.38) - Update ε
        # Corrected indexing and boundary handling
        for i in range(1, N-1):
            ε[i] = ε[i] - Δt / Δx * p[i] / ρ[i] * (u_old[min(i+1, N-1)] - u_old[i])

    #############################
    ##### Start Calculation #####
    #############################

    # Parameters from the problem statement
    x = np.linspace(0, 1, N)
    Δx = x[1] - x[0]
    x_0 = 0.5
    Δt = 0.001
    t_current = 0.0
    t_end = 0.228

    # Initialize all arrays
    ρ = np.zeros(N)
    ρ0 = np.zeros(N)
    ε = np.zeros(N)
    pressure = np.zeros(N)
    temperature = np.zeros(N)
    Δε = np.zeros(N)
    ε_adv = np.zeros(N)
    Δρ = np.zeros(N)
    ρ_adv = np.zeros(N)
    # Changed from N+1 to N
    u = np.zeros(N)  
    u_old = np.zeros(N)  
    mass_flux = np.zeros(N)  
    energy_flux = np.zeros(N)  
    momentum_flux = np.zeros(N)  
    Δu = np.zeros(N)  
    u_adv = np.zeros(N) 

    # Generate initial conditions
    # Simplified initial condition setup
    ρ = np.where(x <= x_0, 1.0, 0.125)
    ε = np.where(x <= x_0, 2.5, 2.0)
    pressure = np.where(x <= x_0, 1.0, 0.1)

    # Added error handling for reference solution file
    try:
        ref_file_path = os.path.join(os.path.dirname(__file__), "Musterlösung.txt")
        data = pd.read_csv(ref_file_path, delimiter=';')
    except FileNotFoundError:
        print("Reference solution file not found. Skipping comparison plots.")
        data = None

    # Start the loop
    while t_current < t_end:
        ρ0 = np.copy(ρ)
        u_old = np.copy(u)

        # Calculate pressure
        calculate_pressure(pressure, ρ, ε, N, γ)

        # Perform advection steps
        ρ_advection(Δt, Δx, ρ, u, mass_flux, Δρ, ρ_adv, N)
        u_advection(Δt, Δx, ρ, ρ0, u, ε, mass_flux, momentum_flux, Δu, u_adv, N)
        ε_advection(Δt, Δx, ρ, ρ0, u, ε, mass_flux, energy_flux, Δε, ε_adv, N)

        # Calculate pressure again
        calculate_pressure(pressure, ρ, ε, N, γ)

        # Update velocity and energy
        update_u(Δt, Δx, ρ, u, pressure, N)
        update_ε(Δt, Δx, ρ, u_old, ε, pressure, N)

        # Calculate temperature
        calculate_temperature(temperature, ε, N, γ)

        # Increment the time
        t_current += Δt

    # Use extracted path if data exists
    if data is not None:
        x_muster = data.iloc[::10, 0].values
        v_muster = data.iloc[::10, 1].values
        ρ_muster = data.iloc[::10, 2].values
        T_muster = data.iloc[::10, 3].values
        P_muster = data.iloc[::10, 4].values

        plot_path = os.path.dirname(__file__)
        plot_and_save(x, u, ρ, temperature, pressure, x_muster, v_muster, ρ_muster, T_muster, P_muster, plot_path)

def plot_and_save(x, u, ρ, temperature, pressure, x_muster, v_muster, ρ_muster, T_muster, P_muster, plot_path):
    # Define a helper function to create plots
    def create_plot(x_sim, y_sim, x_ref, y_ref, xlabel, ylabel, filename):
        plt.figure(figsize=(10, 6))
        plt.plot(x_sim, y_sim, label="Simulation", color='orange', linewidth=1)
        plt.scatter(x_sim, y_sim, color='orange', marker='x', label="Simulation", s=30)
        plt.plot(x_ref, y_ref, label="Reference", color='blue', linewidth=1)
        plt.scatter(x_ref, y_ref, color='black', marker='x', label="Reference", s=10)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_path, filename))
        print(f"Plot saved as {filename}")
        plt.close()

    # Plot for velocity (u)
    create_plot(x, u, x_muster, v_muster, "x", "u", "A2_u_228.png")

    # Plot for density (ρ)
    create_plot(x, ρ, x_muster, ρ_muster, "x", r"$\rho$", "A2_rho_228.png")

    # Plot for temperature (T)
    create_plot(x, temperature, x_muster, T_muster, "x", "T", "A2_epsilon_228.png")

    # Plot for pressure (P)
    create_plot(x, pressure, x_muster, P_muster, "x", "P", "A2_pressure_228.png")

def main():
    N = 100
    γ = 1.4  # Adiabatic exponent according to task
    solve_euler(N, γ)

if __name__ == "__main__":
    main()