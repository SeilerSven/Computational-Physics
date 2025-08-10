############################################################################################################################
##### BASIC IDEA ###########################################################################################################
############################################################################################################################

############################################################################################################################
##### LIBRARIES ############################################################################################################
############################################################################################################################

# Import all neccesary libraries
# numpy -----------------> Numerical calculations, arrays and matrices
# matplotlib.pyplot -----> Plot creation
# tqdm ------------------> Progress bars in loops
# numba -----------------> Just-in-time compilation for faster computations

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import numba as nb    # Added Numba for JIT compilation as the code was very slow

############################################################################################################################
##### TASK 1 ###############################################################################################################
############################################################################################################################

# Task 1a: Monte Carlo calculation of pi
def mc_pi_calculation(num_samples=100000):
    # Implement classic Monte Carlo method to estimate pi
    # Generate random points uniformly distributed in the unit square (0,0) to (1,1)
    # Count how many points are located in the quarter unit circle
    # Ratio of points inside the circle to total points approaches pi/4 with increasing number of samples
    # Multiply by 4 to get estimate of pi

    # Generate random points in the unit square
    x = np.random.random(num_samples)
    y = np.random.random(num_samples)
    
    # Count points inside the unit circle
    inside_circle = np.sum((x**2 + y**2) <= 1.0)
    
    # Estimate pi
    pi_estimate = 4.0 * inside_circle / num_samples
    
    return pi_estimate

# Task 1b: Random number generator for normal distribution and Monte Carlo integration
def normal_random_generator(mu, sigma, size=1):
    # Wrapper around numpy random normal distribution generator
    # Allows to set mean value (mu) and standard deviation (sigma)
    return mu + sigma * np.random.randn(size)

def mc_gaussian_integral(num_samples=100000):
    # Use Monte Carlo integration to estimate int(exp(-t^2/2)dt)

    # Define integration range
    t_min = -10 
    t_max = 10
    
    # Generate uniform random samples in the integration range
    t_samples = np.random.uniform(t_min, t_max, num_samples)
    
    # Calculate function values
    f_values = np.exp(-t_samples**2 / 2)
    
    # Monte Carlo integration: average × integration range
    integral = np.mean(f_values) * (t_max - t_min)
    
    return integral

############################################################################################################################
##### TASK 2 ###############################################################################################################
############################################################################################################################

# Task 2: 2D Ising model with direct enumeration
def exact_ising_2d(L, beta_values):
    # Set up calculation for 2D Ising model
    # Total number of spins (L^2)
    n_sites = L * L
    # Totalnumber of possible configurations (2^(L^2))
    n_configs = 2**n_sites
    
    # Initialize arrays to store results for energy, magnetization and absolute magnetization
    energies = np.zeros(len(beta_values))
    mags = np.zeros(len(beta_values))
    mag_abs = np.zeros(len(beta_values))
    
    # Generate all possible configurations
    configs = np.zeros((n_configs, n_sites), dtype=np.int8)
    
    # Generate all possible configurations using binary representation
    # Convert 0s to -1s and 1s to 1s (standard Ising model spin values)
    for i in range(n_configs):
        binary = format(i, f'0{n_sites}b')
        configs[i] = [1 if bit == '1' else -1 for bit in binary]
    
    # Reshape for 2D lattice
    configs_2d = configs.reshape(n_configs, L, L)
    
    # Calculate energy and magnetization for each configuration
    energies_config = np.zeros(n_configs)
    mags_config = np.zeros(n_configs)
    
    # For each configuration, calculate two key values
    for c in range(n_configs):
        lattice = configs_2d[c]
        
        # Calculate energy
        energy = 0
        for i in range(L):
            for j in range(L):
                # Periodic boundary conditions
                right_nn = lattice[i, (j+1) % L]
                down_nn = lattice[(i+1) % L, j]
                
                # J = 1 for ferromagnetic coupling
                energy -= lattice[i, j] * (right_nn + down_nn)
        
        energies_config[c] = energy
        mags_config[c] = np.sum(lattice)
        
    # Calculate thermal averages for each beta
    for b_idx, beta in enumerate(beta_values):
        # Boltzmann weights
        boltzmann_weights = np.exp(-beta * energies_config)
        partition_function = np.sum(boltzmann_weights)
        
        # Energy and magnetization averages
        # All values have to be normalozed by dividing by L^2 to get per-site quantities
        energies[b_idx] = np.sum(energies_config * boltzmann_weights) / partition_function / (L*L)
        mags[b_idx] = np.sum(mags_config * boltzmann_weights) / partition_function / (L*L)
        mag_abs[b_idx] = np.sum(np.abs(mags_config) * boltzmann_weights) / partition_function / (L*L)
    
    return energies, mags, mag_abs

############################################################################################################################
##### TASK 3 ###############################################################################################################
############################################################################################################################

# Task 3: Metropolis algorithm for the 2D Ising model
def initialize_lattice(L, hot_start=True):
    # Create initial state of Ising model lattice
    # Set hot_start = True as default -> create random configurations with spins ramdomly set to 1 or -1
    # This corresponds to a idsordered state at high temperature
    # For hot_start = False -> ordered state at low temperature with all spins set to 1
    if hot_start:
        return 2 * np.random.randint(2, size=(L, L)) - 1
    else:
        return np.ones((L, L))

# Numba-optimized energy calculation
@nb.njit
def calculate_energy(lattice, L):
    # Calculate total energy of Ising lattice configuration
    # Loop through every lattice site (i,j) and sum up the neighbouring spins
    energy = 0
    for i in range(L):
        for j in range(L):
            spin = lattice[i, j]
            # Periodic boundary conditions
            neighbors = lattice[(i+1)%L, j] + lattice[i, (j+1)%L] + lattice[(i-1)%L, j] + lattice[i, (j-1)%L]
            energy -= spin * neighbors
    
    # Each pair is counted twice, so divide by 2
    return energy / 2

# Numba-optimized magnetization calculation
@nb.njit
def calculate_magnetization(lattice):
    # Calculate total magnetization as sum of all individual spin values in the lattice
    return np.sum(lattice)

# Numba-optimized Metropolis step
@nb.njit
def metropolis_step(lattice, L, beta, h=0):
    # Define metropolis step
    # One sweep = L² attempted spin flips
    for _ in range(L*L):  
        # Choose a random site
        i, j = np.random.randint(0, L), np.random.randint(0, L)
        
        # Calculate energy change if this spin is flipped
        spin = lattice[i, j]
        neighbors = lattice[(i+1)%L, j] + lattice[i, (j+1)%L] + lattice[(i-1)%L, j] + lattice[i, (j-1)%L]
        
        # ΔE = 2*J*s_i*(sum of neighbors) + 2*h*s_i
        delta_E = 2 * spin * neighbors + 2 * h * spin
        
        # Accept or reject according to Metropolis criterion
        # If ΔE <= 0 (energy decreases) -> always accept flip
        # If ΔE >  0 (energy increases) -> accept with probability exp(-ß * ΔE)
        if delta_E <= 0 or np.random.random() < np.exp(-beta * delta_E):
            lattice[i, j] = -spin
    
    return lattice

# Numba-optimized multi-hit Metropolis
@nb.njit
def metropolis_multihit(lattice, L, beta, h=0, multihit=10):
    # Perform multiple metropolis steps in sequence to improve efficiency
    for _ in range(multihit):
        lattice = metropolis_step(lattice, L, beta, h)
    return lattice

# Main Metropolis simulation function ---> partially optimized
# Can't "jit" this function because it uses tqdm, which isn't compatible with Numba!
def run_ising_metropolis(L, beta_values, h=0, thermalization=1000, measurements=5000, multihit=1):
    # Set up Metropolis simulation of Ising model at various temperatures
    # Initialize arrays to store results
    energies = np.zeros(len(beta_values))
    energy_squared = np.zeros(len(beta_values))
    magnetizations = np.zeros(len(beta_values))
    mag_abs = np.zeros(len(beta_values))
    mag_squared = np.zeros(len(beta_values))
    
    # For each temperature (beta_values), create new random lattice configuration
    # Thermalize system by running mulit-hit metropolis function
    for b_idx, beta in enumerate(tqdm(beta_values, desc="Temperatures")):
        # Initialize lattice
        lattice = initialize_lattice(L, hot_start=True)
        
        # Thermalize
        # Don't calculate data as this phase represents non-equilibrium behaviour
        for _ in range(thermalization):
            lattice = metropolis_multihit(lattice, L, beta, h, multihit)
        
        # Measurements
        E_sum = 0
        E2_sum = 0
        M_sum = 0
        M_abs_sum = 0
        M2_sum = 0
        
        # After thermalization, begin measuring phase
        # Calculate energy and magnetization for each configuration
        for _ in range(measurements):
            lattice = metropolis_multihit(lattice, L, beta, h, multihit)
            
            # Calculate observables
            E = calculate_energy(lattice, L)
            M = calculate_magnetization(lattice)
            
            E_sum += E
            E2_sum += E**2
            M_sum += M
            M_abs_sum += abs(M)
            M2_sum += M**2
        
        # Calculate averages by dividing the sums by  the number of measurements
        energies[b_idx] = E_sum / measurements / (L*L)
        energy_squared[b_idx] = E2_sum / measurements / (L*L)**2
        magnetizations[b_idx] = M_sum / measurements / (L*L)
        mag_abs[b_idx] = M_abs_sum / measurements / (L*L)
        mag_squared[b_idx] = M2_sum / measurements / (L*L)**2
    
    # Calculate specific heat: C = β²(⟨E²⟩ - ⟨E⟩²)
    specific_heat = np.zeros(len(beta_values))
    for b_idx, beta in enumerate(beta_values):
        specific_heat[b_idx] = beta**2 * (energy_squared[b_idx] - energies[b_idx]**2 * (L*L))
    
    return energies, magnetizations, mag_abs, mag_squared, specific_heat


# # Added error calculation by blocking method to get hints on how to chose better parameters for task 3
# def blocking_error(data):
    
#     # If there are too few data points, return standard error
#     if len(data) < 10:
#         return np.std(data, ddof=1) / np.sqrt(len(data))
    
#     # Get the length of the array
#     n = len(data)
    
#     # Number of blocks to use - using sqrt(n) as a heuristic
#     n_blocks = max(10, int(np.sqrt(n)))
    
#     # Length of each block
#     block_size = n // n_blocks
    
#     # Create blocks by averaging consecutive measurements
#     blocks = np.array([np.mean(data[i*block_size:(i+1)*block_size]) for i in range(n_blocks)])
    
#     # Return standard error of the blocks
#     return np.std(blocks, ddof=1) / np.sqrt(n_blocks)

############################################################################################################################
##### TASK 4 ###############################################################################################################
############################################################################################################################

# Task 4a: Heat Bath algorithm for the 2D Ising model
@nb.njit
def heat_bath_step(lattice, L, beta, h=0):
    # Do one step of the heat bath algorithm on the lattice
    # Similar structure but different update rule for new spin states
    # One sweep = L² attempted spin flips
    for _ in range(L*L):  
        # Choose a random site
        i, j = np.random.randint(0, L, 2)
        
        # Calculate local field
        neighbors = lattice[(i+1)%L, j] + lattice[i, (j+1)%L] + lattice[(i-1)%L, j] + lattice[i, (j-1)%L]
        local_field = neighbors + h
        
        # Heat Bath probability for spin up
        p_up = 1 / (1 + np.exp(-2 * beta * local_field))
        
        # Set spin according to probability
        lattice[i, j] = 1 if np.random.random() < p_up else -1
    
    return lattice

@nb.njit
def heat_bath_multihit(lattice, L, beta, h=0, multihit=10):
    # Perform multiple heat bath steps in sequence to improve efficiency
    for _ in range(multihit):
        lattice = heat_bath_step(lattice, L, beta, h)
    return lattice

# Main Heat Bath simulation function - NOT fully JIT compiled
def run_ising_heat_bath(L, beta_values, h=0, thermalization=1000, measurements=5000, multihit=1):
    # Set up heat bath simulation of Ising model at various temperatures
    # Structure almost identical to run_ising_metropolis
    energies = np.zeros(len(beta_values))
    energy_squared = np.zeros(len(beta_values))
    magnetizations = np.zeros(len(beta_values))
    mag_abs = np.zeros(len(beta_values))
    mag_squared = np.zeros(len(beta_values))
    
    for b_idx, beta in enumerate(tqdm(beta_values, desc="Temperatures")):
        # Initialize lattice
        lattice = initialize_lattice(L, hot_start=True)
        
        # Thermalize
        for _ in range(thermalization):
            lattice = heat_bath_multihit(lattice, L, beta, h, multihit)
        
        # Measurements
        E_sum = 0
        E2_sum = 0
        M_sum = 0
        M_abs_sum = 0
        M2_sum = 0
        
        for _ in range(measurements):
            lattice = heat_bath_multihit(lattice, L, beta, h, multihit)
            
            # Calculate observables
            E = calculate_energy(lattice, L)
            M = calculate_magnetization(lattice)
            
            E_sum += E
            E2_sum += E**2
            M_sum += M
            M_abs_sum += abs(M)
            M2_sum += M**2
        
        # Calculate averages by dividing the sums by the number of measurements
        energies[b_idx] = E_sum / measurements / (L*L)
        energy_squared[b_idx] = E2_sum / measurements / (L*L)**2
        magnetizations[b_idx] = M_sum / measurements / (L*L)
        mag_abs[b_idx] = M_abs_sum / measurements / (L*L)
        mag_squared[b_idx] = M2_sum / measurements / (L*L)**2
    
    # Calculate specific heat: C = β²(⟨E²⟩ - ⟨E⟩²)
    specific_heat = np.zeros(len(beta_values))
    for b_idx, beta in enumerate(beta_values):
        specific_heat[b_idx] = beta**2 * (energy_squared[b_idx] - energies[b_idx]**2 * (L*L))
    
    return energies, magnetizations, mag_abs, mag_squared, specific_heat

# Task 4b: Check for hysteresis in the ferromagnetic phase
def check_hysteresis(L, beta, h_values, thermalization=1000, measurements=5000, multihit=1):
    # Check for hysteresis by varying external field and measuring the magnetization
    # If hysteresis is present, then the magnetization curves for increasing and decreasing fields should differ and form a hysteresis loop.
    # Occurs in ferromagnetic systems below the critical temperature
    # Initialize magnetizations for increasing and decreasing field
    mag_inc = np.zeros(len(h_values))
    mag_dec = np.zeros(len(h_values))
    
    # Increasing field
    # Start with all spins down
    lattice = -np.ones((L, L))  
    
    for h_idx, h in enumerate(tqdm(h_values, desc="Increasing field")):
        # Thermalize
        for _ in range(thermalization):
            lattice = heat_bath_multihit(lattice, L, beta, h, multihit)
        
        # Measurements
        M_sum = 0
        for _ in range(measurements):
            lattice = heat_bath_multihit(lattice, L, beta, h, multihit)
            M = calculate_magnetization(lattice)
            M_sum += M
        
        mag_inc[h_idx] = M_sum / measurements / (L*L)
    
    # Decreasing field (reverse h_values)
    # Start with all spins up
    lattice = np.ones((L, L)) 
    
    for h_idx, h in enumerate(tqdm(reversed(h_values), desc="Decreasing field")):
        h_rev_idx = len(h_values) - 1 - h_idx
        
        # Thermalize
        for _ in range(thermalization):
            lattice = heat_bath_multihit(lattice, L, beta, h, multihit)
        
        # Measurements
        M_sum = 0
        for _ in range(measurements):
            lattice = heat_bath_multihit(lattice, L, beta, h, multihit)
            M = calculate_magnetization(lattice)
            M_sum += M
        
        mag_dec[h_rev_idx] = M_sum / measurements / (L*L)
    
    return mag_inc, mag_dec

# Task 4c: 3D phase diagram
def phase_diagram_3d(L, beta_values, h_values, thermalization=100, measurements=500, multihit=1):
    # Calculate magnetization as function of temperature and external field
    # Initialize arrays for magnetization and squared magnetization
    magnetizations = np.zeros((len(beta_values), len(h_values)))
    mag_abs = np.zeros((len(beta_values), len(h_values)))
    
    # Create a progress bar for the outer loop only to avoid nested progress bars
    for b_idx, beta in enumerate(tqdm(beta_values, desc="Temperature values")):
        for h_idx, h in enumerate(h_values):
            # Initialize lattice
            lattice = initialize_lattice(L, hot_start=True)
            
            # Thermalize
            for _ in range(thermalization):
                lattice = heat_bath_multihit(lattice, L, beta, h, multihit)
            
            # Measurements
            M_sum = 0
            M_abs_sum = 0
            
            for _ in range(measurements):
                lattice = heat_bath_multihit(lattice, L, beta, h, multihit)
                M = calculate_magnetization(lattice)
                M_sum += M
                M_abs_sum += abs(M)
            
            magnetizations[b_idx, h_idx] = M_sum / measurements / (L*L)
            mag_abs[b_idx, h_idx] = M_abs_sum / measurements / (L*L)
    
    return magnetizations, mag_abs

############################################################################################################################
##### MAIN FUNCTION ########################################################################################################
############################################################################################################################

if __name__ == "__main__":
    # Task 1a: Monte Carlo calculation of pi
    # Try to use large number of samples for good accuracy
    n_samples = 1000000
    pi_estimate = mc_pi_calculation(n_samples)
    print(f"Task 1a: Monte Carlo estimate of π with {n_samples} samples: {pi_estimate}")
    print(f"Exact π: {np.pi}")
    print(f"Relative error: {abs(pi_estimate - np.pi)/np.pi:.8f}")
    
    # Task 1b: Monte Carlo integration of Gaussian
    # Use same number of samples for comparison
    n_samples = 1000000
    integral = mc_gaussian_integral(n_samples)
    exact_integral = np.sqrt(2*np.pi)  # Analytical result
    print(f"\nTask 1b: Monte Carlo integration of exp(-t²/2)")
    print(f"Result with {n_samples} samples: {integral}")
    print(f"Exact result (√(2π)): {exact_integral}")
    print(f"Relative error: {abs(integral - exact_integral)/exact_integral:.8f}")

    #######################
    ##### TASK 1 DONE #####
    #######################

    # Task 2: Exact enumeration for small Ising lattices
    print("\nTask 2: Exact enumeration for Ising model")
    # Range of inverse temperatures
    beta_values = np.linspace(0.1, 1.0, 20)  
    
    # Iterate through lattice sizes as specified in task (2, 3, 4)
    for L in [2, 3, 4]:
        print(f"Computing exact results for L = {L}")
        energies, mags, mag_abs = exact_ising_2d(L, beta_values)
        
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(beta_values, energies, 'o-', label=f'L = {L}')
        plt.xlabel(r'$\beta = 1/T$')
        plt.ylabel(r'$\epsilon = E/L^2$')
        plt.title('Energy Density')
        plt.grid(True)
        plt.legend()
        
        plt.subplot(2, 2, 2)
        plt.plot(beta_values, mags, 'o-', label=f'L = {L}')
        plt.xlabel(r'$\beta = 1/T$')
        plt.ylabel(r'$m = M/L^2$')
        plt.title('Magnetization')
        plt.grid(True)
        plt.legend()
        
        plt.subplot(2, 2, 3)
        plt.plot(beta_values, mag_abs, 'o-', label=f'L = {L}')
        plt.xlabel(r'$\beta = 1/T$')
        plt.ylabel(r'$|m| = |M|/L^2$')
        plt.title('Absolute Magnetization')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'ising_exact_L{L}.png')
        plt.close()

    #######################
    ##### TASK 2 DONE #####
    #######################

    # Task 3a: Metropolis algorithm for 2D Ising model
    print("\nTask 3a: Metropolis algorithm for 2D Ising model")
    L = 128
    beta_values = np.linspace(0.1, 1.0, 20)
    
    # Find optimal parameters for L=128
    thermalization = 1000
    measurements = 5000
    multihit = 5
    
    print(f"Running Metropolis algorithm with L = {L}, thermalization = {thermalization}, measurements = {measurements}, multihit = {multihit}")
    energies, mags, mag_abs, mag_squared, specific_heat = run_ising_metropolis(
        L, beta_values, h=0, thermalization=thermalization, measurements=measurements, multihit=multihit
    )
    
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(beta_values, energies, 'o-')
    plt.xlabel(r'$\beta = 1/T$')
    plt.ylabel(r'$\epsilon = E/L^2$')
    plt.title('Energy Density')
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(beta_values, mag_abs, 'o-')
    plt.xlabel(r'$\beta = 1/T$')
    plt.ylabel(r'$|m| = |M|/L^2$')
    plt.title('Absolute Magnetization')
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.plot(beta_values, specific_heat, 'o-')
    plt.xlabel(r'$\beta = 1/T$')
    plt.ylabel(r'$C_v$')
    plt.title('Specific Heat')
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.plot(beta_values, mag_squared, 'o-')
    plt.xlabel(r'$\beta = 1/T$')
    plt.ylabel(r'$m^2$')
    plt.title('Squared Magnetization')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'ising_metropolis_L{L}.png')
    plt.close()
    
    # Task 3b: Metropolis at critical temperature for different lattice sizes
    print("\nTask 3b: Metropolis at critical temperature for different lattice sizes")
    # Critical inverse temperature
    beta_c = 0.4406868  
    lattice_sizes = [4, 8, 32]
    
    # Results storage
    results_3b = []
    
    for L in lattice_sizes:
        print(f"Running Metropolis at β_c for L = {L}")
        # Use larger statistics for this precise measurement
        sweeps = 200000
        # 20% for thermalization
        thermalization = int(sweeps * 0.2)  
        
        # Initialize lattice
        lattice = initialize_lattice(L, hot_start=True)
        
        # Thermalize
        for _ in tqdm(range(thermalization), desc=f"Thermalization L={L}"):
            lattice = metropolis_multihit(lattice, L, beta_c, h=0, multihit=5)
        
        # Measurements
        E_sum = 0
        E2_sum = 0
        M_sum = 0
        M_abs_sum = 0
        M2_sum = 0
        
        for _ in tqdm(range(sweeps), desc=f"Measurements L={L}"):
            lattice = metropolis_multihit(lattice, L, beta_c, h=0, multihit=5)
            
            # Calculate observables
            E = calculate_energy(lattice, L)
            M = calculate_magnetization(lattice)
            
            E_sum += E
            E2_sum += E**2
            M_sum += M
            M_abs_sum += abs(M)
            M2_sum += M**2
        
        # Calculate averages
        energy = E_sum / sweeps / (L*L)
        magnetization = M_sum / sweeps / (L*L)
        mag_abs_val = M_abs_sum / sweeps / (L*L)
        mag_squared_val = M2_sum / sweeps / (L*L)
        
        results_3b.append({
            'L': L,
            'energy': energy,
            'magnetization': magnetization,
            'magnetization_abs': mag_abs_val,
            'magnetization_squared': mag_squared_val
        })
        
        print(f"L = {L}: energy = {energy:.6f}, |m| = {mag_abs_val:.6f}, m² = {mag_squared_val:.6f}")
    
    #######################
    ##### TASK 3 DONE #####
    #######################

    # Task 4a: Heat Bath algorithm for 2D Ising model
    print("\nTask 4a: Heat Bath algorithm for 2D Ising model")
    L = 128
    beta_values = np.linspace(0.1, 1.0, 20)
    
    print(f"Running Heat Bath algorithm with L = {L}")
    energies_hb, mags_hb, mag_abs_hb, mag_squared_hb, specific_heat_hb = run_ising_heat_bath(
        L, beta_values, h=0, thermalization=thermalization, measurements=measurements, multihit=multihit
    )
    
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(beta_values, energies_hb, 'o-')
    plt.xlabel(r'$\beta = 1/T$')
    plt.ylabel(r'$\epsilon = E/L^2$')
    plt.title('Energy Density (Heat Bath)')
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(beta_values, mag_abs_hb, 'o-')
    plt.xlabel(r'$\beta = 1/T$')
    plt.ylabel(r'$|m| = |M|/L^2$')
    plt.title('Absolute Magnetization (Heat Bath)')
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.plot(beta_values, specific_heat_hb, 'o-')
    plt.xlabel(r'$\beta = 1/T$')
    plt.ylabel(r'$C_v$')
    plt.title('Specific Heat (Heat Bath)')
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.plot(beta_values, mag_squared_hb, 'o-')
    plt.xlabel(r'$\beta = 1/T$')
    plt.ylabel(r'$m^2$')
    plt.title('Squared Magnetization (Heat Bath)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('ising_heatbath_L128.png')
    plt.close()
    
    # Task 4b: Check for hysteresis
    print("\nTask 4b: Checking for hysteresis")
    # Smaller lattice for faster computation
    L = 32 
    # Well into the ferromagnetic phase (T < T_c)
    beta = 0.5  
    # Range of external fields
    h_values = np.linspace(-0.1, 0.1, 20)  
    
    mag_inc, mag_dec = check_hysteresis(
        L, beta, h_values, thermalization=500, measurements=1000, multihit=1
    )
    
    plt.figure(figsize=(10, 6))
    plt.plot(h_values, mag_inc, 'o-', label='Increasing field')
    plt.plot(h_values, mag_dec, 's-', label='Decreasing field')
    plt.xlabel('External field h')
    plt.ylabel('Magnetization m')
    plt.title(f'Hysteresis check at β = {beta}')
    plt.grid(True)
    plt.legend()
    plt.savefig('ising_hysteresis.png')
    plt.close()
    
    # Task 4c: 3D phase diagram
    print("\nTask 4c: 3D phase diagram")
    # Smaller lattice for faster computation
    L = 32  
    # Around critical point
    beta_values = np.linspace(0.3, 0.6, 10)  
    # Moderate field range
    h_values = np.linspace(-0.1, 0.1, 10)  
    
    magnetizations_3d, mag_abs_3d = phase_diagram_3d(
        L, beta_values, h_values, thermalization=200, measurements=500, multihit=1
    )
    
    # Create mesh grid for plotting
    beta_grid, h_grid = np.meshgrid(beta_values, h_values)
    
    # Plot 3D phase diagram for magnetization
    fig = plt.figure(figsize=(12, 10))
    
    # Magnetization plot
    ax1 = fig.add_subplot(121, projection='3d')
    surf1 = ax1.plot_surface(
        beta_grid, h_grid, magnetizations_3d.T, 
        cmap='coolwarm', edgecolor='none'
    )
    ax1.set_xlabel(r'$\beta = 1/T$')
    ax1.set_ylabel('External field h')
    ax1.set_zlabel('Magnetization m')
    ax1.set_title('Magnetization Phase Diagram')
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)
    
    # Absolute magnetization plot
    ax2 = fig.add_subplot(122, projection='3d')
    surf2 = ax2.plot_surface(
        beta_grid, h_grid, mag_abs_3d.T, 
        cmap='viridis', edgecolor='none'
    )
    ax2.set_xlabel(r'$\beta = 1/T$')
    ax2.set_ylabel('External field h')
    ax2.set_zlabel('|Magnetization| |m|')
    ax2.set_title('Absolute Magnetization Phase Diagram')
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)
    
    plt.tight_layout()
    plt.savefig('ising_phase_diagram_3d.png')
    plt.close()
    
    print("\nAll tasks completed!")