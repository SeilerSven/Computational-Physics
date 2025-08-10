############################################################################################################################
##### BASIC IDEA ###########################################################################################################
############################################################################################################################

# Compute density profile and surface tensionof 1D hard rods of length L at a hard wall
# Use grid with M lattice points for computation
# Split geometry with two hard walls
# Use excess free energy derivative (2.14) and (2.15) to calculate the excess chemical potential
# Use minimization equation (2.11) to calculate the equilibrium density profile
# Surface tension is then calculated as excess grand canonical potential of the equilibrium profile relative to the bulk grand potential

# CHANGED:
# Check Gibbs adsorption relation by comparing the numerical expression with - d gamma/ d mu

# Save plots and data to corresponding folder

############################################################################################################################
##### LIBRARIES ############################################################################################################
############################################################################################################################

# Import all neccesary libraries
# os --------------------> Interact with operating system, create paths, move data
# numpy -----------------> Numerical calculations, arrays and matrices
# matplotlib.pyplot -----> Plot creation
# numba -----------------> "njit", just-in-time compiler used to accelerate calculations (use @njit to run function in compiled form)
# tabulate --------------> Improves readability of data in console
# math ------------------> Basic mathematical operations
# time ------------------> Needed for any time dependant operation
# tqdm ------------------> Progress bars in loops
# matplotlib.colors -----> Colormaps/heatmaps
# scipy.optimize --------> Minimize/Maximize functions, apply constraints etc.


import os
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from tabulate import tabulate
import math
import time
from tqdm import tqdm
from scipy.optimize import curve_fit

############################################################################################################################
##### NUMBA SETUP ##########################################################################################################
############################################################################################################################

# Is this necessary?
# ---> Often seen on Stack Overflow and GitHub

# Contrary to the N-Body simulation, here numba seems to be working correctly

# Set numpy error handling
# Throw error whenever numerical calculations fail
np.seterr(all='raise')
plt.switch_backend("agg")

# Allow the use of numba
use_numba = True

def check_for_numba(func):
    # Check if the use of numba is allowed 
    # If so, then use njit to compile functions to machine code
    # If not, then just leave everything as is
    if use_numba:
        return njit(func)
    else:
        return func

def precompile():
    # Call one of the functions here ahead of time, so that numba is forced to compile and isn't slowed down on the first "real" call
    # ---> Why do i call only one function here? 
    # ---> Choose one of the "main" functions so that other related functions are compiled aswell
    # ---> Compilation of a big part of the code is done before i actually start the calculations but this also takes time...what is the benefit?
    # ---> Precompiling means, that the "compilation cost" is paid before rather than during the first call.
    # ---> For time sensitive applications this means, that there is no delay once the actual program is running. Also there is no one time delay in repeated calls.
    # ---> Maybe not necessary here, but might be useful later 
    if not use_numba:
        print("Numba is not used")
    else:
        print("Precompiling functions ...")
        solve_rho(1, 1, 1, compile=True)
        print("Precompiled functions")

############################################################################################################################
##### DFT BASIC FUNCTIONS ##################################################################################################
############################################################################################################################

# Function used in the functional derivative of the excess free energy evaluated at point s (2.14)

@check_for_numba
def phi_prime(n):
    # (2.15)
    return -1 * np.log(1 - n)

@check_for_numba
def phi(n):
    # (2.5)
    return n + (1 - n) * np.log(1 - n)

############################################################################################################################
##### WEIGHTED DENSITIES ###################################################################################################
############################################################################################################################

# Calculate weighted densities by summing up the local density over a range of lattice points

@check_for_numba
def n_1(s: int, l: int, rho: np.ndarray) -> float:
    # (2.1)
    start = 0 if s - l + 1 < 0 else s - l + 1
    return np.sum(rho[start:s+1])

@check_for_numba
def n_0(s: int, l: int, rho: np.ndarray) -> float:
    # (2.6)
    start = 0 if s - l + 1 < 0 else s - l + 1
    return np.sum(rho[start:s])

@check_for_numba
def n_1_bulk(rho_0: float, l: int) -> float:
    return rho_0 * l

@check_for_numba
def n_0_bulk(rho_0: float, l: int) -> float:
    return rho_0 * (l - 1)

############################################################################################################################
##### EXCESS CHEMICAL POTENTIAL ############################################################################################
############################################################################################################################

@check_for_numba
def mu_ex(rho: np.ndarray, l: int, beta: float = 1) -> np.ndarray:
    # Compute the local excess chemical potential mu_ex(s) at every lattice point s (2.14)
    # Evaluate functional derivative
    res = np.zeros_like(rho)
    N = rho.shape[0]
    for s in range(N):
        sum1 = 0.0
        for s_prime in range(s, min(s + l, N)):
            sum1 += phi_prime(n_1(s_prime, l, rho))
        sum2 = 0.0
        for s_prime in range(s + 1, min(s + l, N)):
            sum2 += phi_prime(n_0(s_prime, l, rho))
        res[s] = sum1 - sum2
    return res / beta

def excess_chemical_potential(rho_0: float, l: int, beta: float = 1):
    # Analytical expression for the excess chemical potential in the homogeneous (bulk) case
    n1 = n_1_bulk(rho_0, l)
    n0 = n_0_bulk(rho_0, l)
    return 1 / beta * (- l * np.log(1 - n1) + (l - 1) * np.log(1 - n0))

@check_for_numba
def excess_chemical_potential_homogeneous(rho_0: float, l: int, beta: float = 1) -> float:
    # Excess chemical potential for homogeneous medium
    return 1 / beta * (- l * np.log(1 - l * rho_0) + (l - 1) * np.log(1 - (l - 1) * rho_0))

############################################################################################################################
##### INITIAL CONDITIONS AND EXTERNAL POTENTIAL ############################################################################
############################################################################################################################

@check_for_numba
def initial_rho(N: int, l: int, eta: float):
    # Initial density profile
    # Constant density inside and zero at the walls
    res = np.zeros(N)
    rho_0 = eta / l
    for i in range(l, N - l):
        res[i] = rho_0
    return res

@check_for_numba
def exp_pot(rho: np.ndarray, l: int):
    # External potential
    # 1 inside and 0 at the walls
    res = np.zeros_like(rho)
    N = rho.shape[0]
    for i in range(l, N - l):
        res[i] = 1.0
        # (2.13)
    return res

############################################################################################################################
##### SOLVE MINIMIZATION EQUATION (FIXPOINT ITERATION) #####################################################################
############################################################################################################################

@check_for_numba
def solve_rho(N: int, l: int, eta: float, beta: float = 1, compile: bool = False):
    # Solve minimization equation (2.11) and botain equilibrium density profile
    # N = number of lattice points
    # l = rod length
    # eta = bulk packing fraction
    # beta = inverse temperature

    # "compile" is used for precompilation, returns dummy value

    # Use pickard iteration with mixing
    # rho_new(s) = (eta/l) * exp( beta*(mu_bulk - mu_ex(s)) ) * exp_pot(s)
    # rho_{i+1}(s) = (1 - α)*rho_i(s) + α*rho_new(s)

    # Added statements to check where error occurs

    epsilon = 1e-12
    if compile:
        return np.zeros(N)
    
    # Choose mixing parameter alpha and maximum steps based on eta and l
    # Mixing parameter is used to achieve convergence
    if eta < 0.8:
        alpha = 0.1
        max_steps = 1000
    else:
        alpha = 0.01
        max_steps = 10000
        if l > 3:
            alpha = 0.001
            max_steps = 100000

    rho_i = initial_rho(N, l, eta)
    pot = exp_pot(rho_i, l)
    mu_hom = excess_chemical_potential_homogeneous(eta / l, l, beta)
    
    step = 0
    while True:
        # (2.16)
        rho_new = (eta / l) * np.exp(beta * (mu_hom - mu_ex(rho_i, l, beta))) * pot
        residual = np.sum((rho_new - rho_i) ** 2)
        if residual < epsilon:
            break
        # (2.17)
        rho_i = (1 - alpha) * rho_i + alpha * rho_new
        step += 1
        if step > max_steps:
            print("Warning: Maximum iterations reached for L=" + str(l) + ", eta=" + str(eta) + " (residual=" + str(residual) + ")")


            break
    
    # Verify that the bulk density is correct
    rho_0_computed = rho_i[N // 2]
    if abs(rho_0_computed - eta / l) > 1e-3:
        print("Warning: Bulk density mismatch for L=" + str(l) + ", eta=" + str(eta) + ". Expected: " + str(round(eta/l, 4)) + ", Got: " + str(round(rho_0_computed, 4)))


    
    return rho_i

############################################################################################################################
##### THERMODYNAMIC QUANTITIES #############################################################################################
############################################################################################################################

def free_energy(rho: np.ndarray, l: int):
    # Calculate free energy (ideal gas part + excess part based on weighted densities)
    N = rho.shape[0]
    wall1, wall2 = (l, N - l)
    rho_bulk = rho[wall1:wall2]
    
    # Ideal gas part
    valid_idx = rho_bulk > 0
    # (2.3)
    F_id = np.sum(rho_bulk[valid_idx] * (np.log(rho_bulk[valid_idx]) - 1))
    
    # Excess part
    F_ex = 0.0
    for s in range(rho.shape[0]):
        # (2.4)
        F_ex += phi(n_1(s, l, rho)) - phi(n_0(s, l, rho))
    
    return F_id + F_ex

def grand_potential(rho: np.ndarray, l: int):
    # Calculate grand potential
    N = rho.shape[0]
    wall1, wall2 = (l, N - l)
    rho_0 = rho[N // 2]
    mu_val = excess_chemical_potential(rho_0, l) + np.log(rho_0)
    F = free_energy(rho, l)
    sum_part = -mu_val * np.sum(rho)
    # (2.7)
    return F + sum_part

def grand_potential_density(rho: np.ndarray, l: int):
    # Calculate grand potential density at all lattice points
    N = rho.shape[0]
    rho_0 = rho[N // 2]
    mu_val = excess_chemical_potential(rho_0, l) + np.log(rho_0)
    
    omega = np.zeros_like(rho)
    
    for s in range(N):
        # Ideal gas contribution 
        if rho[s] > 0:
            omega_id = rho[s] * (np.log(rho[s]) - 1)
        else:
            omega_id = 0
        
        # Excess free energy contribution
        omega_ex = phi(n_1(s, l, rho)) - phi(n_0(s, l, rho))
        
        # Chemical potential contribution
        omega_mu = -mu_val * rho[s]
        
        omega[s] = omega_id + omega_ex + omega_mu
    
    return omega

def analyze_asymmetry(rho: np.ndarray, l: int):
    # Symmetry analysis
    # Check if grand potential (omega) is symmetrically distributed around the middle of the slit
    # If so, then the left and right side of the potential well should be mirrored (?)

    # Get size of the grid
    N = rho.shape[0]
    # Calculate midpoint of the slit
    midpoint = N // 2
    
    omega = grand_potential_density(rho, l)
    # Extract density profile from left wall to the middle
    omega_left = omega[l:midpoint]
    # Extract density profile from the middle to the right wall
    omega_right = np.flip(omega[midpoint:N-l])
    
    # Calculate asymmetry
    # Find shorter side of the "halfs"
    min_len = min(len(omega_left), len(omega_right))
    # Calculate assymetry as sum of the absoulte differences
    asymmetry = np.sum(np.abs(omega_left[:min_len] - omega_right[:min_len]))
    
    return asymmetry, omega_left, omega_right

def pressure(rho_0: float, l: int):
    # Calculate analytical expression for pressure
    # n^(1) = ρ₀·L and n^(0) = ρ₀·(L-1)
    n1 = n_1_bulk(rho_0, l)
    n0 = n_0_bulk(rho_0, l)
    return np.log(1 - n0) - np.log(1 - n1)

def surface_tension_numerical(rho: np.ndarray, l: int):
    # Calculate surface tension numerically
    N = rho.shape[0]
    wall1, wall2 = (l, N - l)
    rho_0 = rho[N // 2]
    N_free = wall2 - wall1
    grand_potential_hom = -pressure(rho_0, l) * N_free
    # (2.19)
    return 0.5 * (grand_potential(rho, l) - grand_potential_hom)

def surface_tension_analytical(rho: np.ndarray, l: int):
    # Calculate surface tension analytically using weighted densities
    rho_0 = rho[rho.shape[0] // 2]
    n1 = n_1_bulk(rho_0, l)
    n0 = n_0_bulk(rho_0, l)
    # (2.22)
    return 0.5 * ((l - 1) * np.log(1 - n1) - l * np.log(1 - n0))

def excess_adsorption(rho: np.ndarray, l: int, eta: float):
    # Calculate excess adsorption
    N = rho.shape[0]
    wall1, wall2 = (l, N - l)
    rho_0 = rho[N // 2]
    
    # Sum over one half of the slit
    excess = np.sum(rho[wall1:N//2] - rho_0)
    
    #(2.23)
    return excess # Return only excess (and not 2 * excess) so that it can be compared with - d gamma / d mu

############################################################################################################################
##### GIBBS ADSORPTION CHECK ###############################################################################################
############################################################################################################################

# Fixed, removed gibbs_check and used excess_adsorption, compute_adsorption_derivative and verify_gibbs_adsorption to check the Gibbs relation
# Print statements for comparison of both values and adjusted check_gibbs_plot so that it now uses the correct values

# def gibbs_check(rho: np.ndarray, l: int, eta: float):
#     # Check Gibbs adsorption
#     # Compare excess adsorption with the derivative of surface tension with respect to the chemical potential
#     # Return value of 2*(l-1)*p(0) -> compare with excess adsorption
#     N = rho.shape[0]
#     rho_0 = rho[N // 2]
    
#     # Calculate pressure at the center of the slit
#     p_0 = pressure(rho_0, l)
    
#     # Return 2*(l-1)*p(0) as before
#     return 2 * (l - 1) * p_0

def compute_adsorption_derivative(eta_values, gamma_values, l: int):
    # Calculate the derivative of surface tension with respect to chemical potential
    # Convert packing fractions to bulk densities
    rho_0_values = eta_values / l

    # Calculate chemical potential for each value (μ = ln(ρ₀) + μ_ex)
    mu_values = np.log(rho_0_values) + np.array([excess_chemical_potential_homogeneous(rho_0, l) for rho_0 in rho_0_values])

    # Calculate numerical derivative dγ/dμ
    dgamma_dmu = np.zeros_like(eta_values)

    for i in range(1, len(mu_values) - 1):
        dgamma_dmu[i] = (gamma_values[i+1] - gamma_values[i-1]) / (mu_values[i+1] - mu_values[i-1])

    # Handle endpoints
    dgamma_dmu[0] = (gamma_values[1] - gamma_values[0]) / (mu_values[1] - mu_values[0])
    dgamma_dmu[-1] = (gamma_values[-1] - gamma_values[-2]) / (mu_values[-1] - mu_values[-2])

    # Return negative derivative to match Gibbs equation: Γ = -dγ/dμ
    return -dgamma_dmu

############################################################################################################################
##### PLOTTING FUNCTIONS ###################################################################################################
############################################################################################################################

# Plotting functions

def plot_rho(rho: np.ndarray, l: int, eta: float, filename: str):
    # Plot density profile near one wall (from s = l to s = 6*l)
    cutoff = 6 * l
    rho_trunc = rho[l:cutoff]
    idx = np.arange(len(rho_trunc))
    plt.figure(dpi=300)
    plt.plot(idx, rho_trunc, 'o-', label=f"ρ(s) (L={l}, η={eta})")
    plt.axhline(y=eta/l, color='r', linestyle='--', label=f"Bulk ρ₀={eta/l:.3f}")
    plt.xlabel("Lattice point s (measured from wall)")
    plt.ylabel("Local density ρ(s)")
    plt.title(f"Density profile at a hard wall (L={l}, η={eta})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(filename)
    plt.close()

def plot_grand_potential_density(rho: np.ndarray, l: int, eta: float, filename: str):
    # Plot the grand potential density near one wall
    N = rho.shape[0]
    omega = grand_potential_density(rho, l)
    
    # Extract profiles for one wall
    cutoff = 10 * l
    omega_wall = omega[l:l+cutoff]
    
    plt.figure(dpi=300, figsize=(10, 6))
    plt.plot(np.arange(len(omega_wall)), omega_wall, 'o-', label="ω(s)")
    plt.axhline(y=-pressure(rho[N//2], l), color='r', linestyle='--', label="Bulk value -p(ρ₀)")
    plt.xlabel("Lattice point s (measured from wall)")
    plt.ylabel("Grand potential density ω(s)")
    plt.title(f"Grand potential density profile at a hard wall (L={l}, η={eta})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add asymmetry information
    asymmetry, _, _ = analyze_asymmetry(rho, l)
    plt.annotate(f"Asymmetry measure: {asymmetry:.6e}", 
                 xy=(0.5, 0.05), xycoords='axes fraction', 
                 bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3))
    
    plt.savefig(filename)
    plt.close()

def plot_gibbs_check(eta_list, gamma_list, excess_ads_list, l, filename):
    plt.figure(figsize=(10, 6))

    # Plot excess adsorption
    plt.plot(eta_list, excess_ads_list, 'o-', label=r'Excess Adsorption ($\Gamma = \sum (\rho(s) - \rho_0)$)')

    # Calculate and plot -dγ/dμ
    dgamma_dmu = compute_adsorption_derivative(eta_list, gamma_list, l)
    plt.plot(eta_list, dgamma_dmu, 's--', label='-dγ/dμ')

    plt.xlabel('Packing Fraction (η)')
    plt.ylabel('Value')
    plt.title(f'Verification of Gibbs Adsorption Equation for L={l}')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Save plot
    plt.savefig(filename)
    plt.close()

def plot_comparison_with_analytical_surface_tension(eta_list, st_num_list, st_an_list, l, filename):
    # Plot the comparison between numerical and analytical surface tension values
    plt.figure(dpi=300)
    plt.plot(eta_list, st_num_list, 'o-', label='Numerical')
    plt.plot(eta_list, st_an_list, 'x--', label='Analytical')
    plt.xlabel('Packing fraction η')
    plt.ylabel('Surface tension γ')
    plt.title(f'Surface tension: Numerical vs. Analytical (L={l})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Calculate and display absolute mean error
    mae = np.mean(np.abs(np.array(st_num_list) - np.array(st_an_list)))
    plt.annotate(f"Mean Absolute Error: {mae:.6e}", 
                 xy=(0.5, 0.05), xycoords='axes fraction', 
                 bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3))
    
    plt.savefig(filename)
    plt.close()

############################################################################################################################
##### MAIN FUNCTION: RUN SIMULATIONS AND SAVE RESULTS ######################################################################
############################################################################################################################

def run_all():
    # Start timing
    start_time = time.time()
    print("Starting simulation...")

    # Define path for results
    results_folder = "C:\\Users\\svens\\OneDrive\\Desktop\\Python\\2_Computational_Physics\\Classical_Density_Function"
    os.makedirs(results_folder, exist_ok=True)

    # Parameters: rod lengths and bulk packing fractions
    # Make accessible for other functions
    global l_list
    l_list = [3, 10]
    # Number of lattice points
    N = 1000
    eta_list = np.arange(0.1, 1.0, 0.1)

    # Estimate total iterations and time
    total_configs = len(l_list) * len(eta_list)
    print(f"Running {total_configs} configurations in total...")

    # Track times for estimation
    config_times = []

    # Initialize all result data structures for the final tables
    all_results = []
    all_st_num = []
    all_st_an = []
    all_gibbs = []
    all_excess = []

    # For each rod length, create a separate results collection
    for l_idx, l in enumerate(l_list):
        l_start_time = time.time()
        print(f"\n--- Running simulations for L={l} ---\n")

        # Results for this specific rod length
        l_results_table = []
        l_st_num_list = []
        l_st_an_list = []
        l_gibbs_values = []
        l_excess_ads_list = []

        # Use tqdm for progress bar on eta iterations
        for eta_idx, eta in enumerate(tqdm(eta_list, desc=f"Progress for L={l}")):
            config_start_time = time.time()
            eta = round(eta, 1)
            print(f"  Starting configuration L={l}, eta={eta}...")

            # Solve for equilibrium density profile
            rho_eq = solve_rho(N, l, eta, beta=1)

            # Save density profile plot
            plot_filename = os.path.join(results_folder, f"density_profile_L{l}_eta{eta}.png")
            plot_rho(rho_eq, l, eta, plot_filename)

            # Calculate thermodynamic quantities
            st_num = surface_tension_numerical(rho_eq, l)
            st_an = surface_tension_analytical(rho_eq, l)
            ex_ads = excess_adsorption(rho_eq, l, eta)

            # Analyze grand potential density asymmetry
            asymmetry, _, _ = analyze_asymmetry(rho_eq, l)

            # Save grand potential density plot
            gp_filename = os.path.join(results_folder, f"grand_potential_density_L{l}_eta{eta}.png")
            plot_grand_potential_density(rho_eq, l, eta, gp_filename)

            # Store values for current rod length tables and plots
            l_results_table.append([l, eta, st_num, st_an, ex_ads, asymmetry])
            l_st_num_list.append(st_num)
            l_st_an_list.append(st_an)
            l_excess_ads_list.append(ex_ads)

            # Also add to the global results
            all_results.append([l, eta, st_num, st_an, ex_ads, asymmetry])
            all_st_num.append(st_num)
            all_st_an.append(st_an)
            all_excess.append(ex_ads)

            # Calculate and report time for this configuration
            config_time = time.time() - config_start_time
            config_times.append(config_time)

            # Print results for this configuration
            print(f"Results for L={l}, eta={eta}:")
            print(f"- Surface tension (num): {st_num:.6e}")
            print(f"- Surface tension (ana): {st_an:.6e}")
            print(f"- Asymmetry: {asymmetry:.6e}")
            print(f"---> Configuration completed in {config_time:.2f} seconds")

            # Estimate remaining time
            configs_done = l_idx * len(eta_list) + eta_idx + 1
            configs_left = total_configs - configs_done
            avg_time = sum(config_times) / len(config_times)
            est_time_left = configs_left * avg_time

            print(f"Progress: {configs_done}/{total_configs} configurations completed")
            print(f"Estimated time remaining: {est_time_left:.2f} seconds ({est_time_left/60:.2f} minutes)")

        dgamma_dmu = compute_adsorption_derivative(eta_list, l_st_num_list, l)

        # # Print values
        # for eta_idx, eta in enumerate(eta_list):
        #     ex_ads = l_excess_ads_list[eta_idx]
        #     print(f"Excess Adsorption (Γ) for η={eta}: {ex_ads:.6e}")
        #     print(f"-dγ/dμ for η={eta}: {-dgamma_dmu[eta_idx]:.6e}")
        #     print(f"Difference (Γ - (-dγ/dμ)) for η={eta}: {abs(ex_ads - (abs(-dgamma_dmu[eta_idx]))):.6e}\n")

        # Save results table for this rod length
        headers = ["L", "η", "γ (num)", "γ (ana)", "Excess Ads", "Asymmetry"]
        table = tabulate(l_results_table, headers=headers, tablefmt="grid", floatfmt=".6e")
        table_filename = os.path.join(results_folder, f"results_L{l}.txt")
        with open(table_filename, "w", encoding="utf-8") as f:
            f.write(table)

        # Plot Gibbs check for this rod length
        gibbs_filename = os.path.join(results_folder, f"gibbs_check_L{l}.png")
        plot_gibbs_check(eta_list, l_st_num_list, l_excess_ads_list, l, gibbs_filename)

        # Plot comparison with analytical results
        comp_filename = os.path.join(results_folder, f"surf_tension_comparison_L{l}.png")
        plot_comparison_with_analytical_surface_tension(eta_list, l_st_num_list, l_st_an_list, l, comp_filename)

        # Report time for this rod length
        l_time = time.time() - l_start_time
        print(f"\nCompleted all simulations for L={l} in {l_time:.2f} seconds ({l_time/60:.2f} minutes)")

    # Use tabulate for combined table
    headers = ["L", "η", "γ (num)", "γ (ana)", "Excess Ads", "Asymmetry"]
    all_table = tabulate(all_results, headers=headers, tablefmt="grid", floatfmt=".6e")
    all_table_filename = os.path.join(results_folder, "results_all.txt")
    with open(all_table_filename, "w", encoding="utf-8") as f:
        f.write(all_table)

    # Final timing report
    total_time = time.time() - start_time
    print(f"\nAll results saved to {results_folder}/")
    print(f"Simulation complete!")
    print(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Average time per configuration: {total_time/total_configs:.2f} seconds")

    # Write timing data to file
    timing_file = os.path.join(results_folder, "timing_data.txt")
    with open(timing_file, "w", encoding="utf-8") as f:
        f.write(f"Total configurations: {total_configs}\n")
        f.write(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)\n")
        f.write(f"Average time per configuration: {total_time/total_configs:.2f} seconds\n\n")

        f.write("Time per configuration:\n")
        for i, time_val in enumerate(config_times):
            l_idx = i // len(eta_list)
            eta_idx = i % len(eta_list)
            l = l_list[l_idx]
            eta = round(eta_list[eta_idx], 1)
            f.write(f"L={l}, eta={eta}: {time_val:.2f} seconds\n")

if __name__ == "__main__":
    # Precompile Numba functions (optional, but should improve performance?)
    precompile()
    # Run all simulations
    run_all()