############################################################################################################################
##### BASIC IDEA ###########################################################################################################
############################################################################################################################

# GCMC Simulation
# Set up class containing all basic parameters and initialize lists to store measurement data
# Define functions for inserting and removing rods from the grid
# Add collision check
# Define acceptance probability as shown in the script
# Implement Monte Carlo steps to add/remove rods from the grid based on the acceptance probability
# Calculate observables (packing density, order parameter)
# Plots for number of rods (horizontal, vertical, total) and histogramms
# Add visualization script from Ilias
# After preliminary plots, adjust thermalization steps to 15000 and number of Monte Carlo steps to 2**32 as required

# LATEST CHANGES:

# Adjusted code to match the "dual implementation" i found on GitHub
# ---> Reduced number of lines, improved performance
# Why do many function now have two implementations?
# ---> Dual implementation (static @njit + instance method wrapper) should provide significant performance benefits
# ---> Allegedely this can speed up the simulation 10-100 times in computational expensive function (insertion/deletion, colission check)
# ---> What is a wrapper?
# ---> Function or method, that encapsulates another piece of code and serves as intermediary between calling the code and the wrapped function
# ---> Core logic is impemented in the static method, numba can compile these to machine code for speed but cannot directly access class attributes
# ---> The wrapper (without @njit decorator) however is a regular instance of the class, therefore can access class attributes and pass them to the static methods
# -----> Dual implementation allows to speed up the simulation by using numba and at the same time maintain the object oriented structure

# Only set one activity parameter at a time and run simulation seperately
# Laptop decided to do an update during over night run
# ---> :-(

############################################################################################################################
##### LIBRARIES ############################################################################################################
############################################################################################################################

# Import all neccesary libraries
# os --------------------> Interact with operating system, create paths, move data
# numpy -----------------> Numerical calculations, arrays and matrices
# matplotlib.pyplot -----> Plot creation
# numba -----------------> "njit", just-in-time compiler used to accelerate calculations (use @njit to run function in compiled form)
# math ------------------> Basic mathematical operations
# time ------------------> Needed for any time dependant operation
# tqdm ------------------> Progress bars in loops
# random ----------------> Generates random numbers
# collections.Counter ---> Tracks how often a value appears

import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import os
import random
from collections import Counter
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches

############################################################################################################################
##### GRAND CANONICAL MONTE CARLO SIMULATION ###############################################################################
############################################################################################################################

    ########################################################################################################################
    ##### INITIALIZE #######################################################################################################
    ########################################################################################################################

class HardRodsGCMC:
    # Make class for GCMC simulation that includes grid initialization, functions for inserting and deleting rods and measuring
    def __init__(self, M=64, L=8, z=0.84, output_path="C:\\Users\\svens\\OneDrive\\Desktop\\Python\\2_Computational_Physics\\GCMC_Simulation"):
        # Initialize parameters
        self.M = M
        self.L = L
        self.z = z
        self.output_path = output_path
        # Initialize an empty grid
        self.grid = np.zeros((M, M), dtype=np.int32)
        # Lists for storing rod positions
        self.horizontal_rods = []
        self.vertical_rods = []
        # Counters for rod counts
        self.N_h = 0
        self.N_v = 0
        # Lists for storing measurement history
        self.time_steps = []
        self.N_h_history = []
        self.N_v_history = []
        self.N_total_history = []
        self.S_history = []
        self.rho_history = []
        # Counters for histogram data
        self.N_h_hist = Counter()
        self.N_v_hist = Counter()
        self.N_total_hist = Counter()
        self.S_hist = Counter()
        # Lists for order parameter and density measurements
        self.S_abs_history = []
        self.eta_history = []

    ########################################################################################################################
    ##### ADD AND REMOVE RODS  #############################################################################################
    ########################################################################################################################

    @staticmethod
    @njit
    def _check_collision(grid, pos, orientation, M, L):
        # Check if a rod at a given position overlaps with existing rods
        x, y = pos
        if orientation == 1:
            for i in range(L):
                # Periodic boundary conditions
                xi = (x + i) % M
                if grid[xi, y] != 0:
                    return True
        else:
            for i in range(L):
                # Periodic boundary conditions
                yi = (y + i) % M
                if grid[x, yi] != 0:
                    return True
        return False

    def check_collision(self, pos, orientation):
        # Wrapper for collision check
        return self._check_collision(self.grid, pos, orientation, self.M, self.L)

    @staticmethod
    @njit
    def _insert_rod(grid, pos, orientation, M, L):
        # Add a rod at a given position with orientation
        x, y = pos
        if orientation == 1:
            for i in range(L):
                xi = (x + i) % M
                grid[xi, y] = 1
        else:
            for i in range(L):
                yi = (y + i) % M
                grid[x, yi] = 2

    def insert_rod(self, pos=None, orientation=None):
        # Insert a rod into the grid
        if pos is None:
            x = random.randint(0, self.M - 1)
            y = random.randint(0, self.M - 1)
            pos = (x, y)
        if orientation is None:
            orientation = random.choice([1, 2])
        if self.check_collision(pos, orientation):
            return False
        self._insert_rod(self.grid, pos, orientation, self.M, self.L)
        if orientation == 1:
            self.horizontal_rods.append(pos)
            self.N_h += 1
        else:
            self.vertical_rods.append(pos)
            self.N_v += 1
        return True

    @staticmethod
    @njit
    def _remove_rod(grid, pos, orientation, M, L):
        # Remove a rod from the grid
        x, y = pos
        if orientation == 1:
            for i in range(L):
                xi = (x + i) % M
                grid[xi, y] = 0
        else:
            for i in range(L):
                yi = (y + i) % M
                grid[x, yi] = 0

    def remove_rod(self, rod_type=None, idx=None):
        # Remove a rod from the grid and update lists and counters
        if rod_type is None:
            if self.N_h + self.N_v == 0:
                return False
            if self.N_h == 0:
                rod_type = 2
            elif self.N_v == 0:
                rod_type = 1
            else:
                rod_type = random.choice([1, 2])
        if rod_type == 1 and self.N_h == 0:
            return False
        if rod_type == 2 and self.N_v == 0:
            return False
        if idx is None:
            if rod_type == 1:
                idx = random.randint(0, self.N_h - 1)
            else:
                idx = random.randint(0, self.N_v - 1)
        if rod_type == 1:
            pos = self.horizontal_rods[idx]
            self._remove_rod(self.grid, pos, 1, self.M, self.L)
            self.horizontal_rods.pop(idx)
            self.N_h -= 1
        else:
            pos = self.vertical_rods[idx]
            self._remove_rod(self.grid, pos, 2, self.M, self.L)
            self.vertical_rods.pop(idx)
            self.N_v -= 1
        return True

    @staticmethod
    @njit
    def _acceptance_prob(N_h, N_v, z, M, is_insertion):
        # Calculate acceptance probability for insertion or deletion
        N = N_h + N_v
        if is_insertion:
            alpha = min(1.0, (2.0 * M * M) / (N + 1) * z)
        else:
            if N == 0:
                return 0.0
            alpha = min(1.0, N / (2.0 * M * M) / z)
        return alpha

    def gcmc_step(self):
        # Perform a GCMC step (insertion or deletion)
        if random.random() < 0.5:
            orientation = random.choice([1, 2])
            x = random.randint(0, self.M - 1)
            y = random.randint(0, self.M - 1)
            if self.check_collision((x, y), orientation):
                return
            alpha_ins = self._acceptance_prob(self.N_h, self.N_v, self.z, self.M, True)
            if random.random() < alpha_ins:
                self.insert_rod((x, y), orientation)
        else:
            N = self.N_h + self.N_v
            if N == 0:
                return
            alpha_del = self._acceptance_prob(self.N_h, self.N_v, self.z, self.M, False)
            if random.random() < alpha_del:
                if self.N_h == 0:
                    self.remove_rod(rod_type=2)
                elif self.N_v == 0:
                    self.remove_rod(rod_type=1)
                else:
                    if random.random() < self.N_h / N:
                        self.remove_rod(rod_type=1)
                    else:
                        self.remove_rod(rod_type=2)

    ########################################################################################################################
    ##### OBSERVABLES ######################################################################################################
    ########################################################################################################################

    @staticmethod
    @njit
    def _calculate_rho_and_S(N_h, N_v, M, L):
        # Calculate packing density (rho) and order parameter (S)
        N_total = N_h + N_v
        rho = L * N_total / (M * M)
        if N_total > 0:
            S = (N_h - N_v) / N_total
        else:
            S = 0.0
        return rho, S

    def measure_observables(self, step):
        # Measure and record observables
        rho, S = self._calculate_rho_and_S(self.N_h, self.N_v, self.M, self.L)
        N_total = self.N_h + self.N_v
        self.time_steps.append(step)
        self.N_h_history.append(self.N_h)
        self.N_v_history.append(self.N_v)
        self.N_total_history.append(N_total)
        self.S_history.append(S)
        self.rho_history.append(rho)
        self.N_h_hist[self.N_h] += 1
        self.N_v_hist[self.N_v] += 1
        self.N_total_hist[N_total] += 1
        S_discrete = round(S * 100) / 100
        self.S_hist[S_discrete] += 1

        # Record absolute value of S and density for plotting
        self.S_abs_history.append(abs(S))
        self.eta_history.append(rho)

    ########################################################################################################################
    ##### SIMULATION #######################################################################################################
    ########################################################################################################################

    def run_simulation(self, num_steps, measurement_interval, thermalization_steps):
        # Run thermalization phase
        # Add progress bar
        # Set thermalization_steps = 15000, so this progress bar probably isn't really necessary any longer
        # Keep it anyway, just in case -> might come in handy 
        for step in tqdm(range(thermalization_steps), desc="Thermalization"):
            self.gcmc_step()

        # Run data collection phase
        # Add progress bar here -> enables to roughly interpolate duration of simulation
        for step in tqdm(range(num_steps), desc="Simulation progress"):
            self.gcmc_step()
            if step % measurement_interval == 0:
                self.measure_observables(step)

    def save_rod_positions(self):
        # Save rod positions to files for visualization
        with open(os.path.join(self.output_path, "Senkrechte.dat"), "w") as file:
            for pos in self.vertical_rods:
                file.write(f"{pos[0]} {pos[1]}\n")

        with open(os.path.join(self.output_path, "Waagerechte.dat"), "w") as file:
            for pos in self.horizontal_rods:
                file.write(f"{pos[0]} {pos[1]}\n")

    ########################################################################################################################
    ##### PLOTTING #########################################################################################################
    ########################################################################################################################

    def plot_results(self):
        # Plot the time evolution of rod counts
        plt.figure(figsize=(12, 6))
        plt.plot(self.time_steps, self.N_h_history, label='Horizontal Rods ($N_+$)')
        plt.plot(self.time_steps, self.N_v_history, label='Vertical Rods ($N_-$)')
        plt.plot(self.time_steps, self.N_total_history, label='Total Rods ($N$)')
        plt.xlabel('Monte Carlo Steps')
        plt.ylabel('Number of Rods')
        plt.title(f'Time Evolution of Rod Counts for Activity z={self.z}')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.output_path, f'activity_{self.z}_time_evolution.png'))
        plt.close()

    def plot_histograms(self):
        # Plot histograms of observables
        plt.figure(figsize=(18, 6))

        plt.subplot(1, 3, 1)
        plt.bar(list(self.N_h_hist.keys()), self.N_h_hist.values(), width=0.8, color='blue')
        plt.xlabel('Number of Horizontal Rods ($N_+$)')
        plt.ylabel('Frequency')
        plt.title('Histogram of $N_+$')

        plt.subplot(1, 3, 2)
        plt.bar(list(self.N_v_hist.keys()), self.N_v_hist.values(), width=0.8, color='orange')
        plt.xlabel('Number of Vertical Rods ($N_-$)')
        plt.ylabel('Frequency')
        plt.title('Histogram of $N_-$')

        plt.subplot(1, 3, 3)
        plt.bar(list(self.N_total_hist.keys()), self.N_total_hist.values(), width=0.8, color='green')
        plt.xlabel('Total Number of Rods ($N$)')
        plt.ylabel('Frequency')
        plt.title('Histogram of $N$')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, f'histograms_z_{self.z}.png'))
        plt.close()

    def plot_order_vs_density(self):
        # Bin the data to calculate mean and standard deviation for each bin
        bins = np.linspace(min(self.eta_history), max(self.eta_history), 10)
        bin_indices = np.digitize(self.eta_history, bins)

        eta_means = []
        eta_stds = []
        S_abs_means = []
        S_abs_stds = []

        for i in range(1, len(bins)):
            bin_mask = (bin_indices == i)
            if np.any(bin_mask):
                eta_means.append(np.mean(np.array(self.eta_history)[bin_mask]))
                eta_stds.append(np.std(np.array(self.eta_history)[bin_mask]))
                S_abs_means.append(np.mean(np.array(self.S_abs_history)[bin_mask]))
                S_abs_stds.append(np.std(np.array(self.S_abs_history)[bin_mask]))

        # Plot order parameter vs. density with error bars
        plt.figure(figsize=(8, 6))
        plt.errorbar(eta_means, S_abs_means, xerr=eta_stds, yerr=S_abs_stds, fmt='o', color='purple', ecolor='lightgray', capthick=2, capsize=5)
        plt.xlabel('Average Density ($\\eta$)')
        plt.ylabel('Absolute Order Parameter ($|S|$)')
        plt.title(f'Order Parameter vs. Density for Activity z={self.z}')
        plt.grid(True)
        plt.savefig(os.path.join(self.output_path, f'order_vs_density_z_{self.z}.png'))
        plt.close()

    def plot_S_histograms(self):
        # Plot histograms of order parameter S
        plt.figure(figsize=(10, 6))
        
        # Get order parameter values and frequencies
        S_values = sorted(self.S_hist.keys())
        S_frequencies = [self.S_hist[S] for S in S_values]
        
        plt.bar(S_values, S_frequencies, width=0.02, color='purple')
        plt.xlabel('Order Parameter ($S$)')
        plt.ylabel('Frequency')
        plt.title(f'Histogram of Order Parameter S for Activity z={self.z}')
        plt.grid(True)
        plt.axvline(x=0, color='black', linestyle='--', alpha=0.7)  # Add vertical line at S=0
        
        # Set x-axis limits to symmetrical range
        max_abs_S = max(abs(min(S_values)), abs(max(S_values)))
        plt.xlim(-max_abs_S, max_abs_S)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, f'S_histogram_z_{self.z}.png'))
        plt.close()

    #############################################
    ##### VISUALIZATION SCRIPT FROM ILIAS #######
    #############################################

    # Adjusted slightly to match the rest of the code
    # Uses internal data structure 
    # Variable naming to account for different activation values z
    # Get rod length from HardRodsGCMC class (-> self.L) instead of hardcoding the length

    def visualize_rods(self):
        # Visualization script
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')

        # Plot vertical rods
        a = len(self.vertical_rods)
        if a > 0:
            xpos = np.array([pos[0] for pos in self.vertical_rods])
            ypos = np.array([pos[1] for pos in self.vertical_rods])
            dx = np.ones(a)
            dy = np.ones(a) * self.L  # Aspect ratio
            for y in range(a):
                ax.add_patch(
                    patches.Rectangle(
                        (xpos[y], ypos[y]),
                        dx[y],
                        dy[y],
                        facecolor="red",
                        linewidth=0.3
                    )
                )

        # Plot horizontal rods
        a = len(self.horizontal_rods)
        if a > 0:
            xpos = np.array([pos[0] for pos in self.horizontal_rods])
            ypos = np.array([pos[1] for pos in self.horizontal_rods])
            dx = np.ones(a) * self.L  # Aspect ratio
            dy = np.ones(a)
            for y in range(a):
                ax.add_patch(
                    patches.Rectangle(
                        (xpos[y], ypos[y]),
                        dx[y],
                        dy[y],
                        facecolor="blue",
                        linewidth=0.3
                    )
                )

        plt.axis('equal')
        plt.grid()
        fig.savefig(os.path.join(self.output_path, f'Visual2DStaebchen_z_{self.z}.png'), dpi=300, bbox_inches='tight')
        plt.close()

############################################################################################################################
##### MAIN FUNCTION ########################################################################################################
############################################################################################################################

def main():
    # Set activity parameters
    # Run simulation seperately for each parameter
    # Tried one "complete" run over night -> laptop shut down due to update !?
    activities = [0.84] # 0.005, 0.125, 0.25, 0.56, 0.84, 1.1, 1.15, 1.5
    # Thermalization steps (preliminary plots showed thermalization after roughly 15000 steps)
    # "Preliminary plots" refers to the plots for z = 0.56, 0.84 and 1.1 from 3.2.1
    thermalization_steps = 0 # for testing, used to be 15000
    # Number of steps for data collection
    num_steps = 1000000000 # for testing, used to be 2**32
    # Measurement interval 
    measurement_interval = 1000

    for activity in activities:
        print(f"Simulating for activity z={activity}")
        sim = HardRodsGCMC(z=activity)
        sim.run_simulation(num_steps, measurement_interval, thermalization_steps)
        sim.plot_results()
        sim.plot_histograms()
        sim.plot_S_histograms() # Now with histograms for S
        sim.plot_order_vs_density()
        sim.save_rod_positions()
        sim.visualize_rods()
        print(f"Simulation for activity z={activity} finished!")
    print("--------------------")
    print("All tasks completed!")
    print("--------------------")

if __name__ == "__main__":
    main()
