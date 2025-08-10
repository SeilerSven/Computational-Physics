import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import pandas as pd

class LichtkurveAnalyzer:
    def __init__(self, filename='lichtkurve.dat'):
        self.load_data(filename)
        
    def load_data(self, filename):
    # Load data from lichtkurve.dat 
    # Don't know how to open the file, so try different methods/formats
        try:
            try:
                # Format 1: Two columns separated with spaces
                data = np.loadtxt(filename, delimiter=None)
                self.t = data[:, 0]  # time
                self.f = data[:, 1]  # intensity
            except:
                # Format 2: Read with pandas
                data = pd.read_csv(filename, sep='\s+', header=None)
                self.t = data.iloc[:, 0].values
                self.f = data.iloc[:, 1].values
                
            
            self.sigma = np.std(self.f) * 0.1  # 10% of standard deviation
            self.ndata = len(self.t)
            
            print(f"Data loaded: {self.ndata} data points")
            print(f"Time interval: {self.t.min():.3f} to {self.t.max():.3f}")
            print(f"Intensity interval: {self.f.min():.3f} to {self.f.max():.3f}")
            print(f"Assumed measurement error σ: {self.sigma:.3f}")
            
        except Exception as e:
            print(f"Error while loading the file {filename}: {e}")
            self.generate_example_data()
    
    def generate_example_data(self):
        # Example data for testing
        self.t = np.linspace(0, 100, 200)
        # Linearer trend + sinus component + noise
        self.f = 0.5 * self.t + 20 + 10 * np.sin(0.1 * self.t) + \
                 5 * np.sin(0.3 * self.t) + np.random.normal(0, 2, len(self.t))
        self.sigma = 2.0
        self.ndata = len(self.t)
    
    def linear_model(self, t, params):
        # Define linear model
        a, b = params
        return a * t + b
    
    def sinusoidal_model(self, t, params):
        # Define model with linear trend and sinus-components
        # f(t) = a*t + b + sum(A_i * sin(ω_i * t + φ_i))
        
        a, b = params[0], params[1]
        linear_part = a * t + b
        
        # Sinus-components
        sinusoidal_part = 0
        n_modes = (len(params) - 2) // 3
        
        for i in range(n_modes):
            idx = 2 + i * 3
            A = params[idx]          # Amplitude
            omega = params[idx + 1]  # Frequency
            phi = params[idx + 2]    # Phase
            sinusoidal_part += A * np.sin(omega * t + phi)
        
        return linear_part + sinusoidal_part
    
    def chi_squared(self, params, model_func):
        # Calculate Chi-squared
        model_values = model_func(self.t, params)
        chi2 = np.sum(((self.f - model_values) / self.sigma) ** 2)
        return chi2
    
    def fit_linear(self):
        # Adjust linear function
        print("\n=== Linear Adjustment ===")
        
        # Guess starting values
        a_init = (self.f[-1] - self.f[0]) / (self.t[-1] - self.t[0])
        b_init = np.mean(self.f) - a_init * np.mean(self.t)
        
        initial_params = [a_init, b_init]
        
        # Set boundaries for parameters (analogous to FORTRAN example)
        f_range = self.f.max() - self.f.min()
        t_range = self.t.max() - self.t.min()
        
        bounds = [
            (-10 * f_range / t_range, 10 * f_range / t_range),  # a
            (self.f.min() - f_range, self.f.max() + f_range)     # b
        ]
        
        # Optimization (Minimization in PIKAIA)
        result = minimize(
            self.chi_squared,
            initial_params,
            args=(self.linear_model,),
            bounds=bounds,
            method='L-BFGS-B'
        )
        
        self.linear_params = result.x
        self.linear_chi2 = result.fun
        
        print(f"Linear parameter: a = {self.linear_params[0]:.6f}, b = {self.linear_params[1]:.6f}")
        print(f"Chi-squared: {self.linear_chi2:.6f}")
        print(f"Reduced Chi-squared: {self.linear_chi2 / (self.ndata - 2):.6f}")
        
        return self.linear_params, self.linear_chi2
    
    def fit_sinusoidal(self, n_modes):
        # Fit with n sinus-modes
        print(f"\n--- Fitting with {n_modes} Sinus-modes ---")
    
        n_params = 2 + 3 * n_modes
        
        # Starting values
        initial_params = list(self.linear_params)  # Linear parameters
        
        for i in range(n_modes):
            # Initialize sinus-parameters
            A_init = np.std(self.f) * 0.5  # Amplitude
            omega_init = 2 * np.pi * (i + 1) / (self.t.max() - self.t.min())  # Frequency
            phi_init = 0.0  # Phase
            initial_params.extend([A_init, omega_init, phi_init])
        
        # Define boundaries
        f_range = self.f.max() - self.f.min()
        t_range = self.t.max() - self.t.min()
        
        bounds = [
            (-10 * f_range / t_range, 10 * f_range / t_range),      # a
            (self.f.min() - f_range, self.f.max() + f_range)        # b
        ]
        
        # Boundaries for sinus-parameters
        for i in range(n_modes):
            bounds.extend([
                (0, 2 * f_range),                   # Amplitude
                (0.001, 10 * 2 * np.pi / t_range),  # Frequency
                (-np.pi, np.pi)                     # Phase
            ])
        
        # Optimization
        result = minimize(
            self.chi_squared,
            initial_params,
            args=(self.sinusoidal_model,),
            bounds=bounds,
            method='L-BFGS-B'
        )
        
        params = result.x
        chi2 = result.fun
        
        print(f"Linear parameter: a = {params[0]:.6f}, b = {params[1]:.6f}")
        for i in range(n_modes):
            idx = 2 + i * 3
            A = params[idx]
            omega = params[idx + 1]
            phi = params[idx + 2]
            period = 2 * np.pi / omega
            print(f"Mode {i+1}: A = {A:.4f}, ω = {omega:.4f}, φ = {phi:.4f}, T = {period:.4f}")
        
        print(f"Chi-squared: {chi2:.6f}")
        print(f"Reduced Chi-squared: {chi2 / (self.ndata - n_params):.6f}")
        
        return params, chi2
    
    def systematic_mode_analysis(self, max_modes=5):
        # Analyze number of modes
        print("\n" + "="*50)
        print("Analysis fo sinus modes")
        print("="*50)
        
        results = []
        
        linear_params, linear_chi2 = self.fit_linear()
        results.append({
            'n_modes': 0,
            'params': linear_params,
            'chi2': linear_chi2,
            'reduced_chi2': linear_chi2 / (self.ndata - 2),
            'n_params': 2
        })
        
        # Test various number of modes
        for n_modes in range(1, max_modes + 1):
            try:
                params, chi2 = self.fit_sinusoidal(n_modes)
                n_params = 2 + 3 * n_modes
                reduced_chi2 = chi2 / (self.ndata - n_params)
                
                results.append({
                    'n_modes': n_modes,
                    'params': params,
                    'chi2': chi2,
                    'reduced_chi2': reduced_chi2,
                    'n_params': n_params
                })
                
            except Exception as e:
                print(f"Error at {n_modes} Modes: {e}")
                break
        
        # Show results
        print("\n" + "="*80)
        print("RESULTS")
        print("="*80)
        print(f"{'Modes':<6} {'Parameters':<10} {'Chi²':<12} {'Red. Chi²':<12} {'Improvement':<15}")
        print("-" * 80)
        
        for i, result in enumerate(results):
            improvement = ""
            if i > 0:
                delta_chi2 = results[i-1]['chi2'] - result['chi2']
                improvement = f"{delta_chi2:.2f}"
            
            print(f"{result['n_modes']:<6} {result['n_params']:<10} "
                  f"{result['chi2']:<12.4f} {result['reduced_chi2']:<12.4f} {improvement:<15}")
        
        # Get best number of modes for fit
        best_idx = min(range(len(results)), key=lambda i: results[i]['reduced_chi2'])
        best_result = results[best_idx]
        
        print(f"\nBestnumber of modes: {best_result['n_modes']}")
        print(f"Reduced Chi-squared: {best_result['reduced_chi2']:.6f}")
        
        self.results = results
        return results
    
    def plot_results(self, max_modes=5):
        # Plot results
        if not hasattr(self, 'results'):
            return
        
        n_plots = min(len(self.results), max_modes + 1)
        
        # Arrange subplots dynamically based on the number of subplots
        if n_plots <= 4:
            rows, cols = 2, 2
            figsize = (15, 10)
        elif n_plots <= 6:
            rows, cols = 2, 3
            figsize = (20, 10)
        else:
            rows, cols = 3, 3
            figsize = (20, 15)
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if rows * cols == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        # Plot data points and models
        for i in range(n_plots):
            if i < len(axes):
                ax = axes[i]
                result = self.results[i]
                
                # Original data
                ax.errorbar(self.t, self.f, yerr=self.sigma, fmt='o', alpha=0.6, 
                           markersize=3, label='Data')
                
                # Model
                t_model = np.linspace(self.t.min(), self.t.max(), 1000)
                if result['n_modes'] == 0:
                    f_model = self.linear_model(t_model, result['params'])
                else:
                    f_model = self.sinusoidal_model(t_model, result['params'])
                
                ax.plot(t_model, f_model, 'r-', linewidth=2, 
                       label=f'{result["n_modes"]} Modes')
                
                ax.set_xlabel('Time')
                ax.set_ylabel('Intensity')
                ax.set_title(f'{result["n_modes"]} Modes - red. χ² = {result["reduced_chi2"]:.4f}')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        for i in range(n_plots, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
        
        # Chi-squared evolution
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        modes = [r['n_modes'] for r in self.results]
        chi2_values = [r['chi2'] for r in self.results]
        reduced_chi2_values = [r['reduced_chi2'] for r in self.results]
        
        ax1.plot(modes, chi2_values, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Number of modes')
        ax1.set_ylabel('Chi-squared')
        ax1.set_title('Chi-squared vs. Number of modes')
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(modes, reduced_chi2_values, 'ro-', linewidth=2, markersize=8)
        ax2.set_xlabel('Number of modes')
        ax2.set_ylabel('Reduced Chi-squared')
        ax2.set_title('Reduced Chi-squared vs. Number of modes')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def main():
    # Main function
    print("Show analysis")
    print("Based on Pikaia user manual section 5.2 and 5.3")
    print("="*60)
    
    # Initialize analyzer
    analyzer = LichtkurveAnalyzer('lichtkurve.dat')
    
    #Analysis
    results = analyzer.systematic_mode_analysis(max_modes=6) # 6 modes, as the first one (mode 0) corresponds to the linear model
    
    # Visualize results
    analyzer.plot_results(max_modes=5)
    return analyzer

if __name__ == "__main__":
    analyzer = main()