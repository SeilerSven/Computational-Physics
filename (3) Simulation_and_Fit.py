############################################################################################################################
##### BASIC IDEA ###########################################################################################################
############################################################################################################################

# Explained individually for each task

# Restructured code to now use main function and split tasks 

# Split tasks into several programs next time to avoid confusion with similarly named functions and parameters!

############################################################################################################################
##### LIBRARIES ############################################################################################################
############################################################################################################################

# Import all neccesary libraries
# numpy -----------------> Numerical calculations, arrays and matrices
# pandas ----------------> Data processing, reading in data
# matplotlib.pyplot -----> Plot creation
# scipy.optimize --------> Optimization algorithms
# uncertainties ---------> Handling of numbers with uncertainties
# scipy.stats -----------> Statistical functions and distributions

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, fmin_cg
from uncertainties import ufloat, umath
from scipy import stats
from scipy.stats import chi2
from numpy.polynomial import Polynomial, Legendre

############################################################################################################################
##### TASK 1 ###############################################################################################################
############################################################################################################################

# Three different measurements of the speed of light are given (Bergstrand, Essen, Jones). 
# The task is to find the weighted average and the error of the average.
# A confidence intervall of 95% is assumed -> 5% probability of error
# Differentiate between inner and outer error
# Inner error: calculated solely from the individual errors
# Outer error: does account for the spread of the mean values
# The inner and outer variance have to be calculated as well

def transform_uncertainties(data):
   # Function takes list of tuples (measurement value and corresponding error) and divides every error by 1.96, then the updated list is returned
   # ---> Scales the error onto a 95% confidence interval
    return [(value, error / 1.96) for value, error in data]

def weighted_mean_and_uncertainties(measurements):
    # Calculate weights and then apply them to the mean and the errors
    # Create list of squared errors 
    weights = 1 / np.array([error ** 2 for _, error in measurements])
    weighted_mean = np.sum(np.array([value for value, _ in measurements]) * weights) / np.sum(weights)
    weighted_error = np.sqrt(1 / np.sum(weights))
    return weighted_mean, weighted_error

def internal_uncertainty_task1(measurements):
    # The internal error is calculated as reciprocal sum of the individual errors according to gaussian error propagation
    # Calculate the sum first and then return the reciprocal square root
    sum_reciprocal_variances = np.sum(1 / np.array([error ** 2 for _, error in measurements]))
    return np.sqrt(1 / sum_reciprocal_variances)


def chi(measurements, mean_value):
    # Function calculates the chi squared value step by step

    values = np.array([value for value, _ in measurements]) 
    errors = np.array([error for _, error in measurements])

    differences = values - mean_value

    normalized_differences = differences / errors

    squared_differences = normalized_differences ** 2

    sum_squared_differences = np.sum(squared_differences)

    chi_squared_value = np.sqrt(sum_squared_differences)

    return chi_squared_value


def check_goodness_of_fit_task1(chi_squared_value, tolerance = 0.05):
    # Check if chi^2/(N-M) is close to 1
    # In this case N=3 and M=1
    # ---> Changed: Now the result of the calculation is returned and "is_good_fit" value is adressed earlier

    fit_parameter = chi_squared_value / (3 - 1) 
    result = abs(fit_parameter)

    is_good_fit = result <= tolerance
    return fit_parameter, result, is_good_fit
    
    

def external_uncertainty_task1(fit_parameter, internal_error):
    # The external uncertainty is calculated by multiplying the fit parameter and the squared internal uncertainty
    
    return np.sqrt(fit_parameter * internal_error ** 2)


############################################################################################################################
##### TASK 2 ###############################################################################################################
############################################################################################################################

# A current-voltage measurement series is given including the error for the voltage
# Calculate the ohmic resistance
# Fit the functions:
# ---> I = a * U (with a = 1 / R)
# ---> a * U + b (with a = ! / R and b = I_0)
# Calculate chi^2 and the internal for R and I_0


# def load_data(filename):
    # Load current-voltage measurement data (copied and saved to .txt file, delimeter are whitespaces)

    # filename = "C://Users//svens//OneDrive//Desktop//Python//2_Computational_Physics//Simulation_und_Fit_experimenteller_Daten//A2.txt"
    # # Load data using numpy, skip header and use white spaces for separation
    # data = pd.read_csv(filename, delimiter='\t', skiprows=1)
    # U = data.iloc[:, 0].to_numpy()          # Second column --> voltage
    # I = data.iloc[:, 1].to_numpy()          # Third column ---> current
    # delta_I = data.iloc[:, 2].to_numpy()    # Fourth column --> uncertainty
    # return U, I, delta_I

def load_data_task2():
    # Define data by hand to prove the correct values are being used
    # However these are EXACTLY the same values as in file A2.txt...
    # ---> plotted the wrong column (counter instead of voltage)
    U = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0])
    I = np.array([0.065, 0.206, 0.405, 0.492, 0.606, 0.782, 0.865, 1.018, 1.199, 1.327, 1.408, 1.627]) 
    delta_I = np.array([0.01, 0.01, 0.02, 0.02, 0.02, 0.02, 0.02, 0.03, 0.03, 0.03, 0.03, 0.04])
    
    return U, I, delta_I

# Define auxiliary functions S, Sx, Sy, Sxx, Sxy and D for the calculation of the linear regression
def S(xs, ys):
    sigma_s = np.array([y.std_dev for y in ys])
    return np.sum(1 / sigma_s**2)

def Sx(xs, ys):
    sigma_s = np.array([y.std_dev for y in ys])
    return np.sum(xs / sigma_s**2)

def Sy(xs, ys):
    sigma_s = np.array([y.std_dev for y in ys])
    return np.sum([y.nominal_value for y in ys] / sigma_s**2)

def Sxx(xs, ys):
    sigma_s = np.array([y.std_dev for y in ys])
    return np.sum((xs / sigma_s)**2)

def Sxy(xs, ys):
    sigma_s = np.array([y.std_dev for y in ys])
    return np.sum(xs * np.array([y.nominal_value for y in ys]) / sigma_s**2)

# Calculate determinant
# D = S * Sxx - Sx^2
def D(xs, ys):
    return S(xs, ys) * Sxx(xs, ys) - Sx(xs, ys)**2


def lin_regression(xs, ys):
    # Calculate linear regression with auxiliary functions ---> swapped formula back to how they should be
    # a = (S * Sxy - Sx * Sy) / D        
    # b = (Sxx * Sy - Sx * Sxy) / D      
    a = (S(xs, ys) * Sxy(xs, ys) - Sx(xs, ys) * Sy(xs, ys)) / D(xs, ys)
    b = (Sxx(xs, ys) * Sy(xs, ys) - Sx(xs, ys) * Sxy(xs, ys)) / D(xs, ys)

    # Calculate errors for the coefficients according to script
    σ_a = np.sqrt(np.abs(Sxx(xs, ys) / D(xs, ys)))
    σ_b = np.sqrt(np.abs(S(xs, ys) / D(xs, ys)))
    
    # Return as ufloat object
    return ufloat(a, σ_a), ufloat(b, σ_b)


def chi2_value(xs, ys, ab):
    a, b = ab
    sigma_s = np.array([y.std_dev for y in ys])
    return np.sum(((np.array([y.nominal_value for y in ys]) - (a * xs + b)) / sigma_s)**2)

def check_goodness_of_fit_task2(xs, ys, ab):
    return chi2_value(xs, ys, ab) / (len(xs) - 2)


def internal_uncertainty_task2(ms):
    # Calculate internal uncertainty
    return np.sqrt(1 / np.sum(1 / np.array([m.std_dev for m in ms])**2))


def linear_model_function(x, p):
    return p[0] * x + p[1]

############################################################################################################################
##### TASK 3 ###############################################################################################################
############################################################################################################################

# Fit two functions with linear coefficients to the given angular distribution
# First with "ordinary" polynoms and then with Legendre polynoms
# Calculate chi^2 for different polynomial degrees and find the optimal polynomial degree
# Calculate the reduced chi^2 (divided by number of degrees of freedom) and plot the data including the fit

# Define parameters:
# x = array with cos(theta) values
# y = array with measurement values
# sigma = array with corresponding uncertainties for the measurement values (poisson distributed -> sigma = sqrt(N))
# degree = degree of polynom that has to be fitted

def weighted_fit_ordinary_polynoms(x, y, sigma, degree):
    # Weighted fit with ordinary polynoms
    
    # Create matrix A
    # Use "np.vander" to create vandermonde matrix
    # ---> allows to represent the polynom as linear combination of basis functions
    # ---> A = [1, x, x², ..., x^(N-1)]
    # ---> Here N = degree + 1
    A = np.vander(x, degree + 1, increasing=True)
    
    # Create weight matrix W
    # Diagonal matrix in which every diagonal element is w_i = 1 / sigma_i^2
    # ---> Measurement values with smaller sigma (-> less deviation from the mean value) are weighted higher than those that deviate more noticeably
    W = np.diag(1 / sigma**2)
    
    
    # Set up and solve normal equation
    # A.T corresponds to the transposed of the matrix A
    # The @ is the symbol for matrix multiplication in python
    # Calculate M = A^T @ W @ A
    M = A.T @ (W @ A)       
    # Calculate b = A^T @ Q @ y
    b = A.T @ (W @ y)      

    # Use "np.linalg.solve" to find the coefficient c 
    # This function solves linear systems of equations
    # ---> Discussion on github suggests, that scipy.linalg provides additional functionality but np.linalg is faster for small to medium sets of data (numpy function implemented in optimized C code)
    # ---> M * c = b
    coeffs = np.linalg.solve(M, b)
    
    # Use the previously calculated coefficients to determine the fitted y-values
    y_fit = A @ coeffs
    
    # Calculate chi^2
    # Chi^2 = ((y - y_fit)/sigma)^2
    chi2 = np.sum(((y - y_fit) / sigma) ** 2)
    # Calculate degrees of freedom 
    # Degree of freedom = number of measurements - number of parameters
    degree_of_freedom = len(y) - (degree + 1)
    # ---> reduced chi^2 = chi^2 / degree of freedom
    reduced_chi2 = chi2 / degree_of_freedom if degree_of_freedom > 0 else np.nan
    
    return coeffs, y_fit, chi2, reduced_chi2

def weighted_fit_legendre_polynoms(x, y, sigma, degree):
    # Similar procedure, this time with legendre polynoms

    # Create matrix A_legendre
    A_legendre = np.polynomial.legendre.legvander(x, degree)
    
    # Create weight matrix W and calculate coeficcients as done before
    W = np.diag(1 / sigma**2)
    M = A_legendre.T @ (W @ A_legendre)
    b = A_legendre.T @ (W @ y)
    coeffs = np.linalg.solve(M, b)
    
    y_fit = A_legendre @ coeffs
    chi2 = np.sum(((y - y_fit) / sigma) ** 2)
    degree_of_freedom = len(y) - (degree + 1)
    reduced_chi2 = chi2 / degree_of_freedom if degree_of_freedom > 0 else np.nan
    
    return coeffs, y_fit, chi2, reduced_chi2

def plot_all_fits_task3(x, y, sigma, results_ordinary, results_legendre):
    # Create seperate plots for all polynomial orders
    # First all regular polynoms in one plot
    # Then all Legendre polynoms in one plot
    x_dense = np.linspace(-1, 1, 300)
    
    # Plot for regular polynoms
    plt.figure(figsize=(12, 8))
    plt.errorbar(x, y, yerr=sigma, fmt='o', color='black', label='Measurement data', capsize=5, alpha=0.7)
    
    for deg, coeffs, chi2, red_chi2 in results_ordinary:
        A_dense = np.vander(x_dense, deg + 1, increasing=True)
        y_dense = A_dense @ coeffs
        plt.plot(x_dense, y_dense, label=f'degree {deg}, red. χ²={red_chi2:.3f}', linewidth=2)
    
    plt.xlabel('cos(θ)')
    plt.ylabel('N')
    plt.title('Fits with ordinary polynmos of varying order')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("C://Users//svens//OneDrive//Desktop//Python//2_Computational_Physics//Simulation_und_Fit_experimenteller_Daten//A3_ordinary_all.png")
    plt.show()
    
    # Plot for all Legendre polynoms
    plt.figure(figsize=(12, 8))
    plt.errorbar(x, y, yerr=sigma, fmt='o', color='black', label='Measurement data', capsize=5, alpha=0.7)
    
    for deg, coeffs, chi2, red_chi2 in results_legendre:
        A_dense_leg = np.polynomial.legendre.legvander(x_dense, deg)
        y_dense_leg = A_dense_leg @ coeffs
        plt.plot(x_dense, y_dense_leg, label=f'degree {deg}, red. χ²={red_chi2:.3f}', linewidth=2)
    
    plt.xlabel('cos(θ)')
    plt.ylabel('N')
    plt.title('Fits with Legendre poylnoms of varying order')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("C://Users//svens//OneDrive//Desktop//Python//2_Computational_Physics//Simulation_und_Fit_experimenteller_Daten//A3_legendre_all.png")
    plt.show()

# New function to get best polynomial order
def find_best_polynomial_degree_task3(results):
    best_degree = 0
    
    # Look for highest order with reduced Chi² close to 1 (set 0.7 and 1.3 as limits)
    acceptable_fits = [(deg, red_chi2) for deg, _, _, red_chi2 in results if 0.7 < red_chi2 < 1.3]
    
    if acceptable_fits:
        # Choose lowest order with sufficient fit
        best_degree = min(acceptable_fits, key=lambda x: x[0])[0]
    else:
        # If no acceptable fit is found, use the one with reduced Chi² closest to 1
        best_degree = min(results, key=lambda res: abs(res[3] - 1))[0]
    
    return best_degree

# Calculate AIC for statistical evaluation
def calculate_aic_task3(results, n_samples):
    # Calculate AIC (Akaike Information Criterion) for every polynomial order
    # AIC = chi² + 2k with k being the number of parameters
    aic_values = []
    
    for deg, _, chi2, _ in results:
        k = deg + 1  # number of free parameters
        aic = chi2 + 2*k
        aic_values.append((deg, aic))
    
    best_deg_aic = min(aic_values, key=lambda x: x[1])[0]
    
    return best_deg_aic, aic_values

def calculate_parameter_significance(x, y, sigma, coeffs, degree):
    A = np.vander(x, degree + 1, increasing=True)
    W = np.diag(1 / sigma**2)
    
    # Calculate covariance of parameters
    M = A.T @ (W @ A)
    try:
        cov_matrix = np.linalg.inv(M)
        parameter_errors = np.sqrt(np.diag(cov_matrix))
        
        # t-statistic for every parameter
        t_values = np.abs(coeffs) / parameter_errors
        
        # degrees of freedom
        dof = len(y) - (degree + 1)
        
        # p-vales
        p_values = 2 * (1 - stats.t.cdf(t_values, dof))
        
        return parameter_errors, t_values, p_values
    except np.linalg.LinAlgError:
        return None, None, None

def find_statistically_significant_degree(x, y, sigma, results, alpha=0.05):
    # Find highest polynomial order with significance
    significant_degrees = []
    
    for deg, coeffs, chi2, red_chi2 in results:
        if deg == 0:  
            significant_degrees.append(deg)
            continue
            
        param_errors, t_values, p_values = calculate_parameter_significance(x, y, sigma, coeffs, deg)
        
        if param_errors is not None:

            all_significant = np.all(p_values < alpha)
            highest_coeff_significant = p_values[-1] < alpha
            
            if highest_coeff_significant:  
                significant_degrees.append(deg)
    
    # return highest significant order
    return max(significant_degrees) if significant_degrees else 0

def f_test_polynomial_comparison(x, y, sigma, results):
    # f-test to compare "neighbouring" polynomial orders
    f_test_results = []
    
    for i in range(len(results) - 1):
        deg1, coeffs1, chi2_1, _ = results[i]
        deg2, coeffs2, chi2_2, _ = results[i + 1]
        
        # F-statistic
        # F = [(chi2_1 - chi2_2) / (deg2 - deg1)] / [chi2_2 / (n - deg2 - 1)]
        n = len(y)
        numerator = (chi2_1 - chi2_2) / (deg2 - deg1)
        denominator = chi2_2 / (n - deg2 - 1)
        
        if denominator > 0:
            f_stat = numerator / denominator
            df1 = deg2 - deg1
            df2 = n - deg2 - 1
            
            p_value = 1 - stats.f.cdf(f_stat, df1, df2)
            
            f_test_results.append({
                'from_degree': deg1,
                'to_degree': deg2,
                'f_statistic': f_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            })
    
    return f_test_results

def statistical_model_selection(x, y, sigma, results):
    
    # Check parameter significance
    print("\n Check significance of parameter:")
    print("-" * 40)
    
    for deg, coeffs, chi2, red_chi2 in results:
        param_errors, t_values, p_values = calculate_parameter_significance(x, y, sigma, coeffs, deg)
        
        if param_errors is not None:
            print(f"\nGrad {deg}:")
            for i, (coeff, error, t_val, p_val) in enumerate(zip(coeffs, param_errors, t_values, p_values)):
                significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                print(f"  Coeff. x^{i}: {coeff:8.4f} ± {error:8.4f}, t={t_val:6.2f}, p={p_val:8.4f} {significance}")
    
    # 2. F-Test für Modellvergleich
    print("\nf-test:")
    print("-" * 40)
    
    f_test_results = f_test_polynomial_comparison(x, y, sigma, results)
    for result in f_test_results:
        significance = "***" if result['p_value'] < 0.001 else "**" if result['p_value'] < 0.01 else "*" if result['p_value'] < 0.05 else ""
        print(f"Degree {result['from_degree']} → {result['to_degree']}: F={result['f_statistic']:6.2f}, p={result['p_value']:8.4f} {significance}")
    
    sig_degree = find_statistically_significant_degree(x, y, sigma, results)
    
    print("\n Information criteria:")
    print("-" * 40)
    
    n = len(y)
    aic_values = []
    bic_values = []
    
    for deg, _, chi2, _ in results:
        k = deg + 1  # number of parameters
        aic = chi2 + 2*k
        bic = chi2 + k*np.log(n)  # Bayesian Information Criterion
        aic_values.append((deg, aic))
        bic_values.append((deg, bic))
        print(f"Grad {deg}: AIC={aic:8.2f}, BIC={bic:8.2f}")
    
    best_aic = min(aic_values, key=lambda x: x[1])[0]
    best_bic = min(bic_values, key=lambda x: x[1])[0]
    
    print("\n Recommodation:")
    print("-" * 40)
    print(f"statistically highest significant order: {sig_degree}")
    print(f"Best Order according to AIC: {best_aic}")
    print(f"Best Order according to BIC: {best_bic}")
    
    conservative_choice = min(sig_degree, best_bic)
    print(f"Conservative choice: {conservative_choice}")
    
    return {
        'statistically_significant': sig_degree,
        'best_aic': best_aic,
        'best_bic': best_bic,
        'conservative': conservative_choice,
        'f_test_results': f_test_results
    }


############################################################################################################################
##### TASK 4 ###############################################################################################################
############################################################################################################################


# Load data
# I don't know why, but sometimes "//" doesn't work and sometimes it does?!
data = np.loadtxt("C:\\Users\\svens\\OneDrive\\Desktop\\Python\\2_Computational_Physics\\Simulation_und_Fit_experimenteller_Daten\\Agdecay.dat")
delta_t = 5
# First column = measurement times -> multiply by delta_t to receive actual times
times = data[:, 0] * delta_t
# Second column = number of decays ("counts")
Ns = data[:, 1]
# Error for counts is poisson distributed
errsq = Ns

def func(params, t):
    # Define model function according to task
    # N(t) = N1 * exp(-lambda1 * t) + N2 * exp(-lambda2 * t) + N0
    # N0 is the background decay rate
    l1, l2, N1, N2, N0 = params
    return N1 * np.exp(-l1 * t) + N2 * np.exp(-l2 * t) + N0

def chi_squared(params):
    # Calculate chi^2 value to check if the model represents the data well
    f_values = func(params, times)
    return np.sum((Ns - f_values)**2 / errsq)

def chi_squared_gradient(params):
    # Calculate the gradient for chi^2
    l1, l2, N1, N2, N0 = params
    f_values = func(params, times)
    
    # Calculate each partial derivative individually
    df_dl1 = (-times * N1) * np.exp(-l1 * times)
    df_dl2 = (-times * N2) * np.exp(-l2 * times)
    df_dN1 = np.exp(-l1 * times)
    df_dN2 = np.exp(-l2 * times)
    df_dN0 = 1
        
    gradient = -2 * np.array([
        np.sum(df_dl1 * (Ns - f_values) / errsq),
        np.sum(df_dl2 * (Ns - f_values) / errsq),
        np.sum(df_dN1 * (Ns - f_values) / errsq),
        np.sum(df_dN2 * (Ns - f_values) / errsq),
        np.sum(df_dN0 * (Ns - f_values) / errsq)
    ])
        
    return gradient

def print_results(params, method):
    # Print results
    l1, l2, N1, N2, N0 = params
    print("==============================")
    print(f"Fitting data with {method}")
    print("==============================")
    print(f"lambda 1 = {l1:.3e}")
    print(f"lambda 2 = {l2:.3e}")
    print(f"N1      = {N1:.3e}")
    print(f"N2      = {N2:.3e}")
    print(f"N0      = {N0:.3e}")
    print("==============================")
    # Calculate and plot the half life 
    # ln(2) / lambda
    print(f"Half-life 1 = {np.log(2)/l1:.3e} sec")
    print(f"Half-life 2 = {np.log(2)/l2:.3e} sec")
    print("==============================")

def plot_data_fit(params, times, method_name):
    # Plot results
    plt.figure(figsize=(8, 6))
    plt.fill_between(times, 0, Ns, color='brown', alpha=0.5, label='Measurement data')
    plt.plot(times, func(params, times), "r-", label="Fit")
    
    # Calculate chi^2/ndf
    chi2_ndf = chi_squared(params) / (len(Ns) - len(params))

    l1, l2, N1, N2, N0 = params
    
    # Textbox in bottom right corner of the plot containing the parameters
    plt.text(0.95, 0.05, 
             f"$\\chi^2/ndf = {chi2_ndf:.2f}$\n"
             f"$\\lambda_1 = {l1:.2e} \\pm {np.sqrt(errsq[0]):.2e}$\n"
             f"$\\lambda_2 = {l2:.2e} \\pm {np.sqrt(errsq[1]):.2e}$\n"
             f"$N_1 = {N1:.2e}$\n"
             f"$N_2 = {N2:.2e}$\n"
             f"$N_0 = {N0:.2e}$", 
             transform=plt.gca().transAxes, 
             ha='right', va='bottom', fontsize=10,
             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=1'))
    
    plt.xlabel("Time (s)")
    plt.ylabel("Counts")
    plt.legend(loc="upper right")
    plt.title(f"Fit using {method_name}", fontsize=14)
    plt.tight_layout()
    
    # Show plot and save to specified folder
    plt.savefig(f"C:\\Users\\svens\\OneDrive\\Desktop\\Python\\2_Computational_Physics\\Simulation_und_Fit_experimenteller_Daten\\{method_name}.png")
    plt.show()

############################################################################################################################
##### MAIN FUNCTION ########################################################################################################
############################################################################################################################

def main():
    # Measurement data for task 1
    # List of three tuples, each with measurement value and corresponding error
    # (Tuples are, contrary to lists, immutable in python)

    data = [
        (299793, 2.0),
        (299792, 4.5),
        (299782, 25)
    ]

    transformed_data = transform_uncertainties(data)
    weighted_mean, weighted_error = weighted_mean_and_uncertainties(transformed_data)
    internal_error = internal_uncertainty_task1(transformed_data)
    mean_value = weighted_mean
    chi_squared_value = chi(transformed_data, mean_value)
    external_error = external_uncertainty_task1(chi_squared_value, internal_error)
    
    fit_parameter, result, is_good_fit = check_goodness_of_fit_task1(chi_squared_value)

    print("-------------------------------------------")
    print("----- TASK 1 ------------------------------")
    print("-------------------------------------------")
    print(f"Weighted average: {weighted_mean:.2f} km/s")
    print(f"Weighted error: {weighted_error:.2f} km/s")
    print(f"Internal error: {internal_error:.2f} km/s")
    print(f"External error: {external_error:.2f} km/s")
    print(f"Chi^2-value: {chi_squared_value:.2f}")
    print(f"Calculated fit parameter: {fit_parameter:.2f}")

    # if is_good_fit:
    #     print("--> Good fit! (expected fit parameter: 1)")
    # else:
    #     print("--> Bad fit! (expected fit parameter: 1)")

#######################
##### TASK 1 DONE #####
#######################

    print("-------------------------------------------")
    print("----- TASK 2 ------------------------------")
    print("-------------------------------------------")

    U, I, delta_I = load_data_task2() # adjusted to now use the manually inserted values

    # Convert measurement data to ufloat object to enable automatic error propagation
    I_measurements = [ufloat(i, delta_i) for i, delta_i in zip(I, delta_I)]

    # Plot measurement data
    plt.scatter(U, [i.nominal_value for i in I_measurements], label="Data", color='black', alpha=0.5, marker='o')
    plt.xlabel("U [V]")
    plt.ylabel("I [A]")
    plt.title("Fit")

    # Linear regression: I = a * U + b
    a, b = lin_regression(U, I_measurements)

    U_range = np.linspace(min(U), max(U), 100)
    fit_data = np.array([a.nominal_value * u + b.nominal_value for u in U])

    # Print coeffcients
    print(f"Linear Fit: a = {a.nominal_value}, error = {a.std_dev}")
    print(f"a = {a}, b = {b}")
    print(f"R = {1 / a.nominal_value}, I_0 = {b.nominal_value}")
    print(f"Fit goodness: Chi^2/(N-2) = {check_goodness_of_fit_task2(U, I_measurements, (a.nominal_value, b.nominal_value))}")

    # Optimization
    res = minimize(lambda p: np.sum((np.array([i.nominal_value for i in I_measurements]) - linear_model_function(U, p))**2), [0.1, np.mean([i.nominal_value for i in I_measurements][:3])])
    optimized_fit_data = linear_model_function(U, res.x)

    optimized_a, optimized_b = res.x
    I_nom = np.array([i.nominal_value for i in I_measurements])
    sigma_arr = np.array([i.std_dev for i in I_measurements])
    chi2_opt = np.sum(((I_nom - optimized_fit_data) / sigma_arr)**2)
    red_chi2_opt = chi2_opt / (len(U) - 2)

    print("Optimized fit for comparison:")
    print(f"a = {optimized_a:.2e}")
    print(f"b = {optimized_b:.2e}")
    print(f"R = {1/optimized_a:.2e}")
    print(f"I₀ = {optimized_b:.2e}")
    print(f"Fit goodness (optimized fit): Chi²/(N-2) = {red_chi2_opt:.2e}")

    # Additional testing due to faulty values in linear fit
    from scipy.optimize import curve_fit

    def line(x, a, b):
        return a * x + b

    popt, pcov = curve_fit(line, U, [i.nominal_value for i in I_measurements], sigma=[i.std_dev for i in I_measurements], absolute_sigma=True)
    a_scipy, b_scipy = popt
    perr = np.sqrt(np.diag(pcov))
    print("\nSciPy curve_fit for comparison:")
    print(f"a = {a_scipy:.2e} ± {perr[0]:.2e}")
    print(f"b = {b_scipy:.2e} ± {perr[1]:.2e}")
    print(f"R = {1/a_scipy:.2e}")

    fit_data_scipy = line(U, a_scipy, b_scipy)

    # Plots
    plt.plot(U, fit_data, label="Linear Fit", color='red', linestyle='--')
    plt.plot(U, optimized_fit_data, label="Optimized Fit", color='blue', linestyle=':')
    plt.plot(U, fit_data_scipy, label="SciPy Fit", color='green', linestyle='-.')

    # Save plots in corresponding folder
    plt.legend()
    plt.savefig("C://Users//svens//OneDrive//Desktop//Python//2_Computational_Physics//Simulation_und_Fit_experimenteller_Daten//A2.png")
    plt.show()

    # # Check auxilliary functions for problems
    # print("\Auxilliary functions:")
    # print(f"S = {S(U, I_measurements)}")
    # print(f"Sx = {Sx(U, I_measurements)}")
    # print(f"Sy = {Sy(U, I_measurements)}")
    # print(f"Sxx = {Sxx(U, I_measurements)}")
    # print(f"Sxy = {Sxy(U, I_measurements)}")
    # print(f"D = {D(U, I_measurements)}")

#######################
##### TASK 2 DONE #####
#######################

    # print("-------------------------------------------")
    # print("----- TASK 3 ------------------------------")
    # print("-------------------------------------------")

    # # Define data according to task
    # # x = values for cos(theta)
    # x = np.array([-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9])
    # # y = measurement values
    # y = np.array([81, 50, 35, 27, 26, 60, 106, 189, 318, 520])
    # # Calculate uncertainties with sigma = sqrt(N) (poisson distributed)
    # sigma = np.sqrt(y)

    # max_degree = len(x) - 1
    # degrees = range(0, max_degree + 1)

    # # Initialize empty lists for the results
    # results_ordinary = []  
    # results_legendre = []   

    # for deg in degrees:
    #     # Fit with ordinary polynoms
    #     coeffs_ord, y_fit_ord, chi2_ord, red_chi2_ord = weighted_fit_ordinary_polynoms(x, y, sigma, deg)
    #     results_ordinary.append((deg, coeffs_ord, chi2_ord, red_chi2_ord))
        
    #     # Fit with legendre polynoms
    #     coeffs_leg, y_fit_leg, chi2_leg, red_chi2_leg = weighted_fit_legendre_polynoms(x, y, sigma, deg)
    #     results_legendre.append((deg, coeffs_leg, chi2_leg, red_chi2_leg))

    # # Print results
    # print("Results (ordinary polynoms):")
    # for deg, coeffs, chi2, red_chi2 in results_ordinary:
    #     print(f"Degree: {deg:2d}, chi^2 = {chi2:6.4f}, reduced chi^2 = {red_chi2:6.4f}")

    # print("\nResults (legendre polynoms):")
    # for deg, coeffs, chi2, red_chi2 in results_legendre:
    #     print(f"Degree: {deg:2d}, chi^2 = {chi2:6.4f}, reduced chi^2 = {red_chi2:6.4f}")

    # # Create plots for all polynomial orders
    # plot_all_fits_task3(x, y, sigma, results_ordinary, results_legendre)

    # # Find best polynom degree (chi^2 closest to 1)
    # best_degree_ord = min(results_ordinary, key=lambda res: abs(res[3] - 1))[0]
    # best_degree_leg = min(results_legendre, key=lambda res: abs(res[3] - 1))[0]

    # print(f"\nBest polynomial degree (ordinary polynoms): {best_degree_ord}")
    # print(f"Best polynomial degree (legendre polynoms): {best_degree_leg}")

    # # Calculate statistically best polynomial order
    # statistical_best_ord = find_best_polynomial_degree_task3(results_ordinary)
    # statistical_best_leg = find_best_polynomial_degree_task3(results_legendre)
    
    # print(f"\nStatistically best polynomial degree (ordinary polynoms): {statistical_best_ord}")
    # print(f"Statistically best polynomial degree  (legendre polynoms): {statistical_best_leg}")
    
    # # Additional AIC rating
    # aic_best_ord, aic_values_ord = calculate_aic_task3(results_ordinary, len(x))
    # aic_best_leg, aic_values_leg = calculate_aic_task3(results_legendre, len(x))
    
    # print(f"Best polynomial degree according to AIC rating (ordinary polynoms): {aic_best_ord}")
    # print(f"Best polynomial degree according to AIC rating (legendre polynoms): {aic_best_leg}")

    # # Select best fit for ordinary polynoms
    # best_deg = best_degree_ord
    # coeffs_best, y_fit_best, chi2_best, red_chi2_best = weighted_fit_ordinary_polynoms(x, y, sigma, best_deg)
    # x_dense = np.linspace(-1, 1, 300)
    # A_dense = np.vander(x_dense, best_deg + 1, increasing=True)
    # y_dense_manual = A_dense @ coeffs_best

    # # Select best fit for legendre polynoms
    # coeffs_best_leg, y_fit_best_leg, chi2_best_leg, red_chi2_best_leg = weighted_fit_legendre_polynoms(x, y, sigma, best_degree_leg)
    # A_dense_leg = np.polynomial.legendre.legvander(x_dense, best_degree_leg)
    # y_dense_manual_leg = A_dense_leg @ coeffs_best_leg

    # # Compare with built in fitting methods
    # # Polynomial fit
    # poly_fit = Polynomial.fit(x, y, best_deg, w=1/sigma)
    # poly_fit_standard = poly_fit.convert()
    # y_dense_builtin = poly_fit_standard(x_dense)

    # # Legendre fit
    # leg_fit = Legendre.fit(x, y, best_degree_leg, w=1/sigma)
    # y_dense_legendre_builtin = leg_fit(x_dense)

    # # Plots
    # plt.figure(figsize=(8, 6))
    # # Plot data with error indicator
    # plt.errorbar(x, y, yerr=sigma, fmt='o', color='black', label='Measurement data', capsize=5)

    # # Manual fit with ordinary polynoms
    # plt.plot(x_dense, y_dense_manual, label=f'Manual Fit Ordinary Polynoms (degree {best_deg})', lw=2)
    # # Manual fit with legendre polynoms
    # plt.plot(x_dense, y_dense_manual_leg, '--', label=f'Manual Fit Legendre Polynoms (degree {best_degree_leg})', lw=2)
    # # Automatic fit with polynomial.fit
    # plt.plot(x_dense, y_dense_builtin, ':', label=f'Polynomial.fit (degree {best_deg})', lw=2)
    # # Automatic fit with legendre.fit
    # plt.plot(x_dense, y_dense_legendre_builtin, '-.', label=f'Legendre.fit (degree {best_degree_leg})', lw=2)

    # plt.xlabel('cos(θ)')
    # plt.ylabel('N')
    # plt.title('Comparison of polynomial fits')
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig("C://Users//svens//OneDrive//Desktop//Python//2_Computational_Physics//Simulation_und_Fit_experimenteller_Daten//A3.png")
    # plt.show()

    print("-------------------------------------------")
    print("----- TASK 3 (ADJUSTED) -------------------")
    print("-------------------------------------------")

    # Define data
    x = np.array([-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9])
    y = np.array([81, 50, 35, 27, 26, 60, 106, 189, 318, 520])
    sigma = np.sqrt(y)

    max_degree = len(x) - 1
    degrees = range(0, max_degree + 1)

    # Fitting
    results_ordinary = []  
    results_legendre = []   

    for deg in degrees:
        # Ordinary polynoms
        coeffs_ord, y_fit_ord, chi2_ord, red_chi2_ord = weighted_fit_ordinary_polynoms(x, y, sigma, deg)
        results_ordinary.append((deg, coeffs_ord, chi2_ord, red_chi2_ord))
        
        # Legendre polynoms  
        coeffs_leg, y_fit_leg, chi2_leg, red_chi2_leg = weighted_fit_legendre_polynoms(x, y, sigma, deg)
        results_legendre.append((deg, coeffs_leg, chi2_leg, red_chi2_leg))

    # Show resutls
    print("RESULTS:")
    print("Results (ordinary polynoms):")
    for deg, coeffs, chi2, red_chi2 in results_ordinary:
        print(f"Degree: {deg:2d}, chi^2 = {chi2:6.4f}, reduced chi^2 = {red_chi2:6.4f}")

    print("\nResults (legendre polynoms):")
    for deg, coeffs, chi2, red_chi2 in results_legendre:
        print(f"Degree: {deg:2d}, chi^2 = {chi2:6.4f}, reduced chi^2 = {red_chi2:6.4f}")

    # Statistical analysis
    print("\n" + "="*80)
    print("Statistical analysis")
    print("="*80)
    
    print("\nOrdinary polynoms:")
    ord_stats = statistical_model_selection(x, y, sigma, results_ordinary)
    
    print("\nLegendre polynoms:")
    leg_stats = statistical_model_selection(x, y, sigma, results_legendre)
    
    print("\n" + "="*80)
    print("Recommodation for best polynomial degree")
    print("="*80)
    print(f"Ordinary polynoms - Statistically best polynomial order: {ord_stats['statistically_significant']}")
    print(f"Legendre polynoms - Statistically best polynomial order: {leg_stats['statistically_significant']}")


#######################
##### TASK 3 DONE #####
#######################

    print("-------------------------------------------")
    print("----- TASK 4 ------------------------------")
    print("-------------------------------------------")

    # Initial guesses
    initial_guess = np.array([1e-1, 1e-2, 1000, 100, 10])

    # Optimization with simplex method
    res_simplex = minimize(chi_squared, initial_guess, method='Nelder-Mead')

    # Optimization with conjugate gradient method
    res_cg = fmin_cg(chi_squared, initial_guess, fprime=chi_squared_gradient)

    # Print results
    print("Simplex Optimization Results:")
    print(res_simplex)
    print_results(res_simplex.x, "Simplex method")

    print("Conjugate Gradient Optimization Results:")
    print(res_cg)
    print_results(res_cg, "Conjugate Gradient method")

    # Plot results with both methods
    plot_data_fit(res_simplex.x, times, "Simplex method")
    plot_data_fit(res_cg, times, "Conjugate Gradient method")

#######################
##### TASK 4 DONE #####
#######################


if __name__ == "__main__":
    main()




