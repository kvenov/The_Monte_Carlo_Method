# This is will be the file where the algorithm will be developed.
import numpy as np
import scipy as sp


def sample_customers_normal(mean, sd, size):
    """
    1) Samples the customers from a Normal distribution, by ensuring only whole numbers and no negative values!
    2) Rounding the samples to integers.
    3) Ensuring no-negative numbers.
    4) Returns the generated, processed samples.
    """
    samples = np.random.normal(mean, sd, size)
    
    # Here we rounds the numbers to the nearest whole number
    samples = np.rint(samples).astype(int)
    
    # Ensuring no negative values 
    samples = np.maximum(samples, 0)
    
    return samples


def sample_costs_triangular(min, mode, max, size):
    """
    1) Samples the variables costs from a triangular distribution.
    2) Returning the samples.
    """
    
    return np.random.triangular(min, mode, max, size)


def sample_correlated_variables(mean_N, sd_N,
                                cost_min, cost_mode, cost_max,
                                rho, size):
    """
    1) Generates CORRELATED random samples for:
    - N: number of customers (Normal distribution)
    - Cv: variable cost per order (Triangular distribution)

    Parameters:
    mean_N, sd_N : int, int
        parameters for customer distribution - mean customers, standard deviation
    cost_min, cost_mode, cost_max : float, float, float
        triangular cost parameters
    rho : float
        correlation coefficient between N and Cv
    size : int
        number of samples(The size of the Monte Carlo simulations)

    Returns:
    N  : array of customer counts
    Cv : array of variable costs
    """

    # Correlation matrix defines how variables move together
    # Must be symmetric and positive definite
    corr_matrix = np.array([
        [1, rho],
        [rho, 1]
    ])

    # Cholesky decomposition factorizes the correlation matrix: corr_matrix = L * L^T
    # This allows us to transform independent normal variables into correlated ones.
    L = np.linalg.cholesky(corr_matrix)

    # Z contains independent standard normal variables
    # Shape: (2, size) -> 2 variables (N and Cv), each with 'size' samples
    Z = np.random.normal(0, 1, size=(2, size))

    # Multiplying by L introduces correlation: correlated_Z = L @ Z
    # Now: corr(correlated_Z[0], correlated_Z[1]) ≈ rho
    correlated_Z = L @ Z

    # Using the standard normal CDF
    # This maps: Normal → Uniform(0,1)
    U = sp.stats.norm.cdf(correlated_Z)


    # Inverse transform sampling
    # This converts uniform samples into Normal(mean_N, sd_N)
    N = sp.stats.norm.ppf(U[0], loc=mean_N, scale=sd_N)

    # Ensure integer customers
    N = np.rint(N).astype(int)

    # No negative customers
    N = np.maximum(N, 0)

    # We use inverse transform sampling for triangular distribution.
    # The triangular distribution has a piecewise inverse CDF.
    c = (cost_mode - cost_min) / (cost_max - cost_min)

    U_cost = U[1]
    Cv = np.zeros(size)

    # Left side of triangle
    left = U_cost < c
    Cv[left] = cost_min + np.sqrt(
        U_cost[left] * (cost_max - cost_min) * (cost_mode - cost_min)
    )

    # Right side of triangle
    right = ~left
    Cv[right] = cost_max - np.sqrt(
        (1 - U_cost[right]) * (cost_max - cost_min) * (cost_max - cost_mode)
    )

    return N, Cv


def simulate_profit(customers, price, cost_per_order, fixed_cost):
    """
    1) Calculates the profit, by the formula:
        customers * (price - cost_per_order) - fixed_cost.
    2) Returns the profit.
    """
    
    # First we multiply the amount of customers with the prive per order
    # Second, we are calculating the variable costs per order for all of the customers
    # Then, extracting the costs from the revenue
    revenue = customers * price
    total_var_cost = customers * cost_per_order
    profit = revenue - (total_var_cost + fixed_cost)
    
    return profit


def run_monte_carlo_simulation(simulations_size, 
                               mean_customers, sd_customers, 
                               cost_min, cost_mode, cost_max, 
                               price, fixed_cost,
                               rho=0.0):
    """
    1) Performs a Monte Carlo Simulation
    2) Generates Random Samples based on rho parameter
        - the rho parameter allows switching between:
            - Independent case (rho = 0)
            - Correlated case (rho != 0)
    3) Calculating the base model using the samples
    4) Retuns the caclulated result
    
    """

    # If rho = 0 -> fallback to original independent sampling
    if rho == 0:
        N_samples = sample_customers_normal(mean_customers, sd_customers, simulations_size)
        Cv_samples = sample_costs_triangular(cost_min, cost_mode, cost_max, simulations_size)
    else:
        # Useing correlated sampling
        N_samples, Cv_samples = sample_correlated_variables(
            mean_customers, sd_customers,
            cost_min, cost_mode, cost_max,
            rho, simulations_size
        )

    # Computing profit
    profits = simulate_profit(N_samples, price, Cv_samples, fixed_cost)

    return profits
