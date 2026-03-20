# This is will be the file where the algorithm will be developed.
import numpy as np

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
                               price, fixed_cost):
    """
    1) Performs a Monte Carlo Simulation
    2) Generates Random Samples
    3) Calculating the base model using the samples
    4) Retuns the caclulated result
    """
    
    N_samples = sample_customers_normal(mean_customers, sd_customers, simulations_size)
    Cv_samples = sample_costs_triangular(cost_min, cost_mode, cost_max, simulations_size)
    
    # Calculating the profit, by appling the generated samples on the base model
    profits = simulate_profit(N_samples, price, Cv_samples, fixed_cost)
    
    return profits

# This is a simple analyzing class, which will be used for a collection of statistics.
# class StatisticsAnalyzer:
#     @staticmethod
#     def get_stats(profits):
#         return {
#             "mean": np.mean(profits),
#             "std_dev": np.std(profits),
#             "prob_loss": np.mean(profits < 0),
#             "VaR_5": np.percentile(profits, 5),
#             "VaR_1": np.percentile(profits, 1)
#         }