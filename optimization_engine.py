# optimization_engine.py
import time
from typing import List, Tuple, Dict

from customer_data import Customer, all_customers
from constants import (
    DRIVER_START_TIME, DRIVER_FINISH_TIME,
    BUFFER_MINUTES, ALPHA, EDGE_CONSISTENCY_BONUS
)
from route_construction import insertion_heuristic
from route_optimization import tabu_enhanced_two_opt
from route_enhancement import attempt_additional_insertions
from satisfaction_metrics import calculate_customer_satisfaction, calculate_working_time

# This will store the optimization results to avoid recomputation
_cached_results = None


def get_optimization_results(force_recompute=False):
    """
    Single source of truth for optimization results.
    Returns a dictionary with all route data for all days.

    Args:
        force_recompute: Force recalculation even if cached results exist

    Returns:
        Dictionary with all optimization results
    """
    global _cached_results

    # Return cached results if available and not forcing recomputation
    if _cached_results is not None and not force_recompute:
        return _cached_results

    print("Generating route data...")
    print(f"Using alpha = {ALPHA} (customer weight: {ALPHA}, driver weight: {1.0 - ALPHA})")

    # Run for 5 days (Mon-Fri)
    all_routes = []
    all_arrival_times = []
    all_objective_values = []
    all_customer_satisfactions = []
    all_driver_satisfactions = []
    all_working_times = []
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    day_indices = list(range(5))
    all_unvisited_customers = []

    # Execution timings
    all_execution_times = []
    all_construction_times = []
    all_optimization_times = []
    all_enhancement_times = []

    for day_idx in range(5):
        print(f"\n----- Processing {days[day_idx]} -----")

        # Option B: copy Monday’s route for perfect consistency when alpha=0.0
        if ALPHA == 0.0 and day_idx > 0:
            print("Alpha=0.0: copying Monday’s route to maintain perfect consistency")
            all_routes.append(all_routes[0])
            all_arrival_times.append(all_arrival_times[0])
            all_objective_values.append(all_objective_values[0])
            all_customer_satisfactions.append(all_customer_satisfactions[0])
            all_driver_satisfactions.append(all_driver_satisfactions[0])
            all_working_times.append(all_working_times[0])
            all_unvisited_customers.append(all_unvisited_customers[0])
            # for timing arrays, just record zero
            all_execution_times.append(0.0)
            all_construction_times.append(0.0)
            all_optimization_times.append(0.0)
            all_enhancement_times.append(0.0)
            continue

        # Track execution time
        start_time_total = time.time()

        # Previous routes and working times for driver satisfaction calculation
        previous_routes = all_routes.copy()
        previous_working_times = all_working_times.copy()

        # Step 1: Create initial route using insertion heuristic
        print("Generating initial route using insertion heuristic...")
        start_time_construction = time.time()
        initial_route, initial_cost, initial_arrival_times, initial_cust_sat, initial_driver_sat = insertion_heuristic(
            customers=all_customers,
            day_index=day_idx,
            previous_routes=previous_routes,
            previous_working_times=previous_working_times,
            alpha=ALPHA,
            edge_consistency_bonus=EDGE_CONSISTENCY_BONUS
        )
        construction_time = time.time() - start_time_construction

        # Identify unvisited customers from the initial route
        visited_customer_ids = {customer.id for customer in initial_route}
        unvisited_customers = [customer for customer in all_customers if
                               customer.id not in visited_customer_ids and customer.id != 0]  # Exclude depot

        # Store unvisited customers for this day
        all_unvisited_customers.append(unvisited_customers)

        print(
            f"Initial route contains {len(initial_route) - 2} customers. {len(unvisited_customers)} customers unvisited.")

        # Step 2: Improve route using tabu-enhanced 2-opt
        print("Improving route using tabu-enhanced 2-opt local search...")
        start_time_optimization = time.time()
        improved_route, improved_cost, improved_arrival_times, improved_cust_sat, improved_driver_sat = tabu_enhanced_two_opt(
            route=initial_route,
            day_index=day_idx,
            previous_routes=previous_routes,
            previous_working_times=previous_working_times,
            alpha=ALPHA,
            edge_consistency_bonus=EDGE_CONSISTENCY_BONUS
        )
        optimization_time = time.time() - start_time_optimization

        # Log optimization results for analysis
        if improved_cost < initial_cost:
            improvement_percent = (initial_cost - improved_cost) / initial_cost * 100
            print(
                f"Tabu search improved cost by {improvement_percent:.2f}% (from {initial_cost:.4f} to {improved_cost:.4f})")
        else:
            print(f"No improvement from tabu search (initial: {initial_cost:.4f}, after tabu: {improved_cost:.4f})")

        # Step 3: Attempt to insert additional unvisited customers
        print("Attempting to insert additional customers post-optimization...")
        start_time_enhancement = time.time()
        final_route, final_cost, arrival_times, customer_sat, driver_sat = attempt_additional_insertions(
            route=improved_route,
            unvisited_customers=unvisited_customers,
            day_index=day_idx,
            previous_routes=previous_routes,
            previous_working_times=previous_working_times,
            alpha=ALPHA,
            edge_consistency_bonus=EDGE_CONSISTENCY_BONUS
        )
        enhancement_time = time.time() - start_time_enhancement

        # Calculate working time for this route
        working_time = calculate_working_time(final_route)

        # Calculate total execution time
        execution_time = time.time() - start_time_total

        # Store results
        all_routes.append(final_route)
        all_objective_values.append(final_cost)
        all_arrival_times.append(arrival_times)
        all_customer_satisfactions.append(customer_sat)
        all_driver_satisfactions.append(driver_sat)
        all_working_times.append(working_time)

        # Store timing info
        all_execution_times.append(execution_time)
        all_construction_times.append(construction_time)
        all_optimization_times.append(optimization_time)
        all_enhancement_times.append(enhancement_time)

        print(f"  Day {day_idx + 1} complete. Route has {len(final_route) - 2} customers. Cost: {final_cost:.4f}")
        print(f"  Execution time: {execution_time:.2f}s (Construction: {construction_time:.2f}s, "
              f"Optimization: {optimization_time:.2f}s, Enhancement: {enhancement_time:.2f}s)")

    # Store comprehensive results
    _cached_results = {
        'routes': all_routes,
        'arrival_times': all_arrival_times,
        'objective_values': all_objective_values,
        'customer_satisfactions': all_customer_satisfactions,
        'driver_satisfactions': all_driver_satisfactions,
        'working_times': all_working_times,
        'days': days,
        'day_indices': day_indices,
        'unvisited_customers': all_unvisited_customers,
        'execution_times': all_execution_times,
        'construction_times': all_construction_times,
        'optimization_times': all_optimization_times,
        'enhancement_times': all_enhancement_times
    }

    return _cached_results


# Function to clear the cache if needed
def clear_optimization_cache():
    global _cached_results
    _cached_results = None