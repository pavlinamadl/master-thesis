import math
import random
from typing import List, Tuple, Set

from customer_data import Customer
from constants import (
    DRIVER_START_TIME, DRIVER_FINISH_TIME,
    BUFFER_MINUTES,
    ALPHA,  # Using ALPHA instead of W_CUSTOMER and W_DRIVER
    EDGE_CONSISTENCY_BONUS,
    MAX_TABU_ITERATIONS,
    TABU_DIVERSIFICATION_THRESHOLD,
    TABU_ASPIRATION_COEF,
    TABU_INITIAL_TENURE_FACTOR
)

from time_utils import estimate_arrival_times, calculate_customer_satisfaction
from satisfaction_metrics import calculate_driver_satisfaction


def diversify_route(route: List[Customer]) -> List[Customer]:
    """
    Diversify the route by performing a series of random 2-opt moves.
    This helps escape local optima during the tabu search.

    Note: This preserves all customers (just changes their order).
    """
    if len(route) <= 3:
        return route.copy()

    diversified_route = route.copy()

    # Number of random 2-opt moves to perform
    num_moves = min(5, len(route) // 4)

    for _ in range(num_moves):
        # Select random indices, excluding depot
        i = random.randint(1, len(diversified_route) - 3)
        j = random.randint(i + 1, len(diversified_route) - 2)

        # Perform 2-opt swap (reverse segment between i and j)
        diversified_route[i:j + 1] = reversed(diversified_route[i:j + 1])

    return diversified_route


def tabu_enhanced_two_opt(
        route: List[Customer],
        day_index: int,
        previous_routes: List[List[Customer]],
        previous_working_times: List[float],
        alpha: float = ALPHA,  # Changed parameter name from w_cust
        edge_consistency_bonus: float = EDGE_CONSISTENCY_BONUS,
        driver_start_time: float = DRIVER_START_TIME,
        buffer_minutes: float = BUFFER_MINUTES,
        max_iterations: int = MAX_TABU_ITERATIONS,
        diversification_threshold: int = TABU_DIVERSIFICATION_THRESHOLD
) -> Tuple[List[Customer], float, List[float], float, float]:
    """
    Improves a route using 2-opt local search algorithm enhanced with tabu search,
    optimizing for both customer satisfaction and driver satisfaction.

    This preserves all customers in the route (must-serve constraint is maintained).
    For alpha=0.0, day 0 (Monday) still considers customer satisfaction but days 1-4
    prioritize consistency with previous day's route.

    Uses a dynamic tabu tenure based on the square root of the number of customers in the route.

    Args:
        route: Initial route to improve
        day_index: Current day index (0-4 for Monday-Friday)
        previous_routes: List of routes from previous days
        previous_working_times: List of working times from previous days
        alpha: Weight for customer satisfaction (0-1), driver weight is (1-alpha)
        edge_consistency_bonus: Bonus applied for reusing edges from previous routes
        driver_start_time: Start time for the driver's day
        buffer_minutes: Buffer time for calculating customer satisfaction
        max_iterations: Maximum iterations for tabu search
        diversification_threshold: Iterations without improvement before diversification

    Returns:
        Tuple containing:
        - The improved route
        - The objective value
        - The arrival times
        - Customer satisfaction component
        - Driver satisfaction component
    """
    print(f"Running tabu enhanced 2-opt with alpha={alpha}, edge_consistency_bonus={edge_consistency_bonus}")

    # Special case for alpha=0.0 on the first day (Monday)
    # We use alpha=0.5 temporarily to ensure customer time windows are considered
    effective_alpha = alpha
    if alpha == 0.0 and day_index == 0:
        print("Special case: alpha=0.0 on first day, using alpha=0.5 temporarily to consider time windows")
        effective_alpha = 0.5  # Use balanced weight for Day 0 only

    if len(route) <= 3:  # Not enough nodes for 2-opt
        final_arrival_times = estimate_arrival_times(route, driver_start_time)

        # Calculate satisfaction components for minimal route
        customer_satisfaction = 1.0  # Assume perfect for minimal/empty route

        driver_satisfaction, _, _ = calculate_driver_satisfaction(
            route, day_index, previous_routes, previous_working_times
        )

        # Calculate total cost using actual alpha (not effective_alpha) for consistency in return values
        total_cost = (1.0 - customer_satisfaction) * alpha + (1.0 - driver_satisfaction) * (1.0 - alpha)

        return route, total_cost, final_arrival_times, customer_satisfaction, driver_satisfaction

    # Calculate initial arrival times and costs
    best_route = route.copy()
    current_route = route.copy()

    # Calculate arrival times for the initial route
    best_arrival_times = estimate_arrival_times(best_route, driver_start_time)

    # Calculate number of customers (excluding depot)
    num_customers = len(best_route) - 2

    # Calculate dynamic tabu tenure based on the square root of number of customers
    tabu_tenure = int(math.sqrt(num_customers) * TABU_INITIAL_TENURE_FACTOR) + 1

    print(f"Dynamic tabu tenure: {tabu_tenure} (based on {num_customers} customers)")

    # Calculate initial customer satisfaction
    best_customer_satisfaction = 0.0

    if num_customers > 0:
        total_customer_sat = 0.0
        for i in range(1, len(best_route) - 1):  # Skip depot
            customer = best_route[i]
            arrival_time = best_arrival_times[i]
            satisfaction = calculate_customer_satisfaction(
                arrival_time,
                customer.time_windows[day_index],
                buffer_minutes
            )
            total_customer_sat += satisfaction
        best_customer_satisfaction = total_customer_sat / num_customers
    else:
        best_customer_satisfaction = 1.0

    # Calculate initial driver satisfaction
    best_driver_satisfaction, _, _ = calculate_driver_satisfaction(
        best_route, day_index, previous_routes, previous_working_times
    )

    # Calculate initial best cost using effective_alpha (for day 0 with alpha=0)
    if day_index == 0:
        best_cost = (1.0 - best_customer_satisfaction) * effective_alpha + (1.0 - best_driver_satisfaction) * (
                    1.0 - effective_alpha)
    else:
        best_cost = (1.0 - best_customer_satisfaction) * alpha + (1.0 - best_driver_satisfaction) * (1.0 - alpha)

    current_cost = best_cost

    # Initialize tabu list and counters
    tabu_list = {}
    iteration = 0
    iterations_without_improvement = 0

    # Identify must-serve customers to ensure they remain in the route
    must_serve_ids = {c.id for c in route if c.must_serve}

    while iteration < max_iterations:
        iteration += 1
        best_neighbor_cost = float('inf')
        best_move = None
        best_neighbor_route = None
        best_neighbor_arrival_times = None
        best_neighbor_customer_satisfaction = 0.0
        best_neighbor_driver_satisfaction = 0.0

        # Check if we should apply diversification
        if iterations_without_improvement >= diversification_threshold:
            # Apply diversification by performing a random series of 2-opt moves
            diversified_route = diversify_route(current_route)

            # Check if the diversified route is feasible
            diversified_arrival_times = estimate_arrival_times(diversified_route, driver_start_time)

            # We accept the diversified route even if it exceeds working hours
            # since must-serve customers take priority
            current_route = diversified_route
            iterations_without_improvement = 0

            # If the route structure changes, recalculate the tabu tenure
            new_num_customers = len(current_route) - 2
            if new_num_customers != num_customers:
                num_customers = new_num_customers
                tabu_tenure = int(math.sqrt(num_customers)) + 1
                print(f"Updated tabu tenure after diversification: {tabu_tenure}")

        # Explore neighborhood (all 2-opt swaps)
        for i in range(1, len(current_route) - 2):
            for j in range(i + 1, len(current_route) - 1):
                if j - i == 1:
                    continue  # Skip adjacent edges

                # Create new route with 2-opt swap
                neighbor_route = current_route.copy()
                # Reverse the segment between i and j
                neighbor_route[i:j + 1] = reversed(current_route[i:j + 1])

                # Calculate new arrival times
                neighbor_arrival_times = estimate_arrival_times(neighbor_route, driver_start_time)

                # Skip if the route finishes after the end of working day,
                # unless it includes must-serve customers
                contain_must_serve = any(c.must_serve for c in neighbor_route[i:j + 1])
                if neighbor_arrival_times[-1] > DRIVER_FINISH_TIME and not contain_must_serve:
                    continue

                # Calculate new customer satisfaction
                neighbor_total_customer_sat = 0.0
                for k in range(1, len(neighbor_route) - 1):  # Skip depot
                    customer = neighbor_route[k]
                    arrival_time = neighbor_arrival_times[k]
                    satisfaction = calculate_customer_satisfaction(
                        arrival_time,
                        customer.time_windows[day_index],
                        buffer_minutes
                    )
                    neighbor_total_customer_sat += satisfaction

                neighbor_customer_satisfaction = neighbor_total_customer_sat / num_customers if num_customers > 0 else 1.0

                # Calculate new driver satisfaction
                neighbor_driver_satisfaction, _, _ = calculate_driver_satisfaction(
                    neighbor_route, day_index, previous_routes, previous_working_times
                )

                # Calculate new total cost with edge consistency
                if day_index > 0:
                    # Extract edges from previous day's route
                    previous_edges = set()
                    if previous_routes:
                        prev_route = previous_routes[-1]  # Get most recent route
                        for idx in range(len(prev_route) - 1):
                            previous_edges.add((prev_route[idx].id, prev_route[idx + 1].id))

                    # Check how many edges are consistent with previous day's route
                    consistent_edges = 0
                    for idx in range(len(neighbor_route) - 1):
                        if (neighbor_route[idx].id, neighbor_route[idx + 1].id) in previous_edges:
                            consistent_edges += 1

                    # MODIFIED: Use proportional edge consistency bonus
                    edge_consistency_bonus_weighted = consistent_edges * edge_consistency_bonus * (1.0 - alpha)
                    neighbor_cost = (1.0 - neighbor_customer_satisfaction) * alpha + \
                                    (1.0 - neighbor_driver_satisfaction) * (1.0 - alpha) - \
                                    edge_consistency_bonus_weighted
                else:
                    # For Monday, use effective_alpha to ensure time windows are considered
                    neighbor_cost = (1.0 - neighbor_customer_satisfaction) * effective_alpha + \
                                    (1.0 - neighbor_driver_satisfaction) * (1.0 - effective_alpha)

                move = (i, j)

                # Check tabu condition with aspiration criterion
                is_tabu = move in tabu_list
                # MODIFIED: Use a slightly more permissive aspiration criterion
                aspiration_threshold = best_cost * TABU_ASPIRATION_COEF
                is_aspiration = neighbor_cost < aspiration_threshold

                # Accept the move if it's not tabu or meets aspiration criteria
                if (not is_tabu or is_aspiration) and neighbor_cost < best_neighbor_cost:
                    best_neighbor_cost = neighbor_cost
                    best_move = move
                    best_neighbor_route = neighbor_route.copy()
                    best_neighbor_arrival_times = neighbor_arrival_times.copy()
                    best_neighbor_customer_satisfaction = neighbor_customer_satisfaction
                    best_neighbor_driver_satisfaction = neighbor_driver_satisfaction

        if best_neighbor_route is None:
            print(f"No improvement found after {iteration} iterations")
            break  # No improvement found or no feasible move

        # Verify all must-serve customers are still in the route
        neighbor_customer_ids = {c.id for c in best_neighbor_route}
        if not must_serve_ids.issubset(neighbor_customer_ids):
            print("ERROR: Must-serve customers would be lost in this move!")
            print(f"Missing: {must_serve_ids - neighbor_customer_ids}")
            break  # Stop optimization to prevent losing must-serve customers

        # Update current solution with best neighbor
        current_route = best_neighbor_route
        current_cost = best_neighbor_cost

        # Update global best if improvement found
        if best_neighbor_cost < best_cost:
            best_route = best_neighbor_route.copy()
            best_cost = best_neighbor_cost
            best_arrival_times = best_neighbor_arrival_times.copy()
            best_customer_satisfaction = best_neighbor_customer_satisfaction
            best_driver_satisfaction = best_neighbor_driver_satisfaction
            iterations_without_improvement = 0
            print(f"Improvement found at iteration {iteration}: {best_cost:.4f} → {best_neighbor_cost:.4f}")
        else:
            iterations_without_improvement += 1
            if iterations_without_improvement % 10 == 0:
                print(f"No improvement for {iterations_without_improvement} iterations")

        # Update tabu list (reduce tabu tenure for existing moves)
        moves_to_remove = []
        for move in tabu_list:
            tabu_list[move] -= 1
            if tabu_list[move] <= 0:
                moves_to_remove.append(move)

        # Remove expired tabu moves
        for move in moves_to_remove:
            del tabu_list[move]

        # Add new tabu move (both forward and reverse moves)
        tabu_list[best_move] = tabu_tenure
        # Also add the inverse move to tabu list (j, i)
        inverse_move = (best_move[1], best_move[0])
        tabu_list[inverse_move] = tabu_tenure

    # Verify that all must-serve customers are in the final route
    final_customer_ids = {c.id for c in best_route}
    if not must_serve_ids.issubset(final_customer_ids):
        print("WARNING: Must-serve customers would be lost in the optimized route!")
        print(f"Missing: {must_serve_ids - final_customer_ids}")
        print("Reverting to the original route to preserve must-serve customers.")
        return route, current_cost, estimate_arrival_times(route,
                                                           driver_start_time), best_customer_satisfaction, best_driver_satisfaction

    print(f"Tabu search completed after {iteration} iterations. Final cost: {best_cost:.4f}")

    # Calculate the actual cost with the original alpha for return value consistency
    actual_cost = (1.0 - best_customer_satisfaction) * alpha + (1.0 - best_driver_satisfaction) * (1.0 - alpha)

    return best_route, actual_cost, best_arrival_times, best_customer_satisfaction, best_driver_satisfaction