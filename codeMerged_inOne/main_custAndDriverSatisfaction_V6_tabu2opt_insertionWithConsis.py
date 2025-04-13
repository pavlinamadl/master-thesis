import math
import random
import numpy as np
from typing import NamedTuple, List, Tuple, Dict, Set
import time

# Import customer data from the customer_data.py file
from customer_data import TimeWindow, Customer, all_customers

# Import constants - note SERVICE_TIME_MINUTES is no longer used
# as it's now a property of each customer
from constants import (
    DRIVER_START_TIME, DRIVER_FINISH_TIME,
    BUFFER_MINUTES,
    LUNCH_BREAK_START, LUNCH_BREAK_END,
    SATISFACTION_STEEP_THRESHOLD,
    SPEED_METERS_PER_MINUTE, W_CUSTOMER, W_DRIVER,
    IDEAL_WORKING_HOURS, MAX_WORKING_HOURS,
    WORK_TIME_WEIGHT, ROUTE_CONSISTENCY_WEIGHT,
    MAX_TABU_ITERATIONS, TABU_DIVERSIFICATION_THRESHOLD,
    EDGE_CONSISTENCY_BONUS
)


# Modified functions to use customer-specific service times

def estimate_arrival_times(
        route: List[Customer],
        driver_start_time: float = DRIVER_START_TIME
) -> List[float]:
    """
    Estimates arrival times at each customer in the route.
    Accounts for lunch break and customer-specific service times.
    """
    if not route:
        return []

    arrival_times = [driver_start_time]  # Start time at depot
    current_time = driver_start_time

    for i in range(1, len(route)):
        # Travel time from previous to current customer
        dist = distance(route[i - 1], route[i])
        travel_time = dist / SPEED_METERS_PER_MINUTE

        # Update current time with travel time
        current_time += travel_time

        # Check for lunch break
        current_time = adjust_for_lunch_break(current_time)

        # Add to arrival times list
        arrival_times.append(current_time)

        # Add service time if not at the end depot
        if i < len(route) - 1:
            # Use the customer-specific service time
            service_time = route[i].service_time
            current_time += service_time

    return arrival_times


def insertion_heuristic(
        customers: List[Customer],
        day_index: int,
        previous_routes: List[List[Customer]],
        previous_working_times: List[float],
        w_cust: float = W_CUSTOMER,
        w_driver: float = W_DRIVER,
        edge_consistency_bonus: float = EDGE_CONSISTENCY_BONUS,  # 0 is minimal, >0.5 is very strong
        driver_start_time: float = DRIVER_START_TIME,
        driver_finish_time: float = DRIVER_FINISH_TIME,
        buffer_minutes: float = BUFFER_MINUTES
) -> Tuple[List[Customer], float, List[float], float, float]:
    """
    Constructs a route using an insertion heuristic that considers both
    customer satisfaction and driver satisfaction, with additional consideration
    for edge consistency with previous routes.

    Args:
        customers: List of all customers including depot
        day_index: Current day index (0-4 for Monday-Friday)
        previous_routes: List of routes from previous days
        previous_working_times: List of working times from previous days
        w_cust: Weight for customer satisfaction in objective function
        w_driver: Weight for driver satisfaction in objective function
        edge_consistency_bonus: Bonus applied to driver satisfaction for reusing edges
        driver_start_time: Start time for the driver's day
        driver_finish_time: End time for the driver's day
        buffer_minutes: Buffer time for calculating customer satisfaction

    Returns:
        Tuple containing:
        - Constructed route
        - Total cost (objective value)
        - List of arrival times
        - Customer satisfaction component
        - Driver satisfaction component
    """
    depot = customers[0]
    route = [depot, depot]  # Start and end at depot

    # Calculate arrival times for empty route
    arrival_times = estimate_arrival_times(route, driver_start_time)

    # Create a list of all customers except depot
    unrouted = [c for c in customers if c.id != 0]

    # Extract edges from previous day's route (if available)
    previous_edges = set()
    if day_index > 0 and previous_routes:
        prev_route = previous_routes[-1]  # Get most recent route
        for i in range(len(prev_route) - 1):
            previous_edges.add((prev_route[i].id, prev_route[i + 1].id))

    # Continue inserting customers until no more customers can be inserted
    while unrouted:
        best_customer = None
        best_position = None
        best_insertion_cost = float('inf')
        best_temp_route = None
        best_temp_arrival_times = None
        best_customer_sat = 0.0
        best_driver_sat = 0.0

        for customer in unrouted:
            # Try each possible insertion position
            for i in range(1, len(route)):
                # Create a new route with the customer inserted
                temp_route = route[:i] + [customer] + route[i:]

                # Calculate new arrival times
                temp_arrival_times = estimate_arrival_times(temp_route, driver_start_time)

                # Check if the route is feasible (finishes within working hours)
                if temp_arrival_times[-1] > driver_finish_time:
                    continue

                # Calculate customer satisfaction for this insertion
                num_customers = len(temp_route) - 2  # excluding depot
                if num_customers == 0:
                    customer_satisfaction = 1.0
                else:
                    total_customer_sat = 0.0
                    for j in range(1, len(temp_route) - 1):  # Skip depot
                        cust = temp_route[j]
                        arr_time = temp_arrival_times[j]
                        sat = calculate_customer_satisfaction(
                            arr_time,
                            cust.time_windows[day_index],
                            buffer_minutes
                        )
                        total_customer_sat += sat
                    customer_satisfaction = total_customer_sat / num_customers

                # Calculate base driver satisfaction
                driver_satisfaction, _, _ = calculate_driver_satisfaction(
                    temp_route, day_index, previous_routes, previous_working_times
                )

                # Apply edge consistency bonus - MODIFY THIS SECTION
                edge_bonus = 0.0
                if previous_edges and day_index > 0:  # Only apply for days after Monday
                    # Check if we're reusing an edge from prev_from to new customer
                    if (temp_route[i - 1].id, customer.id) in previous_edges:
                        edge_bonus += edge_consistency_bonus

                    # Check if we're reusing an edge from new customer to prev_to
                    if (customer.id, temp_route[i + 1].id) in previous_edges:
                        edge_bonus += edge_consistency_bonus

                    # Force extremely high edge consistency if zero customer/driver weights are set
                    if w_cust == 0.0 and w_driver == 0.0:
                        # If this is a consistent edge, make it MUCH more favorable
                        if edge_bonus > 0:
                            insertion_cost = -1000.0  # Extremely negative cost = highest priority
                        else:
                            insertion_cost = 1000.0  # Extremely high cost = lowest priority
                    else:
                        # Regular weighted calculation when satisfaction weights are non-zero
                        # Apply the bonus directly to the insertion cost rather than capping at 1.0
                        insertion_cost = (1.0 - customer_satisfaction) * w_cust + (
                                    1.0 - driver_satisfaction) * w_driver - edge_bonus
                else:
                    # For Monday, use regular calculation
                    insertion_cost = (1.0 - customer_satisfaction) * w_cust + (1.0 - driver_satisfaction) * w_driver

                # Update best insertion if this one is better
                if insertion_cost < best_insertion_cost:
                    best_insertion_cost = insertion_cost
                    best_customer = customer
                    best_position = i
                    best_temp_route = temp_route
                    best_temp_arrival_times = temp_arrival_times
                    best_customer_sat = customer_satisfaction
                    best_driver_sat = driver_satisfaction  # Store base driver satisfaction without bonus

        # If we found a feasible insertion, update the route
        if best_customer:
            route = best_temp_route
            unrouted.remove(best_customer)
            arrival_times = best_temp_arrival_times
        else:
            # No more feasible insertions found
            break

    # Calculate final metrics
    num_customers = len(route) - 2  # excluding depot
    customer_satisfaction = 0.0

    if num_customers > 0:
        total_customer_sat = 0.0
        for i in range(1, len(route) - 1):  # Skip depot
            customer = route[i]
            arr_time = arrival_times[i]
            sat = calculate_customer_satisfaction(
                arr_time,
                customer.time_windows[day_index],
                buffer_minutes
            )
            total_customer_sat += sat
        customer_satisfaction = total_customer_sat / num_customers
    else:
        customer_satisfaction = 1.0

    # Calculate final driver satisfaction (without any bonus - using the standard calculation)
    driver_satisfaction, _, _ = calculate_driver_satisfaction(
        route, day_index, previous_routes, previous_working_times
    )

    # Calculate total cost
    total_cost = (1.0 - customer_satisfaction) * w_cust + (1.0 - driver_satisfaction) * w_driver

    return route, total_cost, arrival_times, customer_satisfaction, driver_satisfaction

def calculate_working_time(route: List[Customer]) -> float:
    """
    Calculates the total working time for a route in minutes.
    Accounts for lunch break and customer-specific service times.
    """
    if len(route) <= 1:
        return 0.0

    # Start with the travel time
    total_distance = 0.0
    for i in range(1, len(route)):
        total_distance += distance(route[i - 1], route[i])

    travel_time = total_distance / SPEED_METERS_PER_MINUTE

    # Add the service time for all customers (excluding depot)
    service_time = sum(customer.service_time for customer in route[1:-1])

    # Add lunch break
    lunch_duration = LUNCH_BREAK_END - LUNCH_BREAK_START

    return travel_time + service_time + lunch_duration


# Update the function references for tabu_enhanced_two_opt
def tabu_enhanced_two_opt(
        route: List[Customer],
        day_index: int,
        previous_routes: List[List[Customer]],
        previous_working_times: List[float],
        w_cust: float = W_CUSTOMER,
        w_driver: float = W_DRIVER,
        edge_consistency_bonus: float = EDGE_CONSISTENCY_BONUS,  # Add this parameter
        driver_start_time: float = DRIVER_START_TIME,
        buffer_minutes: float = BUFFER_MINUTES,
        max_iterations: int = MAX_TABU_ITERATIONS,
        diversification_threshold: int = TABU_DIVERSIFICATION_THRESHOLD
) -> Tuple[List[Customer], float, List[float], float, float]:
    """
    Improves a route using 2-opt local search algorithm enhanced with tabu search,
    optimizing for both customer and driver satisfaction.

    Uses a dynamic tabu tenure based on the square root of the number of customers in the route.

    Returns:
    - The improved route
    - The objective value
    - The arrival times
    - Customer satisfaction component
    - Driver satisfaction component
    """
    if len(route) <= 3:  # Not enough nodes for 2-opt
        final_arrival_times = estimate_arrival_times(route, driver_start_time)

        # Calculate satisfaction components for minimal route
        customer_satisfaction = 1.0  # Assume perfect for minimal/empty route

        driver_satisfaction, _, _ = calculate_driver_satisfaction(
            route, day_index, previous_routes, previous_working_times
        )

        total_cost = (1.0 - customer_satisfaction) * w_cust + (1.0 - driver_satisfaction) * w_driver

        return route, total_cost, final_arrival_times, customer_satisfaction, driver_satisfaction

    # Calculate initial arrival times and costs
    best_route = route.copy()
    current_route = route.copy()

    # Calculate arrival times for the initial route
    best_arrival_times = estimate_arrival_times(best_route, driver_start_time)

    # Calculate number of customers (excluding depot)
    num_customers = len(best_route) - 2

    # Calculate dynamic tabu tenure based on the square root of number of customers
    tabu_tenure = int(math.sqrt(num_customers)) + 1

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

    # Calculate initial best cost
    best_cost = (1.0 - best_customer_satisfaction) * w_cust + (1.0 - best_driver_satisfaction) * w_driver
    current_cost = best_cost

    # Initialize tabu list and counters
    tabu_list = {}
    iteration = 0
    iterations_without_improvement = 0

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
            if diversified_arrival_times[-1] <= DRIVER_FINISH_TIME:
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

                # Skip if the route finishes after the end of working day
                if neighbor_arrival_times[-1] > DRIVER_FINISH_TIME:
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

                # Calculate new total cost
                # Fix variable shadowing issue in this section
                if day_index > 0:
                    # Extract edges from previous day's route
                    previous_edges = set()
                    if previous_routes:
                        prev_route = previous_routes[-1]  # Get most recent route
                        for idx in range(len(prev_route) - 1):  # Changed from i to idx
                            previous_edges.add((prev_route[idx].id, prev_route[idx + 1].id))

                    # Check how many edges are consistent with previous day's route
                    consistent_edges = 0
                    for idx in range(len(neighbor_route) - 1):  # Changed from k to idx
                        if (neighbor_route[idx].id, neighbor_route[idx + 1].id) in previous_edges:
                            consistent_edges += 1

                    # Add an edge consistency penalty/bonus for days after Monday
                    edge_consistency_penalty = -consistent_edges * edge_consistency_bonus

                    # If we're in zero-weight mode, prioritize edge consistency above all else
                    if w_cust == 0.0 and w_driver == 0.0:
                        neighbor_cost = -consistent_edges  # Make it negative so more consistent = better
                    else:
                        # Otherwise, just add the bonus to the regular calculation
                        neighbor_cost = (1.0 - neighbor_customer_satisfaction) * w_cust + \
                                        (1.0 - neighbor_driver_satisfaction) * w_driver + \
                                        edge_consistency_penalty
                else:
                    # For Monday, use regular calculation
                    neighbor_cost = (1.0 - neighbor_customer_satisfaction) * w_cust + \
                                    (1.0 - neighbor_driver_satisfaction) * w_driver

                move = (i, j)

                # Check tabu condition with aspiration criterion
                if (move not in tabu_list or neighbor_cost < best_cost) and neighbor_cost < best_neighbor_cost:
                    best_neighbor_cost = neighbor_cost
                    best_move = move
                    best_neighbor_route = neighbor_route.copy()
                    best_neighbor_arrival_times = neighbor_arrival_times.copy()
                    best_neighbor_customer_satisfaction = neighbor_customer_satisfaction
                    best_neighbor_driver_satisfaction = neighbor_driver_satisfaction

        if best_neighbor_route is None:
            break  # No improvement found or no feasible move

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
        else:
            iterations_without_improvement += 1

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

    return best_route, best_cost, best_arrival_times, best_customer_satisfaction, best_driver_satisfaction


# Update attempt_additional_insertions function
def attempt_additional_insertions(
        route: List[Customer],
        unvisited_customers: List[Customer],
        day_index: int,
        previous_routes: List[List[Customer]],
        previous_working_times: List[float],
        w_cust: float = W_CUSTOMER,
        w_driver: float = W_DRIVER,
        edge_consistency_bonus: float = EDGE_CONSISTENCY_BONUS,  # Add this parameter
        driver_start_time: float = DRIVER_START_TIME,
        driver_finish_time: float = DRIVER_FINISH_TIME,
        buffer_minutes: float = BUFFER_MINUTES
) -> Tuple[List[Customer], float, List[float], float, float]:
    """
    Attempts to insert additional unvisited customers into an optimized route
    if time capacity allows.

    Returns:
    - The enhanced route (with additional customers if possible)
    - The objective value
    - The arrival times
    - Customer satisfaction component
    - Driver satisfaction component
    """
    # Skip additional insertions on days after Monday when using zero weights
    # to maintain consistent routes
    if day_index > 0 and w_cust == 0.0 and w_driver == 0.0:
        # Calculate final arrival times
        final_arrival_times = estimate_arrival_times(route, driver_start_time)

        # Calculate final customer satisfaction
        num_customers = len(route) - 2  # excluding depot at start and end
        customer_satisfaction = 0.0

        if num_customers > 0:
            total_customer_sat = 0.0
            for i in range(1, len(route) - 1):  # Skip depot
                customer = route[i]
                arrival_time = final_arrival_times[i]
                satisfaction = calculate_customer_satisfaction(
                    arrival_time,
                    customer.time_windows[day_index],
                    buffer_minutes
                )
                total_customer_sat += satisfaction
            customer_satisfaction = total_customer_sat / num_customers
        else:
            customer_satisfaction = 1.0

        # Calculate final driver satisfaction
        driver_satisfaction, _, _ = calculate_driver_satisfaction(
            route, day_index, previous_routes, previous_working_times
        )

        # Calculate final cost
        total_cost = (1.0 - customer_satisfaction) * w_cust + (1.0 - driver_satisfaction) * w_driver

        print("Skipping additional insertions to maintain route consistency")
        return route, total_cost, final_arrival_times, customer_satisfaction, driver_satisfaction

    # If no unvisited customers, return the original route
    if not unvisited_customers:
        # Calculate final arrival times
        final_arrival_times = estimate_arrival_times(route, driver_start_time)

        # Calculate final customer satisfaction
        num_customers = len(route) - 2  # excluding depot at start and end
        customer_satisfaction = 0.0

        if num_customers > 0:
            total_customer_sat = 0.0
            for i in range(1, len(route) - 1):  # Skip depot
                customer = route[i]
                arrival_time = final_arrival_times[i]
                satisfaction = calculate_customer_satisfaction(
                    arrival_time,
                    customer.time_windows[day_index],
                    buffer_minutes
                )
                total_customer_sat += satisfaction
            customer_satisfaction = total_customer_sat / num_customers
        else:
            customer_satisfaction = 1.0

        # Calculate final driver satisfaction
        driver_satisfaction, _, _ = calculate_driver_satisfaction(
            route, day_index, previous_routes, previous_working_times
        )

        # Calculate final cost
        total_cost = (1.0 - customer_satisfaction) * w_cust + (1.0 - driver_satisfaction) * w_driver

        return route, total_cost, final_arrival_times, customer_satisfaction, driver_satisfaction

    # Extract edges from previous day's route (if available)
    previous_edges = set()
    if day_index > 0 and previous_routes:
        prev_route = previous_routes[-1]  # Get most recent route
        for k in range(len(prev_route) - 1):
            previous_edges.add((prev_route[k].id, prev_route[k + 1].id))

    # Create a copy of the current route to work with
    enhanced_route = route.copy()

    # Try to insert each unvisited customer into the route
    customers_inserted = True

    # Keep track of customers that were successfully inserted
    inserted_customers = []

    # Continue until no more customers can be inserted
    while customers_inserted and unvisited_customers:
        customers_inserted = False
        best_insertion_cost = float('inf')
        best_customer = None
        best_position = None
        best_route = None

        # Calculate current route arrival times
        current_arrival_times = estimate_arrival_times(enhanced_route, driver_start_time)

        # Try each unvisited customer
        for customer in unvisited_customers:
            # Try each possible insertion position
            for i in range(len(enhanced_route) - 1):
                # Create temporary route with customer inserted
                temp_route = enhanced_route[:i + 1] + [customer] + enhanced_route[i + 1:]

                # Estimate arrival times for this temporary route
                temp_arrival_times = estimate_arrival_times(temp_route, driver_start_time)

                # Skip if arrival times can't be calculated
                if not temp_arrival_times or len(temp_arrival_times) <= i + 1:
                    continue

                # Check if the route finishes by the end of working day
                if temp_arrival_times[-1] > driver_finish_time:
                    continue

                # Calculate customer satisfaction
                total_customer_sat = 0.0
                for j in range(1, len(temp_route) - 1):  # Skip depot
                    cust = temp_route[j]
                    arrival_time = temp_arrival_times[j]
                    satisfaction = calculate_customer_satisfaction(
                        arrival_time,
                        cust.time_windows[day_index],
                        buffer_minutes
                    )
                    total_customer_sat += satisfaction

                temp_customer_satisfaction = total_customer_sat / (len(temp_route) - 2)

                # Calculate driver satisfaction
                temp_driver_satisfaction, _, _ = calculate_driver_satisfaction(
                    temp_route, day_index, previous_routes, previous_working_times
                )

                # Apply edge consistency bonus - ADD THIS SECTION
                edge_bonus = 0.0
                if previous_edges and day_index > 0:
                    # Check if we're reusing edges from previous day
                    # Check the new edges being created by this insertion
                    if (temp_route[i].id, customer.id) in previous_edges:
                        edge_bonus += edge_consistency_bonus

                    if (customer.id, temp_route[i + 1].id) in previous_edges:
                        edge_bonus += edge_consistency_bonus

                    # Force extremely high edge consistency if zero customer/driver weights are set
                    if w_cust == 0.0 and w_driver == 0.0:
                        # If this is a consistent edge, make it MUCH more favorable
                        if edge_bonus > 0:
                            insertion_cost = -1000.0  # Extremely negative cost = highest priority
                        else:
                            insertion_cost = 1000.0  # Extremely high cost = lowest priority
                    else:
                        # Regular weighted calculation when satisfaction weights are non-zero
                        insertion_cost = (1.0 - temp_customer_satisfaction) * w_cust + (
                                1.0 - temp_driver_satisfaction) * w_driver - edge_bonus
                else:
                    # For Monday, use regular calculation
                    insertion_cost = (1.0 - temp_customer_satisfaction) * w_cust + (
                            1.0 - temp_driver_satisfaction) * w_driver

                # Update best insertion if this one is better
                if insertion_cost < best_insertion_cost:
                    best_insertion_cost = insertion_cost
                    best_customer = customer
                    best_position = i
                    best_route = temp_route.copy()

        # If we found a feasible insertion, update the route
        if best_customer is not None:
            enhanced_route = best_route
            unvisited_customers.remove(best_customer)
            inserted_customers.append(best_customer)
            customers_inserted = True

    # If we inserted any customers, log this information
    if inserted_customers:
        print(f"Post-optimization: Successfully inserted {len(inserted_customers)} additional customers:")
        for cust in inserted_customers:
            print(f"  - Customer {cust.id}")

    # Calculate final metrics for the enhanced route
    final_arrival_times = estimate_arrival_times(enhanced_route, driver_start_time)

    # Calculate final customer satisfaction
    num_customers = len(enhanced_route) - 2  # excluding depot at start and end
    customer_satisfaction = 0.0

    if num_customers > 0:
        total_customer_sat = 0.0
        for i in range(1, len(enhanced_route) - 1):  # Skip depot
            customer = enhanced_route[i]
            arrival_time = final_arrival_times[i]
            satisfaction = calculate_customer_satisfaction(
                arrival_time,
                customer.time_windows[day_index],
                buffer_minutes
            )
            total_customer_sat += satisfaction
        customer_satisfaction = total_customer_sat / num_customers
    else:
        customer_satisfaction = 1.0

    # Calculate final driver satisfaction
    driver_satisfaction, _, _ = calculate_driver_satisfaction(
        enhanced_route, day_index, previous_routes, previous_working_times
    )

    # Calculate final cost
    total_cost = (1.0 - customer_satisfaction) * w_cust + (1.0 - driver_satisfaction) * w_driver

    return enhanced_route, total_cost, final_arrival_times, customer_satisfaction, driver_satisfaction

# Other functions (kept but not shown for brevity)
def distance(customer1: Customer, customer2: Customer) -> float:
    """Calculate Manhattan distance between two customers"""
    return abs(customer1.x - customer2.x) + abs(customer1.y - customer2.y)

def check_lunch_break_conflict(arrival_time: float) -> bool:
    """Check if the arrival time conflicts with lunch break"""
    return LUNCH_BREAK_START <= arrival_time < LUNCH_BREAK_END


def adjust_for_lunch_break(time_value: float) -> float:
    """
    Adjust the time value to account for lunch break.
    If the time falls within lunch break, it is pushed to the end of lunch break.
    """
    if LUNCH_BREAK_START <= time_value < LUNCH_BREAK_END:
        return LUNCH_BREAK_END
    return time_value


def calculate_customer_satisfaction(
        arrival_time: float,
        time_window: TimeWindow,
        buffer_minutes: float = BUFFER_MINUTES
) -> float:
    """
    Calculate customer satisfaction based on arrival time and time window.

    The satisfaction is calculated as:
    1. 1.0 (100%) if arrival is within the time window
    2. Decreases linearly if arrival is early or late, but within buffer
    3. Decreases more steeply if arrival is very early or very late
    """
    # If arrival is within time window: Perfect satisfaction
    if time_window.start <= arrival_time <= time_window.end:
        return 1.0

    # Calculate deviation from time window
    if arrival_time < time_window.start:
        deviation = time_window.start - arrival_time
    else:  # arrival_time > time_window.end
        deviation = arrival_time - time_window.end

    # Small deviation (within buffer): Linear decrease
    if deviation <= buffer_minutes:
        return 1.0 - (deviation / buffer_minutes) * 0.5

    # Medium deviation (beyond buffer but not severe)
    elif deviation <= SATISFACTION_STEEP_THRESHOLD:
        # Satisfaction at buffer point is 0.5, then decreases to 0.25 at threshold
        buffer_satisfaction = 0.5
        threshold_satisfaction = 0.25
        deviation_beyond_buffer = deviation - buffer_minutes
        max_deviation_beyond_buffer = SATISFACTION_STEEP_THRESHOLD - buffer_minutes
        return buffer_satisfaction - (deviation_beyond_buffer / max_deviation_beyond_buffer) * (
                    buffer_satisfaction - threshold_satisfaction)

    # Large deviation: Steeper decrease
    else:
        # Start at 0.25 for threshold deviation, approach 0 asymptotically
        base_satisfaction = 0.25
        extra_deviation = deviation - SATISFACTION_STEEP_THRESHOLD
        return base_satisfaction / (1.0 + 0.1 * extra_deviation)


def calculate_work_time_satisfaction(working_time: float) -> float:
    """
    Calculate work time satisfaction based on how close the working time
    is to the ideal working hours.

    Returns a satisfaction score between 0 and 1.
    """
    # Convert working time to hours for clarity
    working_hours = working_time / 60.0

    # If working time is less than ideal, satisfaction decreases linearly
    if working_hours <= IDEAL_WORKING_HOURS:
        # Scale linearly from 0.5 (0 hours) to 1.0 (ideal hours)
        return 0.5 + (working_hours / IDEAL_WORKING_HOURS) * 0.5

    # If working time is between ideal and max, satisfaction decreases linearly
    elif working_hours <= MAX_WORKING_HOURS:
        # Scale linearly from 1.0 (ideal hours) to 0.5 (max hours)
        overwork_ratio = (working_hours - IDEAL_WORKING_HOURS) / (MAX_WORKING_HOURS - IDEAL_WORKING_HOURS)
        return 1.0 - overwork_ratio * 0.5

    # If working time exceeds max hours, satisfaction decreases more steeply
    else:
        # Start at 0.5 for max hours and approach 0 asymptotically
        excess_hours = working_hours - MAX_WORKING_HOURS
        return 0.5 / (1.0 + excess_hours)


def calculate_work_time_consistency(
        working_time: float,
        previous_working_times: List[float]
) -> float:
    """
    Calculate consistency in working time across days.
    Higher consistency means more similar working times.

    Returns a satisfaction score between 0 and 1.
    """
    if not previous_working_times:
        return 1.0  # Perfect consistency if no previous days

    # Calculate average of previous working times
    avg_previous_time = sum(previous_working_times) / len(previous_working_times)

    # Calculate deviation from average (as a percentage)
    deviation_percentage = abs(working_time - avg_previous_time) / avg_previous_time

    # Penalize large deviations - satisfaction decreases as deviation increases
    # A 0% deviation gives 1.0 satisfaction, a 20% deviation gives 0.5 satisfaction
    if deviation_percentage <= 0.2:
        return 1.0 - deviation_percentage * 2.5
    else:
        # For deviations > 20%, satisfaction decreases more slowly
        return max(0.5 - (deviation_percentage - 0.2), 0.0)


def calculate_route_consistency(
        route: List[Customer],
        previous_routes: List[List[Customer]]
) -> float:
    """
    Calculate route consistency based on how many customers from previous routes
    are also in the current route.

    Returns a satisfaction score between 0 and 1.
    """
    if not previous_routes:
        return 1.0  # Perfect consistency if no previous days

    # Extract customer IDs from current route (excluding depot)
    current_ids = {customer.id for customer in route if customer.id != 0}

    if not current_ids:
        return 1.0  # Perfect consistency if empty route

    # Calculate consistency with each previous route
    consistencies = []

    for prev_route in previous_routes:
        prev_ids = {customer.id for customer in prev_route if customer.id != 0}

        if not prev_ids:
            consistencies.append(1.0)  # Perfect consistency with an empty route
            continue

        # Calculate Jaccard similarity (intersection over union)
        intersection = len(current_ids.intersection(prev_ids))
        union = len(current_ids.union(prev_ids))

        if union == 0:
            consistencies.append(1.0)  # Both empty routes
        else:
            consistencies.append(intersection / union)

    # Return average consistency across all previous routes
    return sum(consistencies) / len(consistencies)


def calculate_driver_satisfaction(
        route: List[Customer],
        day_index: int,
        previous_routes: List[List[Customer]],
        previous_working_times: List[float],
        work_time_weight: float = WORK_TIME_WEIGHT,
        route_consistency_weight: float = ROUTE_CONSISTENCY_WEIGHT
) -> Tuple[float, float, float]:
    """
    Calculate driver satisfaction based on working time and route consistency.

    Returns:
    - Overall driver satisfaction (0-1)
    - Work time satisfaction component (0-1)
    - Route consistency satisfaction component (0-1)
    """
    # Calculate working time for this route
    working_time = calculate_working_time(route)

    # Calculate work time satisfaction
    work_time_sat = calculate_work_time_satisfaction(working_time)

    # Calculate work time consistency with previous days
    work_time_consistency = calculate_work_time_consistency(working_time, previous_working_times)

    # Calculate route consistency with previous days
    route_consistency = calculate_route_consistency(route, previous_routes)

    # Combine consistency satisfactions (weighted average)
    consistency_sat = work_time_consistency * work_time_weight + route_consistency * route_consistency_weight

    # Overall driver satisfaction is average of time satisfaction and consistency
    overall_sat = (work_time_sat + consistency_sat) / 2.0

    return overall_sat, work_time_sat, consistency_sat


def diversify_route(route: List[Customer]) -> List[Customer]:
    """
    Diversify the route by performing a series of random 2-opt moves.
    This helps escape local optima during the tabu search.
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


def main():
    # Use customers imported from customer_data.py
    customers_list = all_customers

    print(f"Loaded {len(customers_list) - 1} customers and depot from customer_data.py")

    # Run for 5 days (Mon-Fri)
    all_routes = []
    all_objective_values = []
    all_arrival_times = []
    all_customer_satisfactions = []
    all_driver_satisfactions = []
    all_working_times = []

    # Keep track of unvisited customers for each day
    all_unvisited_customers = []

    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

    for day_idx in range(5):
        print(f"\n----- Processing {days[day_idx]} -----")

        # Track execution time
        start_time_execution = time.time()

        # Previous routes and working times for driver satisfaction calculation
        previous_routes = all_routes.copy()
        previous_working_times = all_working_times.copy()

        # For day 0 (Monday), use normal weights
        # For other days, decide if we want to use zero weights or normal weights
        if day_idx == 0:
            # Monday - use normal weights
            day_w_cust = W_CUSTOMER
            day_w_driver = W_DRIVER
            day_edge_bonus = EDGE_CONSISTENCY_BONUS
        else:
            # For subsequent days, use adjusted weights based on your constants.py
            # If you want to force identical routes, set these to 0.0/10.0
            # Otherwise use your normal weights but with a high edge bonus
            day_w_cust = W_CUSTOMER  # This will be 0.0 if using constants_test
            day_w_driver = W_DRIVER  # This will be 0.0 if using constants_test
            day_edge_bonus = EDGE_CONSISTENCY_BONUS  # This will be 10.0 if using constants_test

        print(f"Day weights - Customer: {day_w_cust}, Driver: {day_w_driver}, Edge bonus: {day_edge_bonus}")

        # Step 1: Create initial route using insertion heuristic with day-specific weights
        print("Generating initial route using insertion heuristic...")
        initial_route, initial_cost, _, initial_cust_sat, initial_driver_sat = insertion_heuristic(
            customers=customers_list,
            day_index=day_idx,
            previous_routes=previous_routes,
            previous_working_times=previous_working_times,
            w_cust=day_w_cust,
            w_driver=day_w_driver,
            edge_consistency_bonus=day_edge_bonus
        )

        # ... rest of your main function remains the same
        # Just make sure to pass the day-specific weights to tabu_enhanced_two_opt
        # and attempt_additional_insertions as well

        # Identify unvisited customers from the initial route
        visited_customer_ids = {customer.id for customer in initial_route}
        unvisited_customers = [customer for customer in customers_list if
                               customer.id not in visited_customer_ids and customer.id != 0]  # Exclude depot

        # Store unvisited customers for this day
        all_unvisited_customers.append(unvisited_customers)

        print(
            f"Initial route contains {len(initial_route) - 2} customers. {len(unvisited_customers)} customers unvisited.")

        # Step 2: Improve route using tabu-enhanced 2-opt with dynamic tabu tenure
        print("Improving route using tabu-enhanced 2-opt local search...")
        improved_route, improved_cost, improved_arrival_times, improved_cust_sat, improved_driver_sat = tabu_enhanced_two_opt(
            route=initial_route,
            day_index=day_idx,
            previous_routes=previous_routes,
            previous_working_times=previous_working_times,
            w_cust=day_w_cust,
            w_driver=day_w_driver,
            edge_consistency_bonus=day_edge_bonus  # Add this parameter
        )

        # Step 3: Attempt to insert additional unvisited customers post-optimization
        print("Attempting to insert additional customers post-optimization...")
        final_route, final_cost, arrival_times, customer_sat, driver_sat = attempt_additional_insertions(
            route=improved_route,
            unvisited_customers=unvisited_customers,
            day_index=day_idx,
            previous_routes=previous_routes,
            previous_working_times=previous_working_times,
            w_cust=day_w_cust,
            w_driver=day_w_driver,
            edge_consistency_bonus=day_edge_bonus
        )

        # Calculate working time for this route
        working_time = calculate_working_time(final_route)

        # Calculate execution time
        execution_time = time.time() - start_time_execution

        # Compare results
        print(f"Initial route cost: {initial_cost:.2f}")
        print(f"Improved route cost (after tabu search): {improved_cost:.2f}")
        print(f"Final route cost (after additional insertions): {final_cost:.2f}")

        if improved_cost < initial_cost:
            print(f"Improvement from tabu search: {(initial_cost - improved_cost) / initial_cost * 100:.2f}%")
        else:
            print("No improvement found through tabu-enhanced 2-opt")

        initial_customers = len(initial_route) - 2
        final_customers = len(final_route) - 2
        if final_customers > initial_customers:
            print(f"Additional customers inserted: {final_customers - initial_customers}")
            print(
                f"Percentage increase in customers served: {(final_customers - initial_customers) / initial_customers * 100:.2f}%")

        print(f"Customer satisfaction: {customer_sat:.2f}")
        print(f"Driver satisfaction: {driver_sat:.2f}")
        print(f"Working time: {working_time:.2f} minutes ({working_time / 60:.2f} hours)")
        print(f"Execution time: {execution_time:.2f} seconds")
        print(f"Customers served: {len(final_route) - 2}")  # -2 for depot at start and end

        # Store results
        all_routes.append(final_route)
        all_objective_values.append(final_cost)
        all_arrival_times.append(arrival_times)
        all_customer_satisfactions.append(customer_sat)
        all_driver_satisfactions.append(driver_sat)
        all_working_times.append(working_time)

    # Print detailed results
    print("\n----- FINAL RESULTS -----")
    total_deviation = 0
    total_customers_served = 0

    for d in range(5):
        print(f"\n----- {days[d]} -----")
        print(f"{days[d]} route: {[c.id for c in all_routes[d]]}")

        working_time = all_working_times[d]
        print(f"{days[d]} working time: {working_time:.2f} minutes ({working_time / 60:.2f} hours)")
        print(f"{days[d]} objective value: {all_objective_values[d]:.2f}")
        print(f"{days[d]} customer satisfaction: {all_customer_satisfactions[d]:.2f}")
        print(f"{days[d]} driver satisfaction: {all_driver_satisfactions[d]:.2f}")

        # Print route start and end times
        start_time = DRIVER_START_TIME
        if all_arrival_times[d]:
            end_time = all_arrival_times[d][-1]
        else:
            end_time = start_time

        start_hour = int(start_time // 60)
        start_minute = int(start_time % 60)
        end_hour = int(end_time // 60)
        end_minute = int(end_time % 60)

        print(f"{days[d]} start time: {start_hour:02d}:{start_minute:02d}")
        print(f"{days[d]} end time: {end_hour:02d}:{end_minute:02d}")

        # Print unvisited customer count
        print(f"Unvisited customers: {len(all_unvisited_customers[d])}")

        # Print customer service details
        print("\nCustomer Service Details:")
        total_satisfaction = 0.0

        for i in range(1, len(all_routes[d]) - 1):  # Skip depot
            customer = all_routes[d][i]
            arrival_time = all_arrival_times[d][i]
            time_window = customer.time_windows[d]
            satisfaction = calculate_customer_satisfaction(arrival_time, time_window)
            total_satisfaction += satisfaction

            # Format times for display
            arrival_hour = int(arrival_time // 60)
            arrival_minute = int(arrival_time % 60)
            window_start_hour = int(time_window.start // 60)
            window_start_minute = int(time_window.start % 60)
            window_end_hour = int(time_window.end // 60)
            window_end_minute = int(time_window.end % 60)

            # Calculate time deviation
            if arrival_time < time_window.start:
                deviation = time_window.start - arrival_time
                deviation_type = "early"
            elif arrival_time > time_window.end:
                deviation = arrival_time - time_window.end
                deviation_type = "late"
            else:
                deviation = 0
                deviation_type = "on time"

            total_deviation += deviation
            total_customers_served += 1

            # Format deviation for display
            deviation_hours = int(deviation // 60)
            deviation_minutes = int(deviation % 60)

            # Display service time for each customer
            service_time = customer.service_time

            print(f"  Customer {customer.id}:")
            print(
                f"    Time Window: {window_start_hour:02d}:{window_start_minute:02d}-{window_end_hour:02d}:{window_end_minute:02d}")
            print(f"    Arrival Time: {arrival_hour:02d}:{arrival_minute:02d}")
            print(f"    Service Time: {service_time:.2f} minutes")
            print(f"    Status: {deviation_type.capitalize()}")
            if deviation > 0:
                print(f"    Deviation: {deviation_hours}h {deviation_minutes}min")
            print(f"    Satisfaction: {satisfaction:.2f}")

        # Print day summary
        customers_in_day = len(all_routes[d]) - 2  # Exclude depot at start and end
        if customers_in_day > 0:
            avg_satisfaction = total_satisfaction / customers_in_day
            print(f"\nDay Summary:")
            print(f"  Customers served: {customers_in_day}")
            print(f"  Average satisfaction: {avg_satisfaction:.2f}")

        print("-" * 30)

    # Print overall statistics
    if total_customers_served > 0:
        avg_deviation = total_deviation / total_customers_served
        print(f"\nOverall Statistics:")
        print(f"  Total customers served across all days: {total_customers_served}")
        print(f"  Total time deviation: {total_deviation:.2f} minutes")
        print(f"  Average deviation per customer: {avg_deviation:.2f} minutes")

        # Calculate percentage of customers served from total available
        total_available = (len(customers_list) - 1) * 5  # -1 for depot, *5 for days
        percentage_served = (total_customers_served / total_available) * 100
        print(f"  Percentage of available customer-days served: {percentage_served:.2f}%")


if __name__ == "__main__":
    main()