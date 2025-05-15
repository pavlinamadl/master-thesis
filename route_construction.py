from typing import List, Tuple, Set
from customer_data import Customer

from constants import (
    DRIVER_START_TIME, DRIVER_FINISH_TIME,
    BUFFER_MINUTES,
    ALPHA,  # Using ALPHA instead of W_CUSTOMER and W_DRIVER
    EDGE_CONSISTENCY_BONUS,
    MUST_SERVE_PRIORITY
)

from time_utils import estimate_arrival_times, calculate_customer_satisfaction
from satisfaction_metrics import calculate_driver_satisfaction


def insertion_heuristic(
        customers: List[Customer],
        day_index: int,
        previous_routes: List[List[Customer]],
        previous_working_times: List[float],
        alpha: float = ALPHA,  # Changed parameter name from w_cust
        edge_consistency_bonus: float = EDGE_CONSISTENCY_BONUS,  # 0 is minimal, >0.5 is very strong
        driver_start_time: float = DRIVER_START_TIME,
        driver_finish_time: float = DRIVER_FINISH_TIME,
        buffer_minutes: float = BUFFER_MINUTES
) -> Tuple[List[Customer], float, List[float], float, float]:
    """
    Constructs a route using an insertion heuristic that considers both
    customer satisfaction and driver satisfaction, with additional consideration
    for edge consistency with previous routes.

    IMPORTANT: Modified to ensure all customers marked as must_serve=True are included.
    For alpha=0.0, day 0 (Monday) still considers customer satisfaction but days 1-4
    prioritize consistency with previous day's route.

    Args:
        customers: List of all customers including depot
        day_index: Current day index (0-4 for Monday-Friday)
        previous_routes: List of routes from previous days
        previous_working_times: List of working times from previous days
        alpha: Weight for customer satisfaction (between 0 and 1)
               Driver satisfaction weight will be (1-alpha)
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
    print(f"Running insertion heuristic with alpha={alpha}, edge_consistency_bonus={edge_consistency_bonus}")

    depot = customers[0]
    route = [depot, depot]  # Start and end at depot

    # Calculate arrival times for empty route
    arrival_times = estimate_arrival_times(route, driver_start_time)

    # Create a list of all customers except depot
    unrouted = [c for c in customers if c.id != 0]

    # Separate must-serve customers from others
    must_serve_customers = [c for c in unrouted if c.must_serve]
    optional_customers = [c for c in unrouted if not c.must_serve]

    # Extract edges from previous day's route (if available)
    previous_edges = set()
    if day_index > 0 and previous_routes:
        prev_route = previous_routes[-1]  # Get most recent route
        for i in range(len(prev_route) - 1):
            previous_edges.add((prev_route[i].id, prev_route[i + 1].id))
        print(f"Found {len(previous_edges)} edges from previous day's route")

    # Special case for alpha=0.0 on the first day (Monday)
    # We use alpha=0.5 temporarily to ensure customer time windows are considered
    effective_alpha = alpha

    if alpha == 0.0 and day_index == 0:
          print("Special case: alpha=0.0 on first day, using alpha=0.5 temporarily to consider time windows")
          effective_alpha = 0.5

    # PHASE 1: First insert all must-serve customers
    for customer_list in [must_serve_customers, optional_customers]:
        customer_type = "must-serve" if customer_list == must_serve_customers else "optional"
        print(f"Inserting {len(customer_list)} {customer_type} customers")

        while customer_list:
            best_customer = None
            best_position = None
            best_insertion_cost = float('inf')
            best_temp_route = None
            best_temp_arrival_times = None
            best_customer_sat = 0.0
            best_driver_sat = 0.0

            for customer in customer_list:
                # Try each possible insertion position
                for i in range(1, len(route)):
                    # Create a new route with the customer inserted
                    temp_route = route[:i] + [customer] + route[i:]

                    # Calculate new arrival times
                    temp_arrival_times = estimate_arrival_times(temp_route, driver_start_time)

                    # Check if the route is feasible (finishes within working hours)

                    # ENFORCE time-window feasibility on *this* customer
                    tw = customer.time_windows[day_index]
                    arrival = temp_arrival_times[i]
                    if not (tw.start <= arrival <= tw.end):
                        continue

                    # Allow violations for must-serve customers if necessary
                    if temp_arrival_times[-1] > driver_finish_time and not (
                            customer.must_serve and customer_list == must_serve_customers):
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

                    # Apply eadge consistency bonus - For days after Monday with alpha=0
                    edge_bonus = 0.0
                    if previous_edges and day_index > 0:  # Only apply for days after Monday
                        # Check if we're reusing an edge from prev_from to new customer
                        if (temp_route[i - 1].id, customer.id) in previous_edges:
                            edge_bonus += edge_consistency_bonus

                        # Check if we're reusing an edge from new customer to prev_to
                        if (customer.id, temp_route[i + 1].id) in previous_edges:
                            edge_bonus += edge_consistency_bonus

                        # Create a balanced weight distribution between the three components
                        # For alpha = 0.5, we want:
                        # - 50% for customer satisfaction
                        # - ~35% for driver satisfaction
                        # - ~15% for edge consistency

                        # 1. Base cost: split between customer and driver satisfaction
                        base_cost = (
                                (1.0 - customer_satisfaction) * effective_alpha
                                + (1.0 - driver_satisfaction) * (1.0 - effective_alpha)
                        )

                        # 2. Edge‐consistency reward: edge_bonus is already
                        #    edge_consistency_bonus * (# of reused edges)
                        edge_reward = edge_bonus * (1.0 - effective_alpha)

                        # 3. Final insertion cost: we *subtract* the reward
                        insertion_cost = base_cost - edge_reward

                        # 4. Still give extra priority to must‐serve customers
                        if customer.must_serve:
                            insertion_cost -= MUST_SERVE_PRIORITY

                        # 5. Never go below zero
                        insertion_cost = max(0.0, insertion_cost)

                        # 6. Still give priority to must-serve customers
                        if customer.must_serve:
                            insertion_cost -= MUST_SERVE_PRIORITY

                    else:
                        # For Monday, use the effective alpha (either actual or temporary value)
                        insertion_cost = (1.0 - customer_satisfaction) * effective_alpha + (
                                1.0 - driver_satisfaction) * (1.0 - effective_alpha)

                    # For must-serve customers, give additional priority to feasible insertions
                    if customer.must_serve and customer_list == must_serve_customers:
                        insertion_cost -= 2000.0  # Higher priority than edge consistency

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
                customer_list.remove(best_customer)
                arrival_times = best_temp_arrival_times
                print(
                    f"Inserted customer {best_customer.id} at position {best_position} with cost {best_insertion_cost:.2f}")
            else:
                # No more feasible insertions found
                # If we're still processing must-serve customers, we need to force insertion by extending working hours
                if customer_list == must_serve_customers and customer_list:
                    print("WARNING: Unable to schedule all must-serve customers within normal working hours.")
                    print("Forcing insertion of must-serve customers by extending working hours.")

                    # Try again with relaxed time constraints
                    extended_finish_time = driver_finish_time + 120  # Add 2 hours

                    for customer in customer_list[:]:  # Use a copy to avoid modification during iteration
                        found_position = False
                        min_finish_time = float('inf')
                        best_position = None
                        best_route = None

                        for i in range(1, len(route)):
                            temp_route = route[:i] + [customer] + route[i:]
                            temp_arrival_times = estimate_arrival_times(temp_route, driver_start_time)

                            if temp_arrival_times[-1] < min_finish_time:
                                min_finish_time = temp_arrival_times[-1]
                                best_position = i
                                best_route = temp_route
                                best_arrival_times = temp_arrival_times

                            if temp_arrival_times[-1] <= extended_finish_time:
                                found_position = True
                                break

                        if found_position or best_route:
                            # Use the best route found, even if it exceeds extended working hours
                            route = best_route
                            arrival_times = best_arrival_times
                            customer_list.remove(customer)
                            print(
                                f"  Inserted must-serve customer {customer.id}, route now ends at {min_finish_time:.1f} minutes")
                        else:
                            print(f"  CRITICAL: Could not insert customer {customer.id} even with extended hours!")
                break  # Exit the loop if no more insertions are possible for optional customers

    # Check if we managed to insert all must-serve customers
    unserved_must_serve = [c.id for c in customers if c.must_serve and c not in route]
    if unserved_must_serve:
        print(f"WARNING: Could not serve all must-serve customers. Unserved: {unserved_must_serve}")
        # If absolutely necessary, you could raise an exception here or implement an emergency algorithm

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

    # Calculate total cost using alpha (not effective_alpha) for consistency in return values
    total_cost = (1.0 - customer_satisfaction) * alpha + (1.0 - driver_satisfaction) * (1.0 - alpha)

    return route, total_cost, arrival_times, customer_satisfaction, driver_satisfaction