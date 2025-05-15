from typing import List, Tuple, Set

from customer_data import Customer
from constants import (
    DRIVER_START_TIME, DRIVER_FINISH_TIME,
    BUFFER_MINUTES,
    ALPHA,  # Using ALPHA instead of W_CUSTOMER and W_DRIVER
    EDGE_CONSISTENCY_BONUS
)

from time_utils import estimate_arrival_times, calculate_customer_satisfaction
from satisfaction_metrics import calculate_driver_satisfaction


def attempt_additional_insertions(
        route: List[Customer],
        unvisited_customers: List[Customer],
        day_index: int,
        previous_routes: List[List[Customer]],
        previous_working_times: List[float],
        alpha: float = ALPHA,  # Changed parameter name from w_cust
        edge_consistency_bonus: float = EDGE_CONSISTENCY_BONUS,
        driver_start_time: float = DRIVER_START_TIME,
        driver_finish_time: float = DRIVER_FINISH_TIME,
        buffer_minutes: float = BUFFER_MINUTES
) -> Tuple[List[Customer], float, List[float], float, float]:
    """
    Attempts to insert additional unvisited customers into an optimized route
    if time capacity allows. Modified to ensure all must-serve customers are included.
    For alpha=0.0, day 0 (Monday) still considers customer satisfaction but days 1-4
    prioritize consistency with previous day's route.

    Args:
        route: Initial optimized route
        unvisited_customers: Customers not yet in the route
        day_index: Current day index (0-4 for Monday-Friday)
        previous_routes: List of routes from previous days
        previous_working_times: List of working times from previous days
        alpha: Weight for customer satisfaction (0-1), driver weight is (1-alpha)
        edge_consistency_bonus: Bonus applied for reusing edges from previous routes
        driver_start_time: Start time for the driver's day
        driver_finish_time: End time for the driver's day
        buffer_minutes: Buffer time for calculating customer satisfaction

    Returns:
        Tuple containing:
        - The enhanced route (with additional customers if possible)
        - The objective value
        - The arrival times
        - Customer satisfaction component
        - Driver satisfaction component
    """
    print(f"Running additional insertions with alpha={alpha}, edge_consistency_bonus={edge_consistency_bonus}")

    # Special case for alpha=0.0 on the first day (Monday)
    # We use alpha=0.5 temporarily to ensure customer time windows are considered
    effective_alpha = alpha
    if alpha == 0.0 and day_index == 0:
        print("Special case: alpha=0.0 on first day, using alpha=0.5 temporarily to consider time windows")
        effective_alpha = 0.5  # Use balanced weight for Day 0 only

    # Skip additional insertions on days after Monday when using alpha=0
    # to maintain consistent routes with previous days
    if day_index > 0 and alpha == 0.0:
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

        # Calculate final cost using actual alpha (not effective alpha)
        total_cost = (1.0 - customer_satisfaction) * alpha + (1.0 - driver_satisfaction) * (1.0 - alpha)

        print(f"Alpha=0.0 and day>{day_index}: Skipping additional insertions to maintain route consistency")
        return route, total_cost, final_arrival_times, customer_satisfaction, driver_satisfaction

    # Check for must-serve customers in unvisited list (this should be handled by insertion_heuristic,
    # but we'll double-check here)
    must_serve_unvisited = [c for c in unvisited_customers if c.must_serve]

    if must_serve_unvisited:
        print(f"WARNING: Found {len(must_serve_unvisited)} must-serve customers not in route!")
        print(f"Customer IDs: {[c.id for c in must_serve_unvisited]}")
        print("Forcing insertion of these customers...")

        # Force insertion of must-serve customers
        for customer in must_serve_unvisited:
            best_position = None
            best_insertion_cost = float('inf')
            best_route = None
            best_arrival_times = None

            # Try each possible insertion position
            for i in range(1, len(route)):
                temp_route = route[:i] + [customer] + route[i:]
                temp_arrival_times = estimate_arrival_times(temp_route, driver_start_time)

                # Even if it extends working hours, calculate the cost
                if temp_arrival_times:
                    # Calculate customer satisfaction
                    total_customer_sat = 0.0
                    for j in range(1, len(temp_route) - 1):
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

                    # Calculate insertion cost using effective alpha (for Monday with alpha=0)
                    insertion_cost = (1.0 - temp_customer_satisfaction) * effective_alpha + \
                                     (1.0 - temp_driver_satisfaction) * (1.0 - effective_alpha)

                    # Update best insertion if better
                    if insertion_cost < best_insertion_cost:
                        best_insertion_cost = insertion_cost
                        best_position = i
                        best_route = temp_route
                        best_arrival_times = temp_arrival_times

            if best_route:
                route = best_route
                unvisited_customers.remove(customer)
                arrival_times = best_arrival_times
                print(f"  Successfully inserted must-serve customer {customer.id}")
            else:
                print(f"  CRITICAL: Could not insert must-serve customer {customer.id}!")

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

        # Calculate final cost using actual alpha
        total_cost = (1.0 - customer_satisfaction) * alpha + (1.0 - driver_satisfaction) * (1.0 - alpha)

        return route, total_cost, final_arrival_times, customer_satisfaction, driver_satisfaction

    # Separate remaining unvisited customers into must-serve and optional
    unvisited_must_serve = [c for c in unvisited_customers if c.must_serve]
    unvisited_optional = [c for c in unvisited_customers if not c.must_serve]

    # Extract edges from previous day's route (if available)
    previous_edges = set()
    if day_index > 0 and previous_routes:
        prev_route = previous_routes[-1]  # Get most recent route
        for k in range(len(prev_route) - 1):
            previous_edges.add((prev_route[k].id, prev_route[k + 1].id))

    # Create a copy of the current route to work with
    enhanced_route = route.copy()
    arrival_times = estimate_arrival_times(enhanced_route, driver_start_time)

    # Process must-serve customers first, then optional customers
    for customer_group in [unvisited_must_serve, unvisited_optional]:
        customers_inserted = True
        inserted_customers = []

        while customers_inserted and customer_group:
            customers_inserted = False
            best_insertion_cost = float('inf')
            best_customer = None
            best_position = None
            best_route = None
            best_arrival_times = None

            # Calculate current route arrival times
            current_arrival_times = estimate_arrival_times(enhanced_route, driver_start_time)

            # Try each unvisited customer
            for customer in customer_group:
                # Try each possible insertion position
                for i in range(len(enhanced_route) - 1):
                    # Create temporary route with customer inserted
                    temp_route = enhanced_route[:i + 1] + [customer] + enhanced_route[i + 1:]

                    # Estimate arrival times for this temporary route
                    temp_arrival_times = estimate_arrival_times(temp_route, driver_start_time)

                    # Skip if arrival times can't be calculated
                    if not temp_arrival_times or len(temp_arrival_times) <= i + 1:
                        continue

                    # For must-serve customers, allow exceeding finish time if necessary
                    # For optional customers, enforce the normal constraints
                    if temp_arrival_times[-1] > driver_finish_time and not customer.must_serve:
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

                    # Apply edge consistency bonus
                    edge_bonus = 0.0
                    if previous_edges and day_index > 0:
                        # Check if we're reusing edges from previous day
                        # Check the new edges being created by this insertion
                        if (temp_route[i].id, customer.id) in previous_edges:
                            edge_bonus += edge_consistency_bonus

                        if (customer.id, temp_route[i + 1].id) in previous_edges:
                            edge_bonus += edge_consistency_bonus

                        # Regular weighted calculation using effective_alpha
                        # Scale edge bonus by (1-effective_alpha)
                        insertion_cost = (1.0 - temp_customer_satisfaction) * effective_alpha + \
                                         (1.0 - temp_driver_satisfaction) * (1.0 - effective_alpha) - \
                                         (edge_bonus * (1.0 - effective_alpha))
                    else:
                        # For Monday, use effective_alpha
                        insertion_cost = (1.0 - temp_customer_satisfaction) * effective_alpha + \
                                         (1.0 - temp_driver_satisfaction) * (1.0 - effective_alpha)

                    # For must-serve customers, give additional high priority
                    if customer.must_serve:
                        insertion_cost -= 2000.0  # Higher priority than edge consistency

                    # Update best insertion if this one is better
                    if insertion_cost < best_insertion_cost:
                        best_insertion_cost = insertion_cost
                        best_customer = customer
                        best_position = i
                        best_route = temp_route
                        best_arrival_times = temp_arrival_times

            # If we found a feasible insertion, update the route
            if best_customer is not None:
                enhanced_route = best_route
                customer_group.remove(best_customer)
                unvisited_customers.remove(best_customer)  # Also remove from main list
                inserted_customers.append(best_customer)
                customers_inserted = True
                arrival_times = best_arrival_times
                print(f"Inserted customer {best_customer.id} with cost {best_insertion_cost:.4f}")
            else:
                # If this is the must-serve group and we couldn't insert all, try with extended hours
                if customer_group == unvisited_must_serve and customer_group:
                    print("WARNING: Unable to insert all must-serve customers with normal constraints.")
                    print("Extending working hours to force insertion...")

                    # Try again with extended finish time
                    extended_finish_time = driver_finish_time + 120  # Add 2 hours

                    for customer in customer_group[:]:  # Use a copy for safe iteration
                        # Try all positions to find best
                        best_position = None
                        best_insertion_cost = float('inf')
                        best_temp_route = None
                        best_temp_arrival_times = None

                        for i in range(len(enhanced_route) - 1):
                            temp_route = enhanced_route[:i + 1] + [customer] + enhanced_route[i + 1:]
                            temp_arrival_times = estimate_arrival_times(temp_route, driver_start_time)

                            if temp_arrival_times and len(temp_arrival_times) > i + 1:
                                # Calculate insertion cost regardless of working hours
                                total_customer_sat = 0.0
                                for j in range(1, len(temp_route) - 1):
                                    cust = temp_route[j]
                                    arrival_time = temp_arrival_times[j]
                                    satisfaction = calculate_customer_satisfaction(
                                        arrival_time,
                                        cust.time_windows[day_index],
                                        buffer_minutes
                                    )
                                    total_customer_sat += satisfaction

                                temp_customer_satisfaction = total_customer_sat / (len(temp_route) - 2)

                                # Calculate insertion cost with effective alpha
                                temp_driver_satisfaction, _, _ = calculate_driver_satisfaction(
                                    temp_route, day_index, previous_routes, previous_working_times
                                )

                                insertion_cost = (1.0 - temp_customer_satisfaction) * effective_alpha + \
                                                 (1.0 - temp_driver_satisfaction) * (1.0 - effective_alpha)

                                if insertion_cost < best_insertion_cost:
                                    best_insertion_cost = insertion_cost
                                    best_position = i
                                    best_temp_route = temp_route
                                    best_temp_arrival_times = temp_arrival_times

                        if best_temp_route:
                            enhanced_route = best_temp_route
                            arrival_times = best_temp_arrival_times
                            customer_group.remove(customer)
                            unvisited_customers.remove(customer)
                            inserted_customers.append(customer)
                            print(f"  Forced insertion of must-serve customer {customer.id}")
                        else:
                            print(
                                f"  CRITICAL: Could not insert must-serve customer {customer.id} even with extended hours!")

                # For optional customers, just break the loop if no more can be inserted
                break

    # Double-check that all must-serve customers are in the route
    all_customers_ids = set(c.id for c in enhanced_route)
    must_serve_customers = [c for c in unvisited_customers if c.must_serve]

    if must_serve_customers:
        print(f"ERROR: Still have {len(must_serve_customers)} must-serve customers not in route!")
        print(f"Customer IDs: {[c.id for c in must_serve_customers]}")
        # Implement emergency handling if needed

    # If we inserted any customers, log this information
    if inserted_customers:
        print(f"Post-optimization: Successfully inserted {len(inserted_customers)} additional customers:")
        must_serve_count = sum(1 for c in inserted_customers if c.must_serve)
        optional_count = len(inserted_customers) - must_serve_count
        print(f"  - {must_serve_count} must-serve customers")
        print(f"  - {optional_count} optional customers")
        for cust in inserted_customers:
            print(f"  - Customer {cust.id} (Must-serve: {cust.must_serve})")

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

    # Calculate final cost using original alpha (not effective_alpha)
    total_cost = (1.0 - customer_satisfaction) * alpha + (1.0 - driver_satisfaction) * (1.0 - alpha)

    print(f"Final enhanced route has {num_customers} customers with cost {total_cost:.4f}")
    print(f"Customer satisfaction: {customer_satisfaction:.4f}, Driver satisfaction: {driver_satisfaction:.4f}")

    return enhanced_route, total_cost, final_arrival_times, customer_satisfaction, driver_satisfaction