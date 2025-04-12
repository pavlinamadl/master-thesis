import math
import random
import numpy as np
from typing import NamedTuple, List, Tuple, Dict, Set
import time

# Import customer data from the customer_data.py file
from customer_data import TimeWindow, Customer, all_customers

# Import constants
from constants import (
    DRIVER_START_TIME, DRIVER_FINISH_TIME,
    BUFFER_MINUTES, SERVICE_TIME_MINUTES,
    LUNCH_BREAK_START, LUNCH_BREAK_END,
    SATISFACTION_STEEP_THRESHOLD,
    SPEED_METERS_PER_MINUTE, W_CUSTOMER, W_DRIVER,
    IDEAL_WORKING_HOURS, MAX_WORKING_HOURS,
    WORK_TIME_WEIGHT, ROUTE_CONSISTENCY_WEIGHT,
    MAX_TABU_ITERATIONS, TABU_DIVERSIFICATION_THRESHOLD
)

# Import all the necessary functions from the current file
# (These would be defined in the same file, but for brevity we're referencing them)
from main_custAndDriverSatisfaction_V2_bestImprov import (
    distance, check_lunch_break_conflict, adjust_for_lunch_break,
    estimate_arrival_times, calculate_working_time,
    calculate_customer_satisfaction, calculate_work_time_satisfaction,
    calculate_work_time_consistency, calculate_route_consistency,
    calculate_driver_satisfaction, insertion_heuristic
)


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


def tabu_enhanced_two_opt(
        route: List[Customer],
        day_index: int,
        previous_routes: List[List[Customer]],
        previous_working_times: List[float],
        w_cust: float = W_CUSTOMER,
        w_driver: float = W_DRIVER,
        driver_start_time: float = DRIVER_START_TIME,
        buffer_minutes: float = BUFFER_MINUTES,
        service_time_minutes: float = SERVICE_TIME_MINUTES,
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
        final_arrival_times = estimate_arrival_times(route, service_time_minutes, driver_start_time)

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
    best_arrival_times = estimate_arrival_times(best_route, service_time_minutes, driver_start_time)

    # Calculate number of customers (excluding depot)
    num_customers = len(best_route) - 2

    # Calculate dynamic tabu tenure based on the square root of number of customers
    # Adding 1 to ensure it's at least 1 even with very few customers
    import math
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
            diversified_arrival_times = estimate_arrival_times(diversified_route, service_time_minutes,
                                                               driver_start_time)
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
                neighbor_arrival_times = estimate_arrival_times(neighbor_route, service_time_minutes, driver_start_time)

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
                neighbor_cost = (1.0 - neighbor_customer_satisfaction) * w_cust + (
                            1.0 - neighbor_driver_satisfaction) * w_driver

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

def attempt_additional_insertions(
        route: List[Customer],
        unvisited_customers: List[Customer],
        day_index: int,
        previous_routes: List[List[Customer]],
        previous_working_times: List[float],
        w_cust: float = W_CUSTOMER,
        w_driver: float = W_DRIVER,
        driver_start_time: float = DRIVER_START_TIME,
        driver_finish_time: float = DRIVER_FINISH_TIME,
        buffer_minutes: float = BUFFER_MINUTES,
        service_time_minutes: float = SERVICE_TIME_MINUTES
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
    # If no unvisited customers, return the original route
    if not unvisited_customers:
        # Calculate final arrival times
        final_arrival_times = estimate_arrival_times(route, service_time_minutes, driver_start_time)

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
        current_arrival_times = estimate_arrival_times(enhanced_route, service_time_minutes, driver_start_time)

        # Try each unvisited customer
        for customer in unvisited_customers:
            # Try each possible insertion position
            for i in range(len(enhanced_route) - 1):
                # Create temporary route with customer inserted
                temp_route = enhanced_route[:i + 1] + [customer] + enhanced_route[i + 1:]

                # Estimate arrival times for this temporary route
                temp_arrival_times = estimate_arrival_times(temp_route, service_time_minutes, driver_start_time)

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

                # Calculate insertion cost (objective function value)
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
    final_arrival_times = estimate_arrival_times(enhanced_route, service_time_minutes, driver_start_time)

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

        # Step 1: Create initial route using insertion heuristic
        print("Generating initial route using insertion heuristic...")
        initial_route, initial_cost, _, initial_cust_sat, initial_driver_sat = insertion_heuristic(
            customers=customers_list,
            day_index=day_idx,
            previous_routes=previous_routes,
            previous_working_times=previous_working_times
        )

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
            previous_working_times=previous_working_times
        )

        # Step 3: Attempt to insert additional unvisited customers post-optimization
        print("Attempting to insert additional customers post-optimization...")
        final_route, final_cost, arrival_times, customer_sat, driver_sat = attempt_additional_insertions(
            route=improved_route,
            unvisited_customers=unvisited_customers,
            day_index=day_idx,
            previous_routes=previous_routes,
            previous_working_times=previous_working_times
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

            print(f"  Customer {customer.id}:")
            print(
                f"    Time Window: {window_start_hour:02d}:{window_start_minute:02d}-{window_end_hour:02d}:{window_end_minute:02d}")
            print(f"    Arrival Time: {arrival_hour:02d}:{arrival_minute:02d}")
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