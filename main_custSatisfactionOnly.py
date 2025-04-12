import math
import random
from typing import NamedTuple, List, Tuple
import time

# Import customer data from the customer_data.py file
from customer_data import TimeWindow, Customer, all_customers

# Import constants
from constants import (
    DRIVER_START_TIME, DRIVER_FINISH_TIME,
    BUFFER_MINUTES, SERVICE_TIME_MINUTES,
    LUNCH_BREAK_START, LUNCH_BREAK_END,
    SPEED_METERS_PER_MINUTE, W_CUSTOMER, MAX_ITERATIONS_2OPT
)


# Manhattan distance function + travel time calculation
def distance(c1: Customer, c2: Customer, speed_meters_per_minute: float = SPEED_METERS_PER_MINUTE) -> float:
    """
    Manhattan distance in meters, converted to travel time in minutes.
    """
    manhattan_dist_meters = abs(c1.x - c2.x) + abs(c1.y - c2.y)
    travel_time_minutes = manhattan_dist_meters / speed_meters_per_minute
    return travel_time_minutes


def check_lunch_break_conflict(arrival_time: float, service_time_minutes: float = SERVICE_TIME_MINUTES) -> bool:
    """
    Check if a service at the given arrival time would conflict with the lunch break.
    Returns True if there's a conflict, False otherwise.
    """
    service_end_time = arrival_time + service_time_minutes
    # Check if service starts before lunch and ends during/after lunch
    if arrival_time < LUNCH_BREAK_START and service_end_time > LUNCH_BREAK_START:
        return True
    # Check if service starts during lunch
    if LUNCH_BREAK_START <= arrival_time < LUNCH_BREAK_END:
        return True
    return False


def adjust_for_lunch_break(arrival_time: float) -> float:
    """
    Adjust arrival time if it conflicts with lunch break.
    If arrival time is during lunch break, it's moved to after lunch.
    """
    if LUNCH_BREAK_START <= arrival_time < LUNCH_BREAK_END:
        return LUNCH_BREAK_END
    return arrival_time


def estimate_arrival_times(route: List[Customer], service_time_minutes: float = SERVICE_TIME_MINUTES,
                           start_time_minutes: float = DRIVER_START_TIME) -> List[float]:
    """
    Estimates arrival times, including travel time (Manhattan distance) and
    fixed service time at each customer, respecting the lunch break.
    """
    arrival_times = [start_time_minutes]  # start from driver start time at the depot
    if len(route) <= 1:  # Handle empty or single-depot routes
        return arrival_times

    current_time = start_time_minutes

    for i in range(1, len(route)):
        # Calculate travel time from previous location
        travel_time = distance(route[i - 1], route[i])

        # Update time after travel
        current_time += travel_time

        # Check if we're approaching lunch break
        if current_time < LUNCH_BREAK_START and (current_time + service_time_minutes) > LUNCH_BREAK_START:
            # Service would overlap with start of lunch - wait until after lunch
            current_time = LUNCH_BREAK_END
        # Check if we've arrived during lunch break
        elif LUNCH_BREAK_START <= current_time < LUNCH_BREAK_END:
            # Arrived during lunch break - wait until it's over
            current_time = LUNCH_BREAK_END

        # Record arrival time
        arrival_times.append(current_time)

        # Add service time
        current_time += service_time_minutes

    return arrival_times


def calculate_working_time(route: List[Customer], service_time_minutes: float = SERVICE_TIME_MINUTES) -> float:
    """
    Calculate the total working time (travel time + service time) for a given route,
    using Manhattan distance and fixed service time per customer.
    """
    working_time = 0.0
    if len(route) <= 1:  # Handle empty or single-depot routes
        return working_time

    arrival_times = estimate_arrival_times(route, service_time_minutes)
    if len(arrival_times) > 0:
        # Working time is the time from start to finish, excluding lunch break
        start_time = arrival_times[0]
        end_time = arrival_times[-1] + service_time_minutes  # Add service time for last customer

        working_time = end_time - start_time

        # Subtract lunch break if it falls within the working period
        if start_time <= LUNCH_BREAK_START and end_time >= LUNCH_BREAK_END:
            working_time -= (LUNCH_BREAK_END - LUNCH_BREAK_START)

    return working_time


def calculate_satisfaction(arrival_time: float, time_window: TimeWindow,
                           buffer_minutes: float = BUFFER_MINUTES) -> float:
    """
    Calculates customer satisfaction based on arrival time deviation from the time window.

    Satisfaction model:
    - 1.0 if within the time window
    - Decreases with increasing distance from the window bounds up to buffer_minutes
    - Decreases more quickly when late than when early
    - 0.0 beyond buffer_minutes outside the window
    """
    # Within time window - full satisfaction
    if time_window.start <= arrival_time <= time_window.end:
        return 1.0

    # Early arrival - satisfaction decreases more slowly
    if arrival_time < time_window.start:
        time_deviation = time_window.start - arrival_time
        if time_deviation <= buffer_minutes:
            # Quadratic decrease - slower decrease for small deviations
            return 1.0 - (time_deviation / buffer_minutes) ** 2
        else:
            return 0.0

    # Late arrival - satisfaction decreases more quickly
    if arrival_time > time_window.end:
        time_deviation = arrival_time - time_window.end
        if time_deviation <= buffer_minutes:
            # Linear decrease - faster decrease for being late
            return 1.0 - (time_deviation / buffer_minutes)
        else:
            return 0.0


def insertion_heuristic(
        customers: List[Customer],
        day_index: int,
        w_cust: float = W_CUSTOMER,
        driver_start_time: float = DRIVER_START_TIME,
        driver_finish_time: float = DRIVER_FINISH_TIME,
        buffer_minutes: float = BUFFER_MINUTES,
        service_time_minutes: float = SERVICE_TIME_MINUTES
) -> Tuple[List[Customer], float, List[float]]:
    """
    Build a route for day_index using insertion heuristic, optimizing for customer satisfaction.
    All customers must be served, regardless of working time constraints.
    """
    depot = customers[0]
    unvisited = list(customers[1:])  # exclude depot
    route = [depot, depot]  # start and end at depot

    while unvisited:
        best_insertion_cost = float('inf')
        best_customer = None
        best_position = None

        # Current route evaluation
        current_arrival_times = estimate_arrival_times(route, service_time_minutes, driver_start_time)

        for c in unvisited:
            for i in range(len(route) - 1):
                # Calculate distance cost
                old_dist = distance(route[i], route[i + 1])
                new_dist = distance(route[i], c) + distance(c, route[i + 1])
                delta_dist = new_dist - old_dist

                # Create temporary route with customer c inserted
                temp_route = route[:i + 1] + [c] + route[i + 1:]

                # Estimate arrival times for this temporary route
                temp_arrival_times = estimate_arrival_times(temp_route, service_time_minutes, driver_start_time)

                # Skip if arrival times can't be calculated
                if not temp_arrival_times or len(temp_arrival_times) <= i + 1:
                    continue

                arrival_time_at_c = temp_arrival_times[i + 1]

                # We've removed the working time constraint check since we need to serve all customers

                # Calculate satisfaction based on time window
                satisfaction = calculate_satisfaction(
                    arrival_time_at_c,
                    c.time_windows[day_index],
                    buffer_minutes
                )

                # Convert satisfaction to cost (1 - satisfaction)
                satisfaction_cost = 1.0 - satisfaction

                # Calculate total insertion cost
                insertion_cost = delta_dist + w_cust * satisfaction_cost

                if insertion_cost < best_insertion_cost:
                    best_insertion_cost = insertion_cost
                    best_customer = c
                    best_position = i

        if best_customer is None:
            # No feasible insertion found (shouldn't happen now that we've removed constraints)
            break

        # Insert the best customer
        route.insert(best_position + 1, best_customer)
        unvisited.remove(best_customer)

    # Calculate final arrival times and objective value
    final_arrival_times = estimate_arrival_times(route, service_time_minutes, driver_start_time)

    # Calculate final objective value (total customer satisfaction cost)
    total_satisfaction_cost = 0.0
    for i in range(1, len(route) - 1):  # Skip depot
        customer = route[i]
        arrival_time = final_arrival_times[i]
        satisfaction = calculate_satisfaction(
            arrival_time,
            customer.time_windows[day_index],
            buffer_minutes
        )
        total_satisfaction_cost += (1.0 - satisfaction)

    return route, total_satisfaction_cost, final_arrival_times


def two_opt_improvement(
        route: List[Customer],
        day_index: int,
        w_cust: float = W_CUSTOMER,
        driver_start_time: float = DRIVER_START_TIME,
        buffer_minutes: float = BUFFER_MINUTES,
        service_time_minutes: float = SERVICE_TIME_MINUTES,
        max_iterations: int = MAX_ITERATIONS_2OPT
) -> Tuple[List[Customer], float, List[float]]:
    """
    Improves a route using 2-opt local search algorithm.
    """
    if len(route) <= 3:  # Not enough nodes for 2-opt
        final_arrival_times = estimate_arrival_times(route, service_time_minutes, driver_start_time)
        return route, 0.0, final_arrival_times

    # Calculate initial route cost
    best_route = route.copy()
    arrival_times = estimate_arrival_times(best_route, service_time_minutes, driver_start_time)

    # Calculate initial total cost
    total_distance = 0.0
    for i in range(len(best_route) - 1):
        total_distance += distance(best_route[i], best_route[i + 1])

    # Calculate initial satisfaction cost
    total_satisfaction_cost = 0.0
    for i in range(1, len(best_route) - 1):  # Skip depot
        customer = best_route[i]
        arrival_time = arrival_times[i]
        satisfaction = calculate_satisfaction(
            arrival_time,
            customer.time_windows[day_index],
            buffer_minutes
        )
        total_satisfaction_cost += (1.0 - satisfaction)

    best_cost = total_distance + w_cust * total_satisfaction_cost
    improvement_found = True
    iteration = 0

    while improvement_found and iteration < max_iterations:
        improvement_found = False
        iteration += 1

        # Try all possible 2-opt swaps, excluding the depot (index 0 and last index)
        for i in range(1, len(best_route) - 2):
            for j in range(i + 1, len(best_route) - 1):
                if j - i == 1:
                    continue  # Skip adjacent edges

                # Create new route with 2-opt swap
                new_route = best_route.copy()
                # Reverse the segment between i and j
                new_route[i:j + 1] = reversed(best_route[i:j + 1])

                # Calculate new arrival times
                new_arrival_times = estimate_arrival_times(new_route, service_time_minutes, driver_start_time)

                # Calculate new total distance
                new_total_distance = 0.0
                for k in range(len(new_route) - 1):
                    new_total_distance += distance(new_route[k], new_route[k + 1])

                # Calculate new satisfaction cost
                new_satisfaction_cost = 0.0
                for k in range(1, len(new_route) - 1):  # Skip depot
                    customer = new_route[k]
                    arrival_time = new_arrival_times[k]
                    satisfaction = calculate_satisfaction(
                        arrival_time,
                        customer.time_windows[day_index],
                        buffer_minutes
                    )
                    new_satisfaction_cost += (1.0 - satisfaction)

                new_cost = new_total_distance + w_cust * new_satisfaction_cost

                # If the new route is better, keep it
                if new_cost < best_cost:
                    best_route = new_route
                    best_cost = new_cost
                    arrival_times = new_arrival_times
                    improvement_found = True

    # Calculate final objective value (total customer satisfaction cost)
    final_satisfaction_cost = 0.0
    for i in range(1, len(best_route) - 1):  # Skip depot
        customer = best_route[i]
        arrival_time = arrival_times[i]
        satisfaction = calculate_satisfaction(
            arrival_time,
            customer.time_windows[day_index],
            buffer_minutes
        )
        final_satisfaction_cost += (1.0 - satisfaction)

    return best_route, final_satisfaction_cost, arrival_times


def main():
    # Use customers imported from customer_data.py
    customers_list = all_customers

    print(f"Loaded {len(customers_list) - 1} customers and depot from customer_data.py")

    # Run for 5 days (Mon-Fri)
    all_routes = []
    all_objective_values = []
    all_arrival_times = []

    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

    for day_idx in range(5):
        print(f"\n----- Processing {days[day_idx]} -----")

        # Track execution time
        start_time_execution = time.time()

        # Step 1: Create initial route using insertion heuristic
        print("Generating initial route using insertion heuristic...")
        initial_route, initial_cost, _ = insertion_heuristic(
            customers=customers_list,
            day_index=day_idx
        )

        # Step 2: Improve route using 2-opt
        print("Improving route using 2-opt local search...")
        improved_route, improved_cost, arrival_times = two_opt_improvement(
            route=initial_route,
            day_index=day_idx
        )

        # Calculate execution time
        execution_time = time.time() - start_time_execution

        # Compare results
        print(f"Initial route cost: {initial_cost:.2f}")
        print(f"Improved route cost: {improved_cost:.2f}")
        if improved_cost < initial_cost:
            print(f"Improvement: {(initial_cost - improved_cost) / initial_cost * 100:.2f}%")
        else:
            print("No improvement found through 2-opt")

        print(f"Execution time: {execution_time:.2f} seconds")
        print(f"Customers served: {len(improved_route) - 2}")  # -2 for depot at start and end

        # Store results
        all_routes.append(improved_route)
        all_objective_values.append(improved_cost)
        all_arrival_times.append(arrival_times)

    # Print detailed results
    print("\n----- FINAL RESULTS -----")
    total_deviation = 0
    total_customers_served = 0

    for d in range(5):
        print(f"\n----- {days[d]} -----")
        print(f"{days[d]} route: {[c.id for c in all_routes[d]]}")

        working_time = calculate_working_time(all_routes[d])
        print(f"{days[d]} working time: {working_time:.2f} minutes")
        print(f"{days[d]} objective value: {all_objective_values[d]:.2f}")

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

        # Print customer service details
        print("\nCustomer Service Details:")
        total_satisfaction = 0.0

        for i in range(1, len(all_routes[d]) - 1):  # Skip depot
            customer = all_routes[d][i]
            arrival_time = all_arrival_times[d][i]
            time_window = customer.time_windows[d]
            satisfaction = calculate_satisfaction(arrival_time, time_window)
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