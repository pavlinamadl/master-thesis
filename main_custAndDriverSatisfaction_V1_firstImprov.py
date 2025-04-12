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
    SPEED_METERS_PER_MINUTE, W_CUSTOMER, MAX_ITERATIONS_2OPT,
    IDEAL_WORKING_HOURS, MAX_WORKING_HOURS,
    WORK_TIME_WEIGHT, ROUTE_CONSISTENCY_WEIGHT, W_DRIVER
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
    Returns working time in minutes.
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


def calculate_customer_satisfaction(arrival_time: float, time_window: TimeWindow,
                                    buffer_minutes: float = BUFFER_MINUTES,
                                    steep_threshold: float = SATISFACTION_STEEP_THRESHOLD) -> float:
    """
    Calculates customer satisfaction based on arrival time deviation from the time window.

    Satisfaction model:
    - 1.0 if within the time window
    - For early arrivals:
      - Only marginally decreased within ±15 minutes
      - Then decreases more according to quadratic function up to buffer_minutes
    - For late arrivals:
      - Only marginally decreased within ±15 minutes
      - Then decreases more quickly (linearly) up to buffer_minutes
    - 0.0 beyond buffer_minutes outside the window
    """
    # Within time window - full satisfaction
    if time_window.start <= arrival_time <= time_window.end:
        return 1.0

    # Early arrival
    if arrival_time < time_window.start:
        time_deviation = time_window.start - arrival_time

        # Within steep threshold - only marginal decrease
        if time_deviation <= steep_threshold:
            # Very minor decrease (0.05 at the threshold)
            return 1.0 - (0.05 * time_deviation / steep_threshold)

        # Beyond threshold but within buffer - quadratic decrease
        elif time_deviation <= buffer_minutes:
            # Start from 0.95 at the threshold and decrease quadratically to 0 at buffer
            threshold_satisfaction = 0.95
            remaining_deviation = time_deviation - steep_threshold
            remaining_range = buffer_minutes - steep_threshold

            # Quadratic decrease for remaining deviation
            return threshold_satisfaction - threshold_satisfaction * (remaining_deviation / remaining_range) ** 2

        # Beyond buffer - zero satisfaction
        else:
            return 0.0

    # Late arrival
    if arrival_time > time_window.end:
        time_deviation = arrival_time - time_window.end

        # Within steep threshold - only marginal decrease
        if time_deviation <= steep_threshold:
            # Minor decrease (0.1 at the threshold)
            return 1.0 - (0.1 * time_deviation / steep_threshold)

        # Beyond threshold but within buffer - linear decrease
        elif time_deviation <= buffer_minutes:
            # Start from 0.9 at the threshold and decrease linearly to 0 at buffer
            threshold_satisfaction = 0.9
            remaining_deviation = time_deviation - steep_threshold
            remaining_range = buffer_minutes - steep_threshold

            # Linear decrease for remaining deviation
            return threshold_satisfaction - threshold_satisfaction * (remaining_deviation / remaining_range)

        # Beyond buffer - zero satisfaction
        else:
            return 0.0


def calculate_work_time_satisfaction(working_time_minutes: float) -> float:
    """
    Calculate driver satisfaction based on working time.
    - 1.0 if working exactly ideal hours (8.5 hours including lunch)
    - Decreases as working time deviates from ideal
    - Severe decrease if working more than max hours (9.5 hours including lunch)

    Returns value between 0.0 and 1.0
    """
    # Convert to hours for easier reasoning
    working_hours = working_time_minutes / 60.0
    ideal_hours = IDEAL_WORKING_HOURS
    max_hours = MAX_WORKING_HOURS

    # Perfect satisfaction at ideal working hours
    if working_hours == ideal_hours:
        return 1.0

    # Working too little (less than ideal)
    if working_hours < ideal_hours:
        # Linear decrease: 20% reduction per hour below ideal
        shortfall = ideal_hours - working_hours
        return max(0.0, 1.0 - 0.2 * shortfall)

    # Working more than ideal but less than max
    if ideal_hours < working_hours <= max_hours:
        # Linear decrease: 50% reduction per hour above ideal, up to max
        excess = working_hours - ideal_hours
        return max(0.0, 1.0 - 0.5 * excess)

    # Working beyond max hours
    if working_hours > max_hours:
        # Severe decrease: 0.5 at max hours, then quickly down to 0
        excess_beyond_max = working_hours - max_hours
        # Starting from 0.5 at max hours, decrease to 0 within one additional hour
        base_satisfaction = 0.5
        return max(0.0, base_satisfaction - 0.5 * excess_beyond_max)


def calculate_work_time_consistency(current_time: float, previous_times: List[float]) -> float:
    """
    Calculate how consistent the current working time is with previous days.
    - 1.0 if exactly matching the average of previous days
    - Decreases as deviation from previous days increases

    Returns value between 0.0 and 1.0
    """
    if not previous_times:  # No previous days to compare with
        return 1.0

    # Calculate average working time of previous days
    avg_previous_time = sum(previous_times) / len(previous_times)

    # Calculate the deviation as percentage of the average
    deviation_percent = abs(current_time - avg_previous_time) / avg_previous_time

    # Satisfaction based on deviation
    # 20% deviation or more results in 0 satisfaction
    if deviation_percent >= 0.2:
        return 0.0

    # Linear decrease from 1.0 to 0.0 as deviation increases from 0% to 20%
    return 1.0 - (deviation_percent / 0.2)


def calculate_route_consistency(current_route: List[Customer], previous_routes: List[List[Customer]]) -> float:
    """
    Calculate how consistent the current route is with previous days.
    Based on common segments (pairs of consecutive customers) between routes.

    Returns value between 0.0 and 1.0
    """
    if not previous_routes:  # No previous days to compare with
        return 1.0

    # Extract customer IDs from current route (excluding depot)
    current_route_ids = [c.id for c in current_route[1:-1]]

    # Create segments from current route (pairs of consecutive customers)
    current_segments = set()
    for i in range(len(current_route_ids) - 1):
        segment = (current_route_ids[i], current_route_ids[i + 1])
        current_segments.add(segment)

    # If current route has no segments, return 1.0
    if not current_segments:
        return 1.0

    # Calculate similarity with each previous route
    similarities = []
    for prev_route in previous_routes:
        # Extract customer IDs from previous route (excluding depot)
        prev_route_ids = [c.id for c in prev_route[1:-1]]

        # Create segments from previous route
        prev_segments = set()
        for i in range(len(prev_route_ids) - 1):
            segment = (prev_route_ids[i], prev_route_ids[i + 1])
            prev_segments.add(segment)

        # Skip if previous route has no segments
        if not prev_segments:
            continue

        # Calculate Jaccard similarity (intersection over union)
        common_segments = current_segments.intersection(prev_segments)
        all_segments = current_segments.union(prev_segments)

        if all_segments:
            similarity = len(common_segments) / len(all_segments)
            similarities.append(similarity)

    # Return average similarity with previous routes, or 1.0 if no valid comparisons
    return sum(similarities) / len(similarities) if similarities else 1.0


def calculate_driver_satisfaction(
        route: List[Customer],
        day_index: int,
        previous_routes: List[List[Customer]],
        previous_working_times: List[float]
) -> Tuple[float, float, float]:
    """
    Calculate driver satisfaction based on:
    1. Working time (how close to ideal hours)
    2. Working time consistency (compared to previous days)
    3. Route consistency (compared to previous days)

    Returns:
    - Overall driver satisfaction (0.0 to 1.0)
    - Work time component (0.0 to 1.0)
    - Route consistency component (0.0 to 1.0)
    """
    # Calculate working time
    working_time = calculate_working_time(route)

    # Calculate satisfaction with working time duration
    work_time_sat = calculate_work_time_satisfaction(working_time)

    # Get previous routes and times for this day of the week
    prev_routes_same_day = []
    prev_times_same_day = []

    # Only use previous days of the same type (e.g., only compare Mondays with Mondays)
    for i in range(len(previous_routes)):
        if i % 5 == day_index % 5:  # Same day of the week
            prev_routes_same_day.append(previous_routes[i])
            prev_times_same_day.append(previous_working_times[i])

    # Calculate satisfaction with working time consistency
    work_time_consistency = calculate_work_time_consistency(working_time, prev_times_same_day)

    # Calculate satisfaction with route consistency
    route_consistency = calculate_route_consistency(route, prev_routes_same_day)

    # Combine components with weights
    work_component = (work_time_sat + work_time_consistency) / 2
    overall_satisfaction = (WORK_TIME_WEIGHT * work_component +
                            ROUTE_CONSISTENCY_WEIGHT * route_consistency)

    return overall_satisfaction, work_component, route_consistency


def insertion_heuristic(
        customers: List[Customer],
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
    Build a route for day_index using insertion heuristic, optimizing for:
    - Customer satisfaction (time window preferences)
    - Driver satisfaction (consistent working hours and routes)

    Returns:
    - The constructed route
    - The objective value
    - The arrival times
    - Customer satisfaction component
    - Driver satisfaction component
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
                route_finish_time = temp_arrival_times[-1]

                # The only constraint we check is if the route finishes by the end of working day
                if route_finish_time > driver_finish_time:
                    continue

                # Calculate customer satisfaction based on time window
                customer_satisfaction = calculate_customer_satisfaction(
                    arrival_time_at_c,
                    c.time_windows[day_index],
                    buffer_minutes
                )

                # Calculate driver satisfaction
                driver_satisfaction, _, _ = calculate_driver_satisfaction(
                    temp_route, day_index, previous_routes, previous_working_times
                )

                # Convert satisfaction to cost
                customer_cost = 1.0 - customer_satisfaction
                driver_cost = 1.0 - driver_satisfaction

                # Calculate total insertion cost
                insertion_cost = delta_dist + w_cust * customer_cost + w_driver * driver_cost

                if insertion_cost < best_insertion_cost:
                    best_insertion_cost = insertion_cost
                    best_customer = c
                    best_position = i

        if best_customer is None:
            # No feasible insertion found
            break

        # Insert the best customer
        route.insert(best_position + 1, best_customer)
        unvisited.remove(best_customer)

    # Calculate final arrival times
    final_arrival_times = estimate_arrival_times(route, service_time_minutes, driver_start_time)

    # Calculate final customer satisfaction
    total_customer_sat = 0.0
    num_customers = len(route) - 2  # excluding depot at start and end

    if num_customers > 0:
        for i in range(1, len(route) - 1):  # Skip depot
            customer = route[i]
            arrival_time = final_arrival_times[i]
            satisfaction = calculate_customer_satisfaction(
                arrival_time,
                customer.time_windows[day_index],
                buffer_minutes
            )
            total_customer_sat += satisfaction

        # Normalize customer satisfaction (0 to 1 scale)
        customer_satisfaction = total_customer_sat / num_customers
    else:
        customer_satisfaction = 1.0  # Default for empty route

    # Calculate final driver satisfaction
    driver_satisfaction, _, _ = calculate_driver_satisfaction(
        route, day_index, previous_routes, previous_working_times
    )

    # Calculate total objective value (combined cost)
    customer_cost = 1.0 - customer_satisfaction
    driver_cost = 1.0 - driver_satisfaction
    total_cost = customer_cost * w_cust + driver_cost * w_driver

    return route, total_cost, final_arrival_times, customer_satisfaction, driver_satisfaction


def two_opt_improvement(
        route: List[Customer],
        day_index: int,
        previous_routes: List[List[Customer]],
        previous_working_times: List[float],
        w_cust: float = W_CUSTOMER,
        w_driver: float = W_DRIVER,
        driver_start_time: float = DRIVER_START_TIME,
        buffer_minutes: float = BUFFER_MINUTES,
        service_time_minutes: float = SERVICE_TIME_MINUTES,
        max_iterations: int = MAX_ITERATIONS_2OPT
) -> Tuple[List[Customer], float, List[float], float, float]:
    """
    Improves a route using 2-opt local search algorithm,
    optimizing for both customer and driver satisfaction.

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

    # Calculate initial arrival times
    best_route = route.copy()
    arrival_times = estimate_arrival_times(best_route, service_time_minutes, driver_start_time)

    # Calculate initial customer satisfaction
    total_customer_sat = 0.0
    num_customers = len(best_route) - 2  # excluding depot at start and end

    if num_customers > 0:
        for i in range(1, len(best_route) - 1):  # Skip depot
            customer = best_route[i]
            arrival_time = arrival_times[i]
            satisfaction = calculate_customer_satisfaction(
                arrival_time,
                customer.time_windows[day_index],
                buffer_minutes
            )
            total_customer_sat += satisfaction

        best_customer_satisfaction = total_customer_sat / num_customers
    else:
        best_customer_satisfaction = 1.0  # Default for empty route

    # Calculate initial driver satisfaction
    best_driver_satisfaction, _, _ = calculate_driver_satisfaction(
        best_route, day_index, previous_routes, previous_working_times
    )

    # Calculate initial total cost
    best_cost = (1.0 - best_customer_satisfaction) * w_cust + (1.0 - best_driver_satisfaction) * w_driver

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

                # Check if the route finishes by the end of working day
                if new_arrival_times[-1] > DRIVER_FINISH_TIME:
                    continue

                # Calculate new customer satisfaction
                new_total_customer_sat = 0.0
                for k in range(1, len(new_route) - 1):  # Skip depot
                    customer = new_route[k]
                    arrival_time = new_arrival_times[k]
                    satisfaction = calculate_customer_satisfaction(
                        arrival_time,
                        customer.time_windows[day_index],
                        buffer_minutes
                    )
                    new_total_customer_sat += satisfaction

                new_customer_satisfaction = new_total_customer_sat / num_customers if num_customers > 0 else 1.0

                # Calculate new driver satisfaction
                new_driver_satisfaction, _, _ = calculate_driver_satisfaction(
                    new_route, day_index, previous_routes, previous_working_times
                )

                # Calculate new total cost
                new_cost = (1.0 - new_customer_satisfaction) * w_cust + (1.0 - new_driver_satisfaction) * w_driver

                # If the new route is better, keep it
                if new_cost < best_cost:
                    best_route = new_route
                    best_cost = new_cost
                    arrival_times = new_arrival_times
                    best_customer_satisfaction = new_customer_satisfaction
                    best_driver_satisfaction = new_driver_satisfaction
                    improvement_found = True

    return best_route, best_cost, arrival_times, best_customer_satisfaction, best_driver_satisfaction


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

        # Step 2: Improve route using 2-opt
        print("Improving route using 2-opt local search...")
        improved_route, improved_cost, arrival_times, customer_sat, driver_sat = two_opt_improvement(
            route=initial_route,
            day_index=day_idx,
            previous_routes=previous_routes,
            previous_working_times=previous_working_times
        )

        # Calculate working time for this route
        working_time = calculate_working_time(improved_route)

        # Calculate execution time
        execution_time = time.time() - start_time_execution

        # Compare results
        print(f"Initial route cost: {initial_cost:.2f}")
        print(f"Improved route cost: {improved_cost:.2f}")
        if improved_cost < initial_cost:
            print(f"Improvement: {(initial_cost - improved_cost) / initial_cost * 100:.2f}%")
        else:
            print("No improvement found through 2-opt")

        print(f"Customer satisfaction: {customer_sat:.2f}")
        print(f"Driver satisfaction: {driver_sat:.2f}")
        print(f"Working time: {working_time:.2f} minutes ({working_time / 60:.2f} hours)")
        print(f"Execution time: {execution_time:.2f} seconds")
        print(f"Customers served: {len(improved_route) - 2}")  # -2 for depot at start and end

        # Store results
        all_routes.append(improved_route)
        all_objective_values.append(improved_cost)
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
