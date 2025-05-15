from typing import List, Tuple, Set

from customer_data import Customer
from constants import (
    LUNCH_BREAK_END, LUNCH_BREAK_START,
    SPEED_METERS_PER_MINUTE,
    IDEAL_WORKING_HOURS, MAX_WORKING_HOURS,
    WORK_TIME_WEIGHT, ROUTE_CONSISTENCY_WEIGHT
)
from distance_utils import distance
from time_utils import calculate_customer_satisfaction


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
