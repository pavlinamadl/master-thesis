from typing import List

from customer_data import Customer, TimeWindow
from constants import (
    DRIVER_START_TIME,
    LUNCH_BREAK_START, LUNCH_BREAK_END,
    SPEED_METERS_PER_MINUTE,
    BUFFER_MINUTES,
    SATISFACTION_STEEP_THRESHOLD
)

from distance_utils import distance


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
