from typing import List
from additional_customers import Customer, TimeWindow
from constants import (
    DRIVER_START_TIME,
    LUNCH_BREAK_START, LUNCH_BREAK_END,
    SPEED_METERS_PER_MINUTE,
    BUFFER_MINUTES,
    SATISFACTION_STEEP_THRESHOLD
)
from distance_utils import distance

def check_lunch_break_conflict(arrival_time: float) -> bool: #check conflict with lunch break
    return LUNCH_BREAK_START <= arrival_time < LUNCH_BREAK_END

def adjust_for_lunch_break(time_value: float) -> float: #adjust for lunch break
    if LUNCH_BREAK_START <= time_value < LUNCH_BREAK_END:
        return LUNCH_BREAK_END
    return time_value

def estimate_arrival_times( #estimate arrival times
        route: List[Customer],
        driver_start_time: float = DRIVER_START_TIME
) -> List[float]:
    if not route:
        return []
    arrival_times = [driver_start_time]  #start time at depot
    current_time = driver_start_time
    for i in range(1, len(route)): #travel time from previous to current customer
        dist = distance(route[i - 1], route[i])
        travel_time = dist / SPEED_METERS_PER_MINUTE
        current_time += travel_time #update the current time after travel
        current_time = adjust_for_lunch_break(current_time) #check for lunch break
        arrival_times.append(current_time) #add to arrival time list
        if i < len(route) - 1: #add service time if not in depot
            service_time = route[i].service_time
            current_time += service_time
    return arrival_times


def calculate_customer_satisfaction(
        arrival_time: float,
        time_window: TimeWindow,
        buffer_minutes: float = BUFFER_MINUTES
) -> float:
    if time_window.start <= arrival_time <= time_window.end:
        return 1.0 #perfect satisfaction
    if arrival_time < time_window.start:
        deviation = time_window.start - arrival_time
    else:  deviation = arrival_time - time_window.end # arrival_time > time_window.end

    if deviation <= buffer_minutes:     #small deviation (within buffer)
        return 1.0 - (deviation / buffer_minutes) * 0.5
    else:
        base_satisfaction = 0.25 #steep
        extra_deviation = deviation - SATISFACTION_STEEP_THRESHOLD
        return base_satisfaction / (1.0 + 0.1 * extra_deviation)
