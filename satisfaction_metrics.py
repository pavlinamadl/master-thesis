from typing import List, Tuple, Set

from additional_customers import Customer
from constants import (
    LUNCH_BREAK_END, LUNCH_BREAK_START,
    SPEED_METERS_PER_MINUTE,
    IDEAL_WORKING_HOURS, MAX_WORKING_HOURS,
    WORK_TIME_WEIGHT, ROUTE_CONSISTENCY_WEIGHT
)
from distance_utils import distance
from time_utils import calculate_customer_satisfaction

def calculate_working_time(route: List[Customer]) -> float: #calculates working time in minutes
    total_distance = 0.0
    for i in range(1, len(route)):
        total_distance += distance(route[i - 1], route[i])
    travel_time = total_distance / SPEED_METERS_PER_MINUTE
    service_time = sum(customer.service_time for customer in route[1:-1]) #add the service time for all customers (excluding depot)
    lunch_duration = LUNCH_BREAK_END - LUNCH_BREAK_START #add lunch break
    return travel_time + service_time + lunch_duration

def calculate_work_time_satisfaction(working_time: float) -> float: #calculate worktime satisfaction
    working_hours = working_time / 60.0 #convert to hours
    if working_hours <= IDEAL_WORKING_HOURS: #if working time is less than ideal, satisfaction decreases linearly
        return 0.5 + (working_hours / IDEAL_WORKING_HOURS) * 0.5 #from 0.5 to 1.0

    elif working_hours <= MAX_WORKING_HOURS: #if working time is between ideal and max, linearly decreases
        overwork_ratio = (working_hours - IDEAL_WORKING_HOURS) / (MAX_WORKING_HOURS - IDEAL_WORKING_HOURS)
        return 1.0 - overwork_ratio * 0.5 #from 1.0 to 0.5

    else:   #if exceeds max hours, decreases more steeply (approaches 0 asymptotically)
        excess_hours = working_hours - MAX_WORKING_HOURS
        return 0.5 / (1.0 + excess_hours)

def calculate_work_time_consistency( #calculates consistency in working time across days
        working_time: float,
        previous_working_times: List[float]
) -> float:
    if not previous_working_times:
        return 1.0  #if no previous days, perfect consistency
    avg_previous_time = sum(previous_working_times) / len(previous_working_times) #avg of previous working times
    deviation_percentage = abs(working_time - avg_previous_time) / avg_previous_time #deviation

    if deviation_percentage <= 0.2: #0% deviation gives 1.0 satisfaction,20% deviation gives 0.5 satisfaction
        return 1.0 - deviation_percentage * 2.5
    else: return max(0.5 - (deviation_percentage - 0.2), 0.0) #d> 20%, decreases more slowly

def calculate_route_consistency( #how many customers from previous route are also in the current route
        route: List[Customer],
        previous_routes: List[List[Customer]]
) -> float:
    if not previous_routes:
        return 1.0  #if no previous days, perfect consistency
    current_ids = {customer.id for customer in route if customer.id != 0} #extract customer IDs from current route
    consistencies = [] #calculate consistency with each previous route
    for prev_route in previous_routes:
        prev_ids = {customer.id for customer in prev_route if customer.id != 0}
        intersection = len(current_ids.intersection(prev_ids)) #Jaccard similarity (intersection over union)
        union = len(current_ids.union(prev_ids))
        consistencies.append(intersection / union)
    return sum(consistencies) / len(consistencies) #average consistency across all previous routes

def calculate_driver_satisfaction( #driver sat. calculation
        route: List[Customer],
        day_index: int,
        previous_routes: List[List[Customer]],
        previous_working_times: List[float],
        work_time_weight: float = WORK_TIME_WEIGHT,
        route_consistency_weight: float = ROUTE_CONSISTENCY_WEIGHT
) -> Tuple[float, float, float]:
    working_time = calculate_working_time(route) #working time for the route
    work_time_sat = calculate_work_time_satisfaction(working_time) #work time satisfaction
    work_time_consistency = calculate_work_time_consistency(working_time, previous_working_times) #work time consistency with previous days
    route_consistency = calculate_route_consistency(route, previous_routes) #route consistency with previous days
    consistency_sat = work_time_consistency * work_time_weight + route_consistency * route_consistency_weight #combine consistency satisfactions
    overall_sat = (work_time_sat + consistency_sat) / 2.0 #overall driver satisfaction = average
    return overall_sat, work_time_sat, consistency_sat
