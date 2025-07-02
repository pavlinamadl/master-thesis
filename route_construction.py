from typing import List, Tuple, Set
from additional_customers import Customer, is_must_serve_for_day

from constants import (
    DRIVER_START_TIME, DRIVER_FINISH_TIME,
    BUFFER_MINUTES,
    ALPHA,
    EDGE_CONSISTENCY_BONUS,
    MUST_SERVE_PRIORITY,
    EXTENDED_HOURS_MINUTES
)

from time_utils import estimate_arrival_times, calculate_customer_satisfaction
from satisfaction_metrics import calculate_driver_satisfaction

#route made by insertion heuristics considering customer and driver satisfaction
#additional consideration for edge consistency with previous routes

def insertion_heuristic(
        customers: List[Customer],
        day_index: int,
        previous_routes: List[List[Customer]],
        previous_working_times: List[float],
        alpha: float = ALPHA,
        edge_consistency_bonus: float = EDGE_CONSISTENCY_BONUS,
        driver_start_time: float = DRIVER_START_TIME,
        driver_finish_time: float = DRIVER_FINISH_TIME,
        buffer_minutes: float = BUFFER_MINUTES
) -> Tuple[List[Customer], float, List[float], float, float]:
    depot = customers[0]
    route = [depot, depot]  #start and end at depot
    arrival_times = estimate_arrival_times(route, driver_start_time) #calc. arrival times for empty route
    unrouted = [c for c in customers if c.id != 0] #list of all customers except depot
    must_serve_customers = [c for c in unrouted if is_must_serve_for_day(c, day_index)] #separate must-serve customers
    optional_customers = [c for c in unrouted if not is_must_serve_for_day(c, day_index)]
    previous_edges = set() #extract edges from previous day's route
    if day_index > 0 and previous_routes:
        prev_route = previous_routes[-1]  #most recent route
        for i in range(len(prev_route) - 1):
            previous_edges.add((prev_route[i].id, prev_route[i + 1].id))

    effective_alpha = alpha
    if alpha == 0.0 and day_index == 0: #special case for alpha=0.0 on the first day (Monday)
        effective_alpha = 0.5
    extended_finish_time = driver_finish_time + EXTENDED_HOURS_MINUTES #def. extended working time limit

    for customer_list in [must_serve_customers, optional_customers]:
        customer_type = "must-serve" if customer_list == must_serve_customers else "optional" #insert all must-serve customers, strict +2h enforcement
        while customer_list:
            best_customer = None
            best_position = None
            best_insertion_cost = float('inf')
            best_temp_route = None
            best_temp_arrival_times = None
            best_customer_sat = 0.0
            best_driver_sat = 0.0
            for customer in customer_list: #try each possible insertion position
                for i in range(1, len(route)):
                    temp_route = route[:i] + [customer] + route[i:]
                    temp_arrival_times = estimate_arrival_times(temp_route, driver_start_time) #calculate new arrival times
                    if customer_list == must_serve_customers:
                        if temp_arrival_times[-1] > extended_finish_time: #must serve:allow up to +2h extension but enforce the limit
                            continue  #skip; exceeds +2h limit
                    else: #optional: must fit within normal working hours
                        if temp_arrival_times[-1] > driver_finish_time:
                            continue

                    tw = customer.time_windows[day_index]
                    arrival = temp_arrival_times[i]
                    if not (tw.start <= arrival <= tw.end): continue

                    num_customers = len(temp_route) - 2  #excluding depot, calculate customer satisfaction
                    total_customer_sat = 0.0
                    for j in range(1, len(temp_route) - 1):  #skip depot
                        cust = temp_route[j]
                        arr_time = temp_arrival_times[j]
                        sat = calculate_customer_satisfaction(
                                arr_time,
                                cust.time_windows[day_index],
                                buffer_minutes)
                        total_customer_sat += sat
                    customer_satisfaction = total_customer_sat / num_customers

                    driver_satisfaction, _, _ = calculate_driver_satisfaction( #calculate base driver satisfaction
                        temp_route, day_index, previous_routes, previous_working_times)

                    edge_bonus = 0.0 #edge consistency bonus (after monday)
                    if previous_edges and day_index > 0:
                        if (temp_route[i - 1].id, customer.id) in previous_edges:
                            edge_bonus += edge_consistency_bonus
                        if (customer.id, temp_route[i + 1].id) in previous_edges:
                            edge_bonus += edge_consistency_bonus
                        base_cost = (
                                (1.0 - customer_satisfaction) * effective_alpha
                                + (1.0 - driver_satisfaction) * (1.0 - effective_alpha))
                        edge_reward = edge_bonus * (1.0 - effective_alpha)
                        insertion_cost = base_cost - edge_reward
                        if customer.must_serve: #must-serve priority
                            insertion_cost -= MUST_SERVE_PRIORITY
                        insertion_cost = max(0.0, insertion_cost)
                    else: #Monday - use the effective alpha
                        insertion_cost = (
                                (1.0 - customer_satisfaction) * effective_alpha +
                                (1.0 - driver_satisfaction) * (1.0 - effective_alpha))
                        if customer.must_serve: insertion_cost -= MUST_SERVE_PRIORITY
                        insertion_cost = max(0.0, insertion_cost)

                    if insertion_cost < best_insertion_cost: #update best insertion if is better
                        best_insertion_cost = insertion_cost
                        best_customer = customer
                        best_position = i
                        best_temp_route = temp_route
                        best_temp_arrival_times = temp_arrival_times
                        best_customer_sat = customer_satisfaction
                        best_driver_sat = driver_satisfaction
            if best_customer: #update the route
                route = best_temp_route
                customer_list.remove(best_customer)
                arrival_times = best_temp_arrival_times
            else:
                if customer_list == must_serve_customers and customer_list:
                    unserved_ids = [c.id for c in customer_list]
                break

    served_ids = {c.id for c in route}
    unserved_must_serve = [c for c in must_serve_customers if c.id not in served_ids]
    unserved_ids = [c.id for c in unserved_must_serve]

    #final metrics
    num_customers = len(route) - 2  #excluding depot
    total_customer_sat = 0.0
    for i in range(1, len(route) - 1):  # Skip depot
        customer = route[i]
        arr_time = arrival_times[i]
        sat = calculate_customer_satisfaction(
                arr_time,
                customer.time_windows[day_index],
                buffer_minutes)
        total_customer_sat += sat
    customer_satisfaction = total_customer_sat / num_customers
    driver_satisfaction, _, _ = calculate_driver_satisfaction( #final driver satisfaction
        route, day_index, previous_routes, previous_working_times)
    total_cost = (1.0 - customer_satisfaction) * alpha + (1.0 - driver_satisfaction) * (1.0 - alpha) #total cost using alpha for consistency
    return route, total_cost, arrival_times, customer_satisfaction, driver_satisfaction