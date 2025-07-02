from typing import List, Tuple, Set
from additional_customers import Customer
from constants import (DRIVER_START_TIME, DRIVER_FINISH_TIME, BUFFER_MINUTES, ALPHA, EDGE_CONSISTENCY_BONUS)
from time_utils import estimate_arrival_times, calculate_customer_satisfaction
from satisfaction_metrics import calculate_driver_satisfaction

def attempt_additional_insertions( #attempts to insert additional unvisited customers into an optimized route
        route: List[Customer],
        unvisited_customers: List[Customer],
        day_index: int,
        previous_routes: List[List[Customer]],
        previous_working_times: List[float],
        alpha: float = ALPHA,
        edge_consistency_bonus: float = EDGE_CONSISTENCY_BONUS,
        driver_start_time: float = DRIVER_START_TIME,
        driver_finish_time: float = DRIVER_FINISH_TIME,
        buffer_minutes: float = BUFFER_MINUTES
) -> Tuple[List[Customer], float, List[float], float, float]:
    effective_alpha = alpha
    if alpha == 0.0 and day_index == 0: #case alpha=0.0 on the first day - use alpha=0.5 temporarily
        effective_alpha = 0.5
    if day_index > 0 and alpha == 0.0: #skip additional insertions on days after Monday when alpha -0, maintaining consistency
        final_arrival_times = estimate_arrival_times(route, driver_start_time)
        num_customers = len(route) - 2 #final customer satisfaction
        customer_satisfaction = 0.0
        total_customer_sat = 0.0
        for i in range(1, len(route) - 1):  # Skip depot
            customer = route[i]
            arrival_time = final_arrival_times[i]
            satisfaction = calculate_customer_satisfaction(
                arrival_time,
                customer.time_windows[day_index],
                buffer_minutes)
            total_customer_sat += satisfaction
        customer_satisfaction = total_customer_sat / num_customers

        driver_satisfaction, _, _ = calculate_driver_satisfaction( #final driver satisfaction
            route, day_index, previous_routes, previous_working_times
        )
        total_cost = (1.0 - customer_satisfaction) * alpha + (1.0 - driver_satisfaction) * (1.0 - alpha) #final cost
        return route, total_cost, final_arrival_times, customer_satisfaction, driver_satisfaction

    must_serve_unvisited = [c for c in unvisited_customers if c.must_serve]
    if must_serve_unvisited: #force insertion of must-serve customers without error
        for customer in must_serve_unvisited:
            best_position = None
            best_insertion_cost = float('inf')
            best_route = None
            best_arrival_times = None
            for i in range(1, len(route)): #try each possible insertion position
                temp_route = route[:i] + [customer] + route[i:]
                temp_arrival_times = estimate_arrival_times(temp_route, driver_start_time)
                if temp_arrival_times: #calculate cost and satisfaction
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
                    temp_driver_satisfaction, _, _ = calculate_driver_satisfaction( #driver satisfaction
                        temp_route, day_index, previous_routes, previous_working_times)
                    insertion_cost = (1.0 - temp_customer_satisfaction) * effective_alpha + \
                                     (1.0 - temp_driver_satisfaction) * (1.0 - effective_alpha) #insertion cost using effective alpha

                    if insertion_cost < best_insertion_cost: #update best insertion if better
                        best_insertion_cost = insertion_cost
                        best_position = i
                        best_route = temp_route
                        best_arrival_times = temp_arrival_times
            if best_route:
                route = best_route
                unvisited_customers.remove(customer)
                arrival_times = best_arrival_times

    if not unvisited_customers: #if no unvisited customers, return the original route
        final_arrival_times = estimate_arrival_times(route, driver_start_time)

        num_customers = len(route) - 2  #final customer satisfaction
        customer_satisfaction = 0.0
        if num_customers > 0:
            total_customer_sat = 0.0
            for i in range(1, len(route) - 1):  # Skip depot
                customer = route[i]
                arrival_time = final_arrival_times[i]
                satisfaction = calculate_customer_satisfaction(
                    arrival_time,
                    customer.time_windows[day_index],
                    buffer_minutes)
                total_customer_sat += satisfaction
            customer_satisfaction = total_customer_sat / num_customers
        driver_satisfaction, _, _ = calculate_driver_satisfaction( #final driver satisfaction
            route, day_index, previous_routes, previous_working_times)
        total_cost = (1.0 - customer_satisfaction) * alpha + (1.0 - driver_satisfaction) * (1.0 - alpha) #final cost
        return route, total_cost, final_arrival_times, customer_satisfaction, driver_satisfaction

    unvisited_must_serve = [c for c in unvisited_customers if c.must_serve] #remaining unvisited customers into must-serve and optional
    unvisited_optional = [c for c in unvisited_customers if not c.must_serve]
    previous_edges = set() #extract edges from previous day
    if day_index > 0 and previous_routes:
        prev_route = previous_routes[-1]
        for k in range(len(prev_route) - 1):
            previous_edges.add((prev_route[k].id, prev_route[k + 1].id))
    enhanced_route = route.copy()
    arrival_times = estimate_arrival_times(enhanced_route, driver_start_time)
    for customer_group in [unvisited_must_serve, unvisited_optional]: #must-serve customers first, then optional customers
        customers_inserted = True
        inserted_customers = []
        while customers_inserted and customer_group:
            customers_inserted = False
            best_insertion_cost = float('inf')
            best_customer = None
            best_position = None
            best_route = None
            best_arrival_times = None
            current_arrival_times = estimate_arrival_times(enhanced_route, driver_start_time) #current route arrival times
            for customer in customer_group: #each unvisited customer, each position
                for i in range(len(enhanced_route) - 1):
                    temp_route = enhanced_route[:i + 1] + [customer] + enhanced_route[i + 1:]
                    temp_arrival_times = estimate_arrival_times(temp_route, driver_start_time) #arrival times for this temporary route
                    if temp_arrival_times[-1] > driver_finish_time and not customer.must_serve: #must-serve customers, exceed finish time if necessary
                        continue
                    total_customer_sat = 0.0 #customer satisfaction
                    for j in range(1, len(temp_route) - 1):
                        cust = temp_route[j]
                        arrival_time = temp_arrival_times[j]
                        satisfaction = calculate_customer_satisfaction(
                            arrival_time,
                            cust.time_windows[day_index],
                            buffer_minutes)
                        total_customer_sat += satisfaction
                    temp_customer_satisfaction = total_customer_sat / (len(temp_route) - 2)
                    temp_driver_satisfaction, _, _ = calculate_driver_satisfaction( #driver satisfaction
                        temp_route, day_index, previous_routes, previous_working_times)
                    edge_bonus = 0.0 #edge consistency bonus
                    if previous_edges and day_index > 0: #if reusing edges from previous day and if new edges being created by this insertion
                        if (temp_route[i].id, customer.id) in previous_edges:
                            edge_bonus += edge_consistency_bonus
                        if (customer.id, temp_route[i + 1].id) in previous_edges:
                            edge_bonus += edge_consistency_bonus
                        insertion_cost = (1.0 - temp_customer_satisfaction) * effective_alpha + \
                                         (1.0 - temp_driver_satisfaction) * (1.0 - effective_alpha) - \
                                         (edge_bonus * (1.0 - effective_alpha))
                    else: #use effective alpha for monday
                        insertion_cost = (1.0 - temp_customer_satisfaction) * effective_alpha + \
                                         (1.0 - temp_driver_satisfaction) * (1.0 - effective_alpha)
                    if customer.must_serve: insertion_cost -= 50.0  ##must-serve customers, additional high priority, higher than edge cons.
                    if insertion_cost < best_insertion_cost:
                        best_insertion_cost = insertion_cost
                        best_customer = customer
                        best_position = i
                        best_route = temp_route
                        best_arrival_times = temp_arrival_times
            if best_customer is not None: #update the route if found feasible insertion
                enhanced_route = best_route
                customer_group.remove(best_customer)
                unvisited_customers.remove(best_customer)
                inserted_customers.append(best_customer)
                customers_inserted = True
                arrival_times = best_arrival_times
            else:
                if customer_group == unvisited_must_serve and customer_group:
                    extended_finish_time = driver_finish_time + 120
                    for customer in customer_group[:]: #try all positions to find best
                        best_position = None
                        best_insertion_cost = float('inf')
                        best_temp_route = None
                        best_temp_arrival_times = None
                        for i in range(len(enhanced_route) - 1):
                            temp_route = enhanced_route[:i + 1] + [customer] + enhanced_route[i + 1:]
                            temp_arrival_times = estimate_arrival_times(temp_route, driver_start_time)
                            if temp_arrival_times and len(temp_arrival_times) > i + 1: #insertion cost regardless of working hours
                                total_customer_sat = 0.0
                                for j in range(1, len(temp_route) - 1):
                                    cust = temp_route[j]
                                    arrival_time = temp_arrival_times[j]
                                    satisfaction = calculate_customer_satisfaction(
                                        arrival_time,
                                        cust.time_windows[day_index],
                                        buffer_minutes)
                                    total_customer_sat += satisfaction
                                temp_customer_satisfaction = total_customer_sat / (len(temp_route) - 2)
                                temp_driver_satisfaction, _, _ = calculate_driver_satisfaction( #insertion cost with effective alpha
                                    temp_route, day_index, previous_routes, previous_working_times)
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
                break

    if inserted_customers: print(f" Successfully inserted {len(inserted_customers)} additional customers:")
    final_arrival_times = estimate_arrival_times(enhanced_route, driver_start_time) #final metrics
    num_customers = len(enhanced_route) - 2  #final customer satisfaction
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
    driver_satisfaction, _, _ = calculate_driver_satisfaction( #final driver satisfaction
        enhanced_route, day_index, previous_routes, previous_working_times)
    total_cost = (1.0 - customer_satisfaction) * alpha + (1.0 - driver_satisfaction) * (1.0 - alpha) #final cost using original alpha
    print(f"Final enhanced route has {num_customers} customers with cost {total_cost:.4f}")
    print(f"Customer satisfaction: {customer_satisfaction:.4f}, Driver satisfaction: {driver_satisfaction:.4f}")
    return enhanced_route, total_cost, final_arrival_times, customer_satisfaction, driver_satisfaction