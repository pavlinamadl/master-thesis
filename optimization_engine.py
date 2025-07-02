import time
from additional_customers import Customer, all_customers, customer_wants_service_on_day
from constants import (DRIVER_START_TIME, DRIVER_FINISH_TIME, BUFFER_MINUTES, ALPHA, EDGE_CONSISTENCY_BONUS)
from route_construction import insertion_heuristic
from route_optimization import tabu_enhanced_two_opt
from route_enhancement import attempt_additional_insertions
from satisfaction_metrics import calculate_customer_satisfaction, calculate_working_time
_cached_results = None

def get_optimization_results(force_recompute=False):
    global _cached_results
    if _cached_results is not None and not force_recompute: #return cached results if available
        return _cached_results
    print(f"Using alpha = {ALPHA} (customer weight: {ALPHA}, driver weight: {1.0 - ALPHA})")
    all_routes = []
    all_arrival_times = []
    all_objective_values = []
    all_customer_satisfactions = []
    all_driver_satisfactions = []
    all_working_times = []
    days = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    day_indices = list(range(5))
    all_unvisited_customers = []
    all_execution_times = []
    all_construction_times = []
    all_optimization_times = []
    all_enhancement_times = []

    for day_idx in range(5):
        if ALPHA == 0.0 and day_idx > 0: #copy Monday’s route for perfect consistency
            all_routes.append(all_routes[0])
            all_arrival_times.append(all_arrival_times[0])
            all_objective_values.append(all_objective_values[0])
            all_customer_satisfactions.append(all_customer_satisfactions[0])
            all_driver_satisfactions.append(all_driver_satisfactions[0])
            all_working_times.append(all_working_times[0])
            all_unvisited_customers.append(all_unvisited_customers[0])
            all_execution_times.append(0.0)
            all_construction_times.append(0.0)
            all_optimization_times.append(0.0)
            all_enhancement_times.append(0.0)
            continue

        start_time_total = time.time() #track execution time
        previous_routes = all_routes.copy() #previous route
        previous_working_times = all_working_times.copy() #previous working time

        day_customers = [c for c in all_customers #customers wanting service on this day
                         if c.id == 0 or customer_wants_service_on_day(c, day_idx)]

        print("Initial route using insertion heuristic is generating.") #creating initial route using insertion heuristic
        start_time_construction = time.time()
        initial_route, initial_cost, initial_arrival_times, initial_cust_sat, initial_driver_sat = insertion_heuristic(
            customers=day_customers,
            day_index=day_idx,
            previous_routes=previous_routes,
            previous_working_times=previous_working_times,
            alpha=ALPHA,
            edge_consistency_bonus=EDGE_CONSISTENCY_BONUS)
        construction_time = time.time() - start_time_construction

        visited_customer_ids = {customer.id for customer in initial_route} #identify unvisited customers
        unvisited_customers = [customer for customer in day_customers if
                               customer.id not in visited_customer_ids and customer.id != 0]
        all_unvisited_customers.append(unvisited_customers)

        print("Improving route using tabu-enhanced 2-opt local search.") #improve route using tabu-enhanced 2-opt
        start_time_optimization = time.time()
        improved_route, improved_cost, improved_arrival_times, improved_cust_sat, improved_driver_sat = tabu_enhanced_two_opt(
            route=initial_route,
            day_index=day_idx,
            previous_routes=previous_routes,
            previous_working_times=previous_working_times,
            alpha=ALPHA,
            edge_consistency_bonus=EDGE_CONSISTENCY_BONUS)
        optimization_time = time.time() - start_time_optimization

        print("Attempting to insert additional customers post-optimization.") #attempting to insert additional unvisited customers
        start_time_enhancement = time.time()
        final_route, final_cost, arrival_times, customer_sat, driver_sat = attempt_additional_insertions(
            route=improved_route,
            unvisited_customers=unvisited_customers,
            day_index=day_idx,
            previous_routes=previous_routes,
            previous_working_times=previous_working_times,
            alpha=ALPHA,
            edge_consistency_bonus=EDGE_CONSISTENCY_BONUS)
        enhancement_time = time.time() - start_time_enhancement
        working_time = calculate_working_time(final_route) #total working time
        execution_time = time.time() - start_time_total #total execution time

        all_routes.append(final_route) #store results
        all_objective_values.append(final_cost)
        all_arrival_times.append(arrival_times)
        all_customer_satisfactions.append(customer_sat)
        all_driver_satisfactions.append(driver_sat)
        all_working_times.append(working_time)
        all_execution_times.append(execution_time)
        all_construction_times.append(construction_time)
        all_optimization_times.append(optimization_time)
        all_enhancement_times.append(enhancement_time)

    _cached_results = { #store results
        'routes': all_routes,
        'arrival_times': all_arrival_times,
        'objective_values': all_objective_values,
        'customer_satisfactions': all_customer_satisfactions,
        'driver_satisfactions': all_driver_satisfactions,
        'working_times': all_working_times,
        'days': days,
        'day_indices': day_indices,
        'unvisited_customers': all_unvisited_customers,
        'execution_times': all_execution_times,
        'construction_times': all_construction_times,
        'optimization_times': all_optimization_times,
        'enhancement_times': all_enhancement_times
    }
    return _cached_results

def clear_optimization_cache(): #clear cache if needed
    global _cached_results
    _cached_results = None