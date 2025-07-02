import math
import random
from typing import List, Tuple, Set
from additional_customers import Customer
from constants import (
    DRIVER_START_TIME, DRIVER_FINISH_TIME, BUFFER_MINUTES,
    ALPHA, EDGE_CONSISTENCY_BONUS,
    MAX_TABU_ITERATIONS, TABU_DIVERSIFICATION_THRESHOLD,
    TABU_ASPIRATION_COEF, TABU_INITIAL_TENURE_FACTOR)
from time_utils import estimate_arrival_times, calculate_customer_satisfaction
from satisfaction_metrics import calculate_driver_satisfaction


def diversify_route(route: List[Customer]) -> List[Customer]: #series of random 2-opt moves
    diversified_route = route.copy()
    num_moves = min(5, len(route) // 4) #num. of random 2-opt moves
    for _ in range(num_moves):
        i = random.randint(1, len(diversified_route) - 3)
        j = random.randint(i + 1, len(diversified_route) - 2)
        diversified_route[i:j + 1] = reversed(diversified_route[i:j + 1]) #2-opt swap; reverse segment between i and j
    return diversified_route

def tabu_enhanced_two_opt( #2-opt local search algorithm enhanced with tabu search
        route: List[Customer],
        day_index: int,
        previous_routes: List[List[Customer]],
        previous_working_times: List[float],
        alpha: float = ALPHA,  # Changed parameter name from w_cust
        edge_consistency_bonus: float = EDGE_CONSISTENCY_BONUS,
        driver_start_time: float = DRIVER_START_TIME,
        buffer_minutes: float = BUFFER_MINUTES,
        max_iterations: int = MAX_TABU_ITERATIONS,
        diversification_threshold: int = TABU_DIVERSIFICATION_THRESHOLD
) -> Tuple[List[Customer], float, List[float], float, float]:
    effective_alpha = alpha
    if alpha == 0.0 and day_index == 0: #case for alpha=0.0 on Monday - use alpha=0.5 temporarily
        effective_alpha = 0.5
    best_route = route.copy() #initial arrival times and costs
    current_route = route.copy()
    best_arrival_times = estimate_arrival_times(best_route, driver_start_time) #arrival times for the initial route
    num_customers = len(best_route) - 2 #number of customers (excluding depot)
    #dynamic tabu tenure based on the square root of number of customers
    tabu_tenure = int(math.sqrt(num_customers) * TABU_INITIAL_TENURE_FACTOR) + 1
    print(f"Tabu tenure: {tabu_tenure} ({num_customers} customers)")
    best_customer_satisfaction = 0.0
    if num_customers > 0:
        total_customer_sat = 0.0
        for i in range(1, len(best_route) - 1):  #skip depot
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
    best_driver_satisfaction, _, _ = calculate_driver_satisfaction( #initial driver satisfaction
        best_route, day_index, previous_routes, previous_working_times
    )
    if day_index == 0: #initial best cost using effective_alpha
        best_cost = (1.0 - best_customer_satisfaction) * effective_alpha + (1.0 - best_driver_satisfaction) * (
                    1.0 - effective_alpha)
    else: best_cost = (1.0 - best_customer_satisfaction) * alpha + (1.0 - best_driver_satisfaction) * (1.0 - alpha)
    current_cost = best_cost

    tabu_list = {} #tabu list and counters
    iteration = 0
    iterations_without_improvement = 0
    must_serve_ids = {c.id for c in route if c.must_serve}
    while iteration < max_iterations:
        iteration += 1
        best_neighbor_cost = float('inf')
        best_move = None
        best_neighbor_route = None
        best_neighbor_arrival_times = None
        best_neighbor_customer_satisfaction = 0.0
        best_neighbor_driver_satisfaction = 0.0
        if iterations_without_improvement >= diversification_threshold: #check if to apply diversification
            diversified_route = diversify_route(current_route)
            diversified_arrival_times = estimate_arrival_times(diversified_route, driver_start_time) #check if feasible
            current_route = diversified_route #accept diversified route even if exceeds working hours
            iterations_without_improvement = 0
            new_num_customers = len(current_route) - 2 #if the route structure changes, recalculate tenure
            if new_num_customers != num_customers:
                num_customers = new_num_customers
                tabu_tenure = int(math.sqrt(num_customers)) + 1

        for i in range(1, len(current_route) - 2): #explore all 2-opt swaps
            for j in range(i + 1, len(current_route) - 1):
                if j - i == 1:
                    continue  #skip adjacent edges
                neighbor_route = current_route.copy() #new route with 2-opt swap
                neighbor_route[i:j + 1] = reversed(current_route[i:j + 1]) #reverse between i and j
                neighbor_arrival_times = estimate_arrival_times(neighbor_route, driver_start_time) #calculate new arrival times

                contain_must_serve = any(c.must_serve for c in neighbor_route[i:j + 1]) #skip if finishes after the end of day, unless includes must serve
                if neighbor_arrival_times[-1] > DRIVER_FINISH_TIME and not contain_must_serve:
                    continue
                neighbor_total_customer_sat = 0.0 #new customer satisfaction
                for k in range(1, len(neighbor_route) - 1):  #skip depot
                    customer = neighbor_route[k]
                    arrival_time = neighbor_arrival_times[k]
                    satisfaction = calculate_customer_satisfaction(
                        arrival_time,
                        customer.time_windows[day_index],
                        buffer_minutes
                    )
                    neighbor_total_customer_sat += satisfaction
                neighbor_customer_satisfaction = neighbor_total_customer_sat / num_customers if num_customers > 0 else 1.0
                neighbor_driver_satisfaction, _, _ = calculate_driver_satisfaction( #new driver satisfaction
                    neighbor_route, day_index, previous_routes, previous_working_times
                )
                if day_index > 0: #new total cost with edge consistency
                    previous_edges = set()
                    if previous_routes:
                        prev_route = previous_routes[-1]  #most recent route
                        for idx in range(len(prev_route) - 1):
                            previous_edges.add((prev_route[idx].id, prev_route[idx + 1].id))
                    consistent_edges = 0 #how many edges are consistent with previous day
                    for idx in range(len(neighbor_route) - 1):
                        if (neighbor_route[idx].id, neighbor_route[idx + 1].id) in previous_edges:
                            consistent_edges += 1
                    edge_consistency_bonus_weighted = consistent_edges * edge_consistency_bonus * (1.0 - alpha)
                    neighbor_cost = (1.0 - neighbor_customer_satisfaction) * alpha + \
                                    (1.0 - neighbor_driver_satisfaction) * (1.0 - alpha) - \
                                    edge_consistency_bonus_weighted
                else: #monday - use effective alpha
                    neighbor_cost = (1.0 - neighbor_customer_satisfaction) * effective_alpha + \
                                    (1.0 - neighbor_driver_satisfaction) * (1.0 - effective_alpha)
                move = (i, j)

                is_tabu = move in tabu_list #tabu condition with aspiration criterion
                aspiration_threshold = best_cost * TABU_ASPIRATION_COEF
                is_aspiration = neighbor_cost < aspiration_threshold
                if (not is_tabu or is_aspiration) and neighbor_cost < best_neighbor_cost: #accept if not tabu or meets aspiration criteria
                    best_neighbor_cost = neighbor_cost
                    best_move = move
                    best_neighbor_route = neighbor_route.copy()
                    best_neighbor_arrival_times = neighbor_arrival_times.copy()
                    best_neighbor_customer_satisfaction = neighbor_customer_satisfaction
                    best_neighbor_driver_satisfaction = neighbor_driver_satisfaction

        if best_neighbor_route is None: #no improvement found or no feasible move
            break

        current_route = best_neighbor_route #update
        current_cost = best_neighbor_cost #update
        if best_neighbor_cost < best_cost:
            best_route = best_neighbor_route.copy()
            best_cost = best_neighbor_cost
            best_arrival_times = best_neighbor_arrival_times.copy()
            best_customer_satisfaction = best_neighbor_customer_satisfaction
            best_driver_satisfaction = best_neighbor_driver_satisfaction
            iterations_without_improvement = 0
        else:
            iterations_without_improvement += 1

        moves_to_remove = [] #reduce tabu tenure, update tabu list
        for move in tabu_list:
            tabu_list[move] -= 1
            if tabu_list[move] <= 0:
                moves_to_remove.append(move)
        for move in moves_to_remove: #remove expired tabu moves
            del tabu_list[move]

        tabu_list[best_move] = tabu_tenure
        inverse_move = (best_move[1], best_move[0])
        tabu_list[inverse_move] = tabu_tenure

    # Calculate the actual cost with the original alpha for return value consistency
    actual_cost = (1.0 - best_customer_satisfaction) * alpha + (1.0 - best_driver_satisfaction) * (1.0 - alpha)
    return best_route, actual_cost, best_arrival_times, best_customer_satisfaction, best_driver_satisfaction