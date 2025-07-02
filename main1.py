import sys
import os
from typing import List, Tuple
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
from additional_customers import Customer, all_customers
from constants import (DRIVER_START_TIME, DRIVER_FINISH_TIME, ALPHA, EDGE_CONSISTENCY_BONUS)
from optimization_engine import get_optimization_results
from satisfaction_metrics import calculate_customer_satisfaction, calculate_working_time
from distance_utils import distance


def calculate_route_distance(route: List[Customer]) -> float: #route distance calculation
    total_distance = 0.0
    for i in range(1, len(route)):
        total_distance += distance(route[i - 1], route[i])
    return total_distance

def get_route_edges(route: List[Customer]) -> set: #extract edges
    edges = set()
    for i in range(len(route) - 1):
        edges.add((route[i].id, route[i + 1].id))
    return edges

def calculate_consecutive_day_edge_comparison(routes: List[List[Customer]]) -> Tuple[List[int], float, float]:
    #how many edges each day shares with the previous day, compare each day with previous day
    same_edge_counts = []
    total_edge_bonus = 0.0
    for day in range(1, 5):
        prev_day_edges = get_route_edges(routes[day - 1])
        current_day_edges = get_route_edges(routes[day])
        same_edges = len(prev_day_edges.intersection(current_day_edges))
        same_edge_counts.append(same_edges)
        total_edge_bonus += same_edges * EDGE_CONSISTENCY_BONUS #calculate edge bonus for this day
    if len(same_edge_counts) == 0:
        average_same_edges = 0.0
    else: average_same_edges = sum(same_edge_counts) / len(same_edge_counts)
    return same_edge_counts, average_same_edges, total_edge_bonus

def calculate_single_objective_value(routes: List[List[Customer]], customer_satisfactions: List[float],
                                     driver_satisfactions: List[float]) -> float: #average daily objective value
    avg_driver_satisfaction = sum(driver_satisfactions) / len(driver_satisfactions)
    avg_customer_satisfaction = sum(customer_satisfactions) / len(customer_satisfactions)
    total_reused_edges = 0 #total reused edges across all consecutive day
    for day in range(1, 5):  # Tue-Fri
        prev_day_edges = get_route_edges(routes[day - 1])
        current_day_edges = get_route_edges(routes[day])
        same_edges = len(prev_day_edges.intersection(current_day_edges))
        total_reused_edges += same_edges
    avg_reused_edges_per_day = total_reused_edges / 5.0 #average reused edges per day (including Monday with 0)
    objective = ((1.0 - ALPHA) * avg_driver_satisfaction +
                 ALPHA * avg_customer_satisfaction +
                 (1.0 - ALPHA) * avg_reused_edges_per_day * EDGE_CONSISTENCY_BONUS) #final objective
    return objective

def main():
    results = get_optimization_results()
    all_routes = results['routes']
    all_objective_values = results['objective_values']
    all_arrival_times = results['arrival_times']
    all_customer_satisfactions = results['customer_satisfactions']
    all_driver_satisfactions = results['driver_satisfactions']
    all_working_times = results['working_times']
    days = results['days']
    all_unvisited_customers = results['unvisited_customers']
    execution_times = results['execution_times']
    must_serve_count = sum(1 for c in all_customers if c.must_serve)
    print(f"Must-serve customers: {must_serve_count} (must be served everyday.)")
    daily_distances = [] #daily distances
    for d in range(5):
        daily_distance = calculate_route_distance(all_routes[d])
        daily_distances.append(daily_distance)
    total_deviation = 0
    total_customers_served = 0
    for d in range(5):
        print(f"\n----- {days[d]} -----")
        print(f"{days[d]} route: {[c.id for c in all_routes[d]]}")

        working_time = all_working_times[d]
        route_distance = daily_distances[d]
        print(f"{days[d]} working time: {working_time:.2f} minutes ({working_time / 60:.2f} hours)")
        print(f"{days[d]} total distance: {route_distance:.2f} meters ({route_distance / 1000:.2f} km)")
        print(f"{days[d]} customer satisfaction: {all_customer_satisfactions[d]:.2f}")
        print(f"{days[d]} driver satisfaction: {all_driver_satisfactions[d]:.2f}")
        print(f"{days[d]} execution time: {execution_times[d]:.2f} seconds")
        start_time = DRIVER_START_TIME # Print route start and end times
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

        print("\nCustomer Service Details:") #customer service details
        total_satisfaction = 0.0

        for i in range(1, len(all_routes[d]) - 1):  # Skip depot
            customer = all_routes[d][i]
            arrival_time = all_arrival_times[d][i]
            time_window = customer.time_windows[d]
            satisfaction = calculate_customer_satisfaction(arrival_time, time_window)
            total_satisfaction += satisfaction

            arrival_hour = int(arrival_time // 60) #format times
            arrival_minute = int(arrival_time % 60)
            window_start_hour = int(time_window.start // 60)
            window_start_minute = int(time_window.start % 60)
            window_end_hour = int(time_window.end // 60)
            window_end_minute = int(time_window.end % 60)

            if arrival_time < time_window.start: #time deviation
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

            deviation_hours = int(deviation // 60) #format deviation
            deviation_minutes = int(deviation % 60)
            service_time = customer.service_time #service time for each customer

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

    total_distance_meters = sum(daily_distances) #total distance
    total_distance_km = total_distance_meters / 1000
    if total_customers_served > 0: #print results
        print(f"\n  === FINAL METRICS ===")

        avg_deviation = total_deviation / total_customers_served
        print(f"  Total customers served across all days: {total_customers_served}")
        print(f"  Total time deviation: {total_deviation:.2f} minutes")
        print(f"  Average deviation per customer: {avg_deviation:.2f} minutes")
        print(f"  Total distance traveled (all days): {total_distance_meters:.2f} meters ({total_distance_km:.2f} km)")
        print(f"  Average daily distance: {total_distance_meters / 5:.2f} meters ({total_distance_km / 5:.2f} km)")
        print(f"  Daily distances breakdown:")
        for d in range(5):
            daily_km = daily_distances[d] / 1000
            print(f"    {days[d]}: {daily_distances[d]:.2f} meters ({daily_km:.2f} km)")
        avg_driver_satisfaction = sum(all_driver_satisfactions) / len(all_driver_satisfactions)
        print(f"  Average Driver Satisfaction: {avg_driver_satisfaction:.4f}") #avg driver sat
        avg_customer_satisfaction = sum(all_customer_satisfactions) / len(all_customer_satisfactions)
        print(f"  Average Customer Satisfaction: {avg_customer_satisfaction:.4f}") #avg customer sat
        same_edge_counts, avg_same_edges, total_edge_bonus = calculate_consecutive_day_edge_comparison(all_routes)
        print(f"  Average Number of Same Edges/Day (consecutive comparisons): {avg_same_edges:.2f}")
        print(f"    Edge counts: Tue vs Mon={same_edge_counts[0]}, Wed vs Tue={same_edge_counts[1]}, Thu vs Wed={same_edge_counts[2]}, Fri vs Thu={same_edge_counts[3]}")
        print(f"  Edge Bonus (Total): {total_edge_bonus:.4f}")
        objective_value = calculate_single_objective_value(all_routes, all_customer_satisfactions,
                                                           all_driver_satisfactions)
        print(f"  Average Objective Value/Day: {objective_value:.4f}")

if __name__ == "__main__":
    main()