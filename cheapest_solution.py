#Cheapest Distance Solution Calculator
import sys
import os
from typing import List

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from customer_data import Customer, all_customers, depot
from distance_utils import distance
from time_utils import estimate_arrival_times
from constants import DRIVER_START_TIME

def nearest_neighbor_route(customers: List[Customer], depot: Customer) -> List[Customer]: #route using nearest neighbor heuristic starting and ending at depot
    if not customers:
        return [depot, depot]

    route = [depot] #start at depot
    unvisited = customers.copy()
    current_customer = depot

    while unvisited:  #visit all customers using nearest neighbor
        nearest_customer = min(unvisited, key=lambda c: distance(current_customer, c))
        route.append(nearest_customer) #add to route, mark as visited
        unvisited.remove(nearest_customer)
        current_customer = nearest_customer

    route.append(depot) #return to depot
    return route

def calculate_route_distance(route: List[Customer]) -> float: #calculate manhattan distance for the route
    total_distance = 0.0
    for i in range(1, len(route)):
        total_distance += distance(route[i-1], route[i])
    return total_distance

def calculate_working_time_simple(route: List[Customer]) -> tuple: #calculate working time
    arrival_times = estimate_arrival_times(route, DRIVER_START_TIME)
    start_time = arrival_times[0]  #start at depot
    end_time = arrival_times[-1]   #end at depot
    total_working_time = end_time - start_time
    return start_time, end_time, total_working_time

def main():
    customers = all_customers
    route = nearest_neighbor_route(customers, depot)
    distance_m = calculate_route_distance(route)
    distance_km = distance_m / 1000
    start_time, end_time, working_time = calculate_working_time_simple(route)

    print(f"\nRoute: {[c.id for c in route]}")
    print(f"Customers served: {len(route)-2}")
    print(f"Total distance: {distance_m:.2f} meters ({distance_km:.2f} km)")
    print(f"Working time: {working_time:.2f} minutes ({working_time/60:.2f} hours)")

    start_hour = int(start_time // 60)     #format start and end times
    start_minute = int(start_time % 60)
    end_hour = int(end_time // 60)
    end_minute = int(end_time % 60)

    print(f"Start time: {start_hour:02d}:{start_minute:02d}")
    print(f"End time: {end_hour:02d}:{end_minute:02d}")

if __name__ == "__main__":
    main()