import random
import numpy as np
from typing import NamedTuple, List
from collections import Counter
from constants import (
    AREA_SIZE, DEPOT_X, DEPOT_Y,
    TIME_WINDOW_START, TIME_WINDOW_END, TIME_WINDOW_INTERVAL,
    LUNCH_BREAK_START, LUNCH_BREAK_END
)
#define TimeWindow and Customer classes
class TimeWindow(NamedTuple):
    start: float
    end: float

class Customer(NamedTuple):
    id: int
    x: float
    y: float
    time_windows: List[TimeWindow]
    service_time: float
    must_serve: bool = True

def generate_time_windows(num_days: int = 5): #generate all possible time windows with specified interval
    time_windows = []
    for start_time in range(
            int(TIME_WINDOW_START),
            int(TIME_WINDOW_END),
            TIME_WINDOW_INTERVAL
    ):
        if start_time == LUNCH_BREAK_START: #skip lunch break
            continue

        end_time = start_time + TIME_WINDOW_INTERVAL
        time_windows.append(TimeWindow(start=start_time, end=end_time))
    return time_windows

random.seed(42) #random seed
np.random.seed(42)

possible_time_windows = generate_time_windows(5) #generate all possible time windows

dummy_tw = [TimeWindow(start=0, end=0) for _ in range(5)] #depot with dummy time windows
depot = Customer(0, DEPOT_X, DEPOT_Y, dummy_tw, 0.0, must_serve=False)  #zero service time and doesn't need to be served

customers_list = [depot]

time_window_counts = [Counter() for _ in range(5)]  #track time window choices

for i in range(1, 31):  # 30 customers
    x = random.uniform(0, AREA_SIZE)
    y = random.uniform(0, AREA_SIZE)

    #normally distributed service time between 1.5-7.5 minutes, mean of 4.5 and std dev of 1.0
    service_time = np.random.normal(4.5, 1.0)
    service_time = max(1.5, min(7.5, service_time))

    customer_time_windows = [] #randomly select a time window for each day
    for day in range(5):
        available_windows = [tw for tw in possible_time_windows  # Filter available time windows
                             if time_window_counts[day][tw] < 10]

        if not available_windows: #if no available windows, find the least crowded one
            selected_window = min(possible_time_windows,
                                  key=lambda tw: time_window_counts[day][tw])
        else:
            selected_window = random.choice(available_windows)

        time_window_counts[day][selected_window] += 1  #increment the count
        customer_time_windows.append(selected_window)

    c = Customer(i, x, y, customer_time_windows, service_time, must_serve=True)  #all must-serve
    customers_list.append(c)

all_customers = customers_list

# Print summary
if __name__ == "__main__":
    print(f"Generated {len(customers_list) - 1} customers plus depot")
    print(f"\nTime windows: {len(possible_time_windows)} slots of {TIME_WINDOW_INTERVAL} minutes")
    print(
        f"From {int(TIME_WINDOW_START // 60):02d}:{int(TIME_WINDOW_START % 60):02d} to {int(TIME_WINDOW_END // 60):02d}:{int(TIME_WINDOW_END % 60):02d}")
    print(
        f"Excluding lunch break: {int(LUNCH_BREAK_START // 60):02d}:{int(LUNCH_BREAK_START % 60):02d}-{int(LUNCH_BREAK_END // 60):02d}:{int(LUNCH_BREAK_END % 60):02d}")

    print("\nSample of customer data:") # Print first 2 customers as a sample
    for customer in customers_list[1:3]:
            print(f"Customer {customer.id}: ({customer.x:.2f}, {customer.y:.2f})")
            print(f"  Service time: {customer.service_time:.2f} minutes")
            print(f"  Time windows for Monday: {customer.time_windows[0].start}-{customer.time_windows[0].end} min")
            print(f"  Must be served: {customer.must_serve}")