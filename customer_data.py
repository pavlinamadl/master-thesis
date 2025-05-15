import random
import numpy as np
from typing import NamedTuple, List
from collections import Counter
from constants import (
    AREA_SIZE, DEPOT_X, DEPOT_Y,
    TIME_WINDOW_START, TIME_WINDOW_END, TIME_WINDOW_INTERVAL,
    LUNCH_BREAK_START, LUNCH_BREAK_END
)


# Define the TimeWindow and Customer classes
class TimeWindow(NamedTuple):
    """Time window with start and end times in minutes from midnight"""
    start: float  # Start time of the window in minutes from midnight
    end: float  # End time of the window in minutes from midnight


class Customer(NamedTuple):
    id: int
    x: float
    y: float
    time_windows: List[TimeWindow]  # Time windows for each day
    service_time: float  # Service time in minutes for this customer
    must_serve: bool = True  # Flag indicating if customer must be served (default True)


def generate_time_windows(num_days: int = 5):
    """
    Generate all possible 30-minute time windows between 8 AM and 4:30 PM,
    excluding the lunch break period (12:00-12:30).
    """
    time_windows = []

    # Generate all possible time windows with specified interval
    for start_time in range(
            int(TIME_WINDOW_START),
            int(TIME_WINDOW_END),
            TIME_WINDOW_INTERVAL
    ):
        # Skip lunch break time slot
        if start_time == LUNCH_BREAK_START:
            continue

        end_time = start_time + TIME_WINDOW_INTERVAL
        time_windows.append(TimeWindow(start=start_time, end=end_time))

    return time_windows


# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Generate all possible time windows
possible_time_windows = generate_time_windows(5)

# Create depot with dummy time windows (not used)
dummy_tw = [TimeWindow(start=0, end=0) for _ in range(5)]
depot = Customer(0, DEPOT_X, DEPOT_Y, dummy_tw, 0.0, must_serve=False)  # Depot has zero service time and doesn't need to be "served"

# Generate 75 random customers with time window constraints
customers_list = [depot]

# Track time window choices to ensure at most 10 customers per time window per day
time_window_counts = [Counter() for _ in range(5)]  # One counter for each day

for i in range(1, 31):  # 30 customers
    x = random.uniform(0, AREA_SIZE)
    y = random.uniform(0, AREA_SIZE)

    # Generate a normally distributed service time between 1.5-7.5 minutes
    # Using mean of 4.5 and std dev of 1.0 to mostly stay within 1.5-7.5 range
    service_time = np.random.normal(4.5, 1.0)
    # Clamp to ensure it's within 1.5-7.5 range
    service_time = max(1.5, min(7.5, service_time))

    # Randomly select a time window for each day
    customer_time_windows = []
    for day in range(5):
        # Filter available time windows (those with fewer than 10 customers)
        available_windows = [tw for tw in possible_time_windows
                             if time_window_counts[day][tw] < 10]

        # If no available windows, try to find the least crowded one
        if not available_windows:
            selected_window = min(possible_time_windows,
                                  key=lambda tw: time_window_counts[day][tw])
        else:
            selected_window = random.choice(available_windows)

        # Increment the count for the selected window
        time_window_counts[day][selected_window] += 1
        customer_time_windows.append(selected_window)

    c = Customer(i, x, y, customer_time_windows, service_time, must_serve=True)  # All customers must be served
    customers_list.append(c)

# This variable will be imported by main.py
all_customers = customers_list

# Print summary information when the file is run directly
if __name__ == "__main__":
    print(f"Generated {len(customers_list) - 1} customers plus depot")
    print(f"Depot location: ({DEPOT_X}, {DEPOT_Y})")
    print(f"All customers have must_serve=True (must be included in each day's route)")

    # Print time window ranges
    print(f"\nTime windows: {len(possible_time_windows)} slots of {TIME_WINDOW_INTERVAL} minutes")
    print(
        f"From {int(TIME_WINDOW_START // 60):02d}:{int(TIME_WINDOW_START % 60):02d} to {int(TIME_WINDOW_END // 60):02d}:{int(TIME_WINDOW_END % 60):02d}")
    print(
        f"Excluding lunch break: {int(LUNCH_BREAK_START // 60):02d}:{int(LUNCH_BREAK_START % 60):02d}-{int(LUNCH_BREAK_END // 60):02d}:{int(LUNCH_BREAK_END % 60):02d}")

    # Print first 5 customers as a sample
    print("\nSample of customer data:")
    for customer in customers_list[:6]:
        if customer.id == 0:
            print(f"Customer {customer.id} (Depot): ({customer.x}, {customer.y})")
        else:
            print(f"Customer {customer.id}: ({customer.x:.2f}, {customer.y:.2f})")
            print(f"  Service time: {customer.service_time:.2f} minutes")
            print(f"  Time windows for Monday: {customer.time_windows[0].start}-{customer.time_windows[0].end} min")
            print(f"  Must be served: {customer.must_serve}")

    print("...")

    # Print time window distribution statistics
    print("\nTime Window Distribution:")
    for day in range(5):
        print(f"\nDay {day + 1}:")
        tw_distribution = time_window_counts[day]
        for tw, count in sorted(tw_distribution.items(), key=lambda x: x[0].start):
            tw_start_hour = int(tw.start // 60)
            tw_start_minute = int(tw.start % 60)
            tw_end_hour = int(tw.end // 60)
            tw_end_minute = int(tw.end % 60)
            print(
                f"  {tw_start_hour:02d}:{tw_start_minute:02d}-{tw_end_hour:02d}:{tw_end_minute:02d}: {count} customers")