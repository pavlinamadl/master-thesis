# At the top of main.py, add:
import sys
import os
from typing import List, Tuple

# Current directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import from other modules
from customer_data import Customer, all_customers
from constants import (
    DRIVER_START_TIME, DRIVER_FINISH_TIME,
    ALPHA, EDGE_CONSISTENCY_BONUS
)
# Import optimization engine
from optimization_engine import get_optimization_results
from satisfaction_metrics import calculate_customer_satisfaction, calculate_working_time


def main():
    # Get optimization results from central engine
    results = get_optimization_results()

    # Extract data
    all_routes = results['routes']
    all_objective_values = results['objective_values']
    all_arrival_times = results['arrival_times']
    all_customer_satisfactions = results['customer_satisfactions']
    all_driver_satisfactions = results['driver_satisfactions']
    all_working_times = results['working_times']
    days = results['days']
    all_unvisited_customers = results['unvisited_customers']
    execution_times = results['execution_times']

    # Print overview
    print(f"Loaded {len(all_customers) - 1} customers and depot")
    print(f"Using alpha value: {ALPHA} (customer weight: {ALPHA}, driver weight: {1.0 - ALPHA})")

    # Count must-serve customers
    must_serve_count = sum(1 for c in all_customers if c.must_serve)
    print(f"Must-serve customers: {must_serve_count} (these customers must be included in each day's route)")

    # Print detailed results
    print("\n----- FINAL RESULTS -----")
    print(f"Alpha parameter used: {ALPHA} (customer weight: {ALPHA}, driver weight: {1.0 - ALPHA})")
    print(f"All must-serve customers ({must_serve_count}) were included in every day's route.")

    total_deviation = 0
    total_customers_served = 0

    for d in range(5):
        print(f"\n----- {days[d]} -----")
        print(f"{days[d]} route: {[c.id for c in all_routes[d]]}")

        working_time = all_working_times[d]
        print(f"{days[d]} working time: {working_time:.2f} minutes ({working_time / 60:.2f} hours)")
        print(f"{days[d]} objective value: {all_objective_values[d]:.2f}")
        print(f"{days[d]} customer satisfaction: {all_customer_satisfactions[d]:.2f}")
        print(f"{days[d]} driver satisfaction: {all_driver_satisfactions[d]:.2f}")
        print(f"{days[d]} execution time: {execution_times[d]:.2f} seconds")

        # Print route start and end times
        start_time = DRIVER_START_TIME
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

        # Count must-serve customers in route
        must_serve_in_route = sum(1 for c in all_routes[d] if c.must_serve)
        print(f"Must-serve customers in route: {must_serve_in_route} of {must_serve_count}")

        # Print unvisited customer count
        unvisited_count = len(all_unvisited_customers[d])
        unvisited_optional = sum(1 for c in all_unvisited_customers[d] if not c.must_serve)
        print(f"Unvisited customers: {unvisited_count} (all optional: {unvisited_optional == unvisited_count})")

        # Print customer service details
        print("\nCustomer Service Details:")
        total_satisfaction = 0.0

        for i in range(1, len(all_routes[d]) - 1):  # Skip depot
            customer = all_routes[d][i]
            arrival_time = all_arrival_times[d][i]
            time_window = customer.time_windows[d]
            satisfaction = calculate_customer_satisfaction(arrival_time, time_window)
            total_satisfaction += satisfaction

            # Format times for display
            arrival_hour = int(arrival_time // 60)
            arrival_minute = int(arrival_time % 60)
            window_start_hour = int(time_window.start // 60)
            window_start_minute = int(time_window.start % 60)
            window_end_hour = int(time_window.end // 60)
            window_end_minute = int(time_window.end % 60)

            # Calculate time deviation
            if arrival_time < time_window.start:
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

            # Format deviation for display
            deviation_hours = int(deviation // 60)
            deviation_minutes = int(deviation % 60)

            # Display service time for each customer
            service_time = customer.service_time

            print(f"  Customer {customer.id}:")
            print(f"    Must Serve: {'Yes' if customer.must_serve else 'No'}")
            print(
                f"    Time Window: {window_start_hour:02d}:{window_start_minute:02d}-{window_end_hour:02d}:{window_end_minute:02d}")
            print(f"    Arrival Time: {arrival_hour:02d}:{arrival_minute:02d}")
            print(f"    Service Time: {service_time:.2f} minutes")
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

    # Print overall statistics
    if total_customers_served > 0:
        avg_deviation = total_deviation / total_customers_served
        print(f"\nOverall Statistics:")
        print(f"  Total customers served across all days: {total_customers_served}")
        print(f"  Total time deviation: {total_deviation:.2f} minutes")
        print(f"  Average deviation per customer: {avg_deviation:.2f} minutes")

        # Calculate percentage of customers served from total available
        total_available = (len(all_customers) - 1) * 5  # -1 for depot, *5 for days
        percentage_served = (total_customers_served / total_available) * 100
        print(f"  Percentage of available customer-days served: {percentage_served:.2f}%")
        print(f"  Alpha value used: {ALPHA} (customer weight: {ALPHA}, driver weight: {1.0 - ALPHA})")

        # Count must-serve customers served across all days
        total_must_serve_available = must_serve_count * 5  # Must-serve customers * 5 days
        total_must_serve_served = sum(sum(1 for c in route if c.must_serve) for route in all_routes)
        must_serve_percentage = (total_must_serve_served / total_must_serve_available) * 100
        print(f"  Must-serve customers always included: {must_serve_percentage:.2f}% (should be 100%)")
        # total cost across all days
        total_cost = sum(all_objective_values)
        print(f"  Total cost of solution (sum of daily objectives): {total_cost:.2f}")


if __name__ == "__main__":
    main()