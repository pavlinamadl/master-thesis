import random
import numpy as np
from typing import List
from collections import Counter
from customer_data import TimeWindow, Customer, generate_time_windows
from constants import (AREA_SIZE)
def customer_wants_service_on_day(customer: Customer, day_index: int) -> bool:
    if customer.id == 0:#depot always available
        return True
    time_window = customer.time_windows[day_index] #TW>= 2000; impossible: no service wanted
    return time_window.start < 2000  #reasonable time window means they want service

def count_daily_service_requests(customers: list) -> dict: #count daily requests
    days = ["Mon", "Tue", "Wed", "Thur", "Fri"]
    daily_counts = {}
    for day_idx, day_name in enumerate(days):
        count = sum(1 for c in customers 
                   if c.id != 0 and customer_wants_service_on_day(c, day_idx))
        daily_counts[day_name] = count
    return daily_counts

def is_must_serve_for_day(customer: Customer, day_index: int) -> bool:
    if customer.id == 0: return False #depot is never must-serve
    return customer_wants_service_on_day(customer, day_index) #must-serve when they want service

def print_service_summary(customers: list):
    non_depot_customers = [c for c in customers if c.id != 0]
    daily_counts = count_daily_service_requests(customers)
    
    print(f"Service Request Summary ({len(non_depot_customers)} total customers):")
    for day, count in daily_counts.items():
        percentage = (count / len(non_depot_customers)) * 100 if non_depot_customers else 0
        print(f"  {day}: {count} customers want service ({percentage:.1f}%)")
    
    total_requests = sum(daily_counts.values())
    max_possible = len(non_depot_customers) * 5
    overall_percentage = (total_requests / max_possible) * 100 if max_possible > 0 else 0
    
    print(f"  Total: {total_requests}/{max_possible} customer-days ({overall_percentage:.1f}%)")

def generate_additional_customers(num_customers: int = 45, start_id: int = 1) -> List[Customer]:
    random.seed(42)
    np.random.seed(42)
    possible_time_windows = generate_time_windows(5)
    no_service_window = TimeWindow(start=2000, end=2030)  # Impossible time
    additional_customers = []
    time_window_counts = [Counter() for _ in range(5)]

    for i in range(num_customers):
        customer_id = start_id + i
        x = random.uniform(0, AREA_SIZE)
        y = random.uniform(0, AREA_SIZE)
        service_time = max(1.5, min(7.5, np.random.normal(4.5, 1.0))) #same

        customer_time_windows = [] #time windows for each day
        wants_service_any_day = False
        for day in range(5):
            wants_service = random.random() < 0.5
            if wants_service:
                wants_service_any_day = True
                available_windows = [tw for tw in possible_time_windows #customers per time window per day (max 3)
                                     if time_window_counts[day][tw] < 3]
                if available_windows:
                    selected_window = random.choice(available_windows)
                else: #pick the least crowded one if full
                    selected_window = min(possible_time_windows, key=lambda tw: time_window_counts[day][tw])
                time_window_counts[day][selected_window] += 1
                customer_time_windows.append(selected_window)
            else: customer_time_windows.append(no_service_window)
        if wants_service_any_day: #create customer if they want service at least one day
            customer = Customer(
                id=customer_id,
                x=x,
                y=y,
                time_windows=customer_time_windows,
                service_time=service_time,
                must_serve=False)
            additional_customers.append(customer)
    return additional_customers

def preprocess_customers_for_optimization(customers: List[Customer]) -> List[Customer]: #preprocessing
    processed = []
    for customer in customers:
        if customer.id == 0:  #always include depot
            processed.append(customer)
        else:
            wants_service_sometime = any(
                customer_wants_service_on_day(customer, day) 
                for day in range(5) #check if customer wants service on at least one day
            )
            
            if wants_service_sometime: #modify customers tw in case they want service at least once
                new_time_windows = []
                for day in range(5):
                    if customer_wants_service_on_day(customer, day):
                        new_time_windows.append(customer.time_windows[day])
                    else: #replace (0,0) with a very late time window that will be skipped
                        new_time_windows.append(TimeWindow(start=2000, end=2030))  #impossible time

                processed_customer = Customer(
                    id=customer.id,
                    x=customer.x,
                    y=customer.y,
                    time_windows=new_time_windows,
                    service_time=customer.service_time,
                    must_serve=False  #make them optional so they get skipped when time window is impossible
                )
                processed.append(processed_customer)
    return processed

from customer_data import depot #import depot
additional_customers_list = generate_additional_customers() #generate the additional customers
all_customers = preprocess_customers_for_optimization([depot] + additional_customers_list) #new customer list with depot

if __name__ == "__main__":
    print(f"Generated {len(additional_customers_list)} additional customers")
    print_service_summary(all_customers) #print service summary using utility function

    days = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    print("\nSample of additional customer data:")
    for i, customer in enumerate(additional_customers_list[:3]):
        print(f"Customer {customer.id}: ({customer.x:.2f}, {customer.y:.2f})")
        service_days = []
        must_serve_days = []
        for day in range(5):
            tw = customer.time_windows[day]
            if customer_wants_service_on_day(customer, day):
                service_days.append(f"{days[day]} ({tw.start}-{tw.end})")
            if is_must_serve_for_day(customer, day):
                must_serve_days.append(days[day])
        print(f"  Wants service on: {', '.join(service_days)}")

