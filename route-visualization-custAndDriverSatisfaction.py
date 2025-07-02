import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from typing import List
import traceback
from additional_customers import Customer
from optimization_engine import get_optimization_results
from constants import (BUFFER_MINUTES, DRIVER_START_TIME, DRIVER_FINISH_TIME,
                      LUNCH_BREAK_START, LUNCH_BREAK_END, ALPHA, VISUALIZATION_DPI)
from time_utils import calculate_customer_satisfaction
VISUALIZATION_SIZE = (16, 12)
MAP_SIZE = (16, 12)
TIMELINE_SIZE = (16, 12)


def get_satisfaction_color(satisfaction): #gradient color for satisfaction score
    if satisfaction >= 0.9: return 'darkblue'
    elif satisfaction >= 0.7: return 'blue'
    elif satisfaction >= 0.5: return 'lightblue'
    elif satisfaction >= 0.3: return 'orange'
    else: return 'red'

def get_time_window_color(arrival_time, time_window): #color based on arrival time vs time window
    if time_window.start <= arrival_time <= time_window.end:
        return 'blue'
    elif arrival_time < time_window.start:
        time_deviation = time_window.start - arrival_time
        return 'green' if time_deviation <= BUFFER_MINUTES else 'darkgreen'
    else:
        time_deviation = arrival_time - time_window.end
        return 'orange' if time_deviation <= BUFFER_MINUTES else 'red'

def format_time(minutes): #format time
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    return f"{hours:02d}:{mins:02d}"


def visualize_route_map(route: List[Customer], arrival_times: List[float], day_name: str, day_idx: int,
                       buffer_minutes: float = BUFFER_MINUTES, save_to_file: bool = True, show_plot: bool = False):
    #geographical route map
    fig, ax = plt.subplots(figsize=MAP_SIZE)
    fig.suptitle(f'Route Map for {day_name} (Alpha = {ALPHA})', fontsize=18)
    ax.set_title(f'Geographical Route (Customer Weight: {ALPHA}, Driver Weight: {1.0 - ALPHA})', fontsize=16)
    ax.set_xlabel('X coordinate (meters)', fontsize=14)
    ax.set_ylabel('Y coordinate (meters)', fontsize=14)

    depot = route[0] #plot depot
    ax.scatter(depot.x, depot.y, c='black', marker='s', s=150, label='Depot')
    ax.text(depot.x, depot.y + 200, 'Depot', ha='center', fontsize=12)

    for i in range(1, len(route) - 1): #plot customers and connections
        c = route[i]
        arrival_time = arrival_times[i]
        time_window = c.time_windows[day_idx]
        satisfaction = calculate_customer_satisfaction(arrival_time, time_window, buffer_minutes)
        color = get_satisfaction_color(satisfaction)
        ax.scatter(c.x, c.y, c=color, s=80, alpha=0.8)
        ax.text(c.x + 100, c.y + 50, f"#{c.id}", fontsize=10, weight='bold')
        ax.text(c.x + 100, c.y - 150, f"{format_time(arrival_time)}", fontsize=10)
        ax.plot([route[i - 1].x, c.x], [route[i - 1].y, c.y], 'k-', alpha=0.3, linewidth=1.5)
        ax.text(c.x - 100, c.y, f"{i}", fontsize=10, bbox=dict(facecolor='white', alpha=0.7))

    ax.plot([route[-2].x, depot.x], [route[-2].y, depot.y], 'k-', alpha=0.3, linewidth=1.5) #last customer to depot
    ax.set_xlim(0, 10000)
    ax.set_ylim(0, 10000)
    ax.grid(True, alpha=0.3)

    legend_patches = [ #satisfaction legend
        mpatches.Patch(color='darkblue', label='Very Satisfied (≥ 0.9)'),
        mpatches.Patch(color='blue', label='Satisfied (≥ 0.7)'),
        mpatches.Patch(color='lightblue', label='Neutral (≥ 0.5)'),
        mpatches.Patch(color='orange', label='Dissatisfied (≥ 0.3)'),
         mpatches.Patch(color='red', label='Very Dissatisfied (< 0.3)')
    ]
    ax.legend(handles=legend_patches, loc='lower right', fontsize=12)

    customers_served = len(route) - 2 #route summary
    total_satisfaction = sum(calculate_customer_satisfaction(arrival_times[i], route[i].time_windows[day_idx], buffer_minutes)
                            for i in range(1, len(route) - 1))
    avg_satisfaction = total_satisfaction / customers_served if customers_served > 0 else 0

    summary_text = (f"Customers served: {customers_served}\n"
                    f"Average satisfaction: {avg_satisfaction:.2f}\n"
                    f"Route start: {format_time(arrival_times[0])}\n"
                    f"Route end: {format_time(arrival_times[-1])}\n"
                    f"Alpha: {ALPHA}")
    fig.text(0.02, 0.02, summary_text, fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    if save_to_file:
        filename= f"route_map_{day_name.lower()}_alpha_{ALPHA}.png"
        plt.savefig(filename, dpi=VISUALIZATION_DPI, bbox_inches='tight')

def visualize_time_windows(route: List[Customer], arrival_times: List[float], day_name: str, day_idx: int,
                          buffer_minutes: float = BUFFER_MINUTES, save_to_file: bool = True, show_plot: bool = False):
    #time window visualization
    fig, ax = plt.subplots(figsize=TIMELINE_SIZE)
    fig.suptitle(f'Time Window Satisfaction for {day_name} (Alpha = {ALPHA})', fontsize=18)
    ax.set_title(f'Customer Time Windows and Arrival Times (Customer Weight: {ALPHA}, Driver Weight: {1.0 - ALPHA})', fontsize=16)
    ax.set_xlabel('Customer (in route order)', fontsize=14)
    ax.set_ylabel('Time (minutes from midnight)', fontsize=14)

    customer_indices = np.arange(1, len(route) - 1)  #skip depot

    ax.axhspan(LUNCH_BREAK_START, LUNCH_BREAK_END, color='lightgray', alpha=0.5, label='Lunch Break') #plot lunch break
    lunch_text = f"Lunch: {format_time(LUNCH_BREAK_START)} - {format_time(LUNCH_BREAK_END)}"
    ax.text(len(customer_indices) / 2, (LUNCH_BREAK_START + LUNCH_BREAK_END) / 2,
            lunch_text, ha='center', va='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))

    for i, idx in enumerate(customer_indices): #time windows and arrivals
        c = route[idx]
        tw = c.time_windows[day_idx]
        arrival = arrival_times[idx]

        ax.plot([i, i], [tw.start, tw.end], 'b-', linewidth=8, alpha=0.3)
        color = get_time_window_color(arrival, tw)
        ax.scatter(i, arrival, c=color, s=120, zorder=3)

        ax.text(i, tw.start - 25, f"#{c.id}", ha='center', fontsize=12)
        ax.text(i, arrival + 15, format_time(arrival), ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
        ax.text(i, tw.end + 15, f"{format_time(tw.start)}-{format_time(tw.end)}", ha='center', fontsize=10)
    ax.set_ylim(DRIVER_START_TIME - 60, DRIVER_FINISH_TIME + 60)

    if len(customer_indices) > 0:
        ax.set_xticks(np.arange(len(customer_indices)))
        ax.set_xticklabels([f"Stop {i + 1}" for i in range(len(customer_indices))], fontsize=10)
    y_ticks = np.arange(DRIVER_START_TIME - 60, DRIVER_FINISH_TIME + 61, 60)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([format_time(t) for t in y_ticks], fontsize=10)
    ax.grid(True, alpha=0.3)

    legend_patches = [ #legend
        mpatches.Patch(color='blue', label='On time'),
        mpatches.Patch(color='green', label=f'Early (≤{buffer_minutes}m)'),
        mpatches.Patch(color='darkgreen', label=f'Very early (>{buffer_minutes}m)'),
        mpatches.Patch(color='orange', label=f'Late (≤{buffer_minutes}m)'),
        mpatches.Patch(color='red', label=f'Very late (>{buffer_minutes}m)'),
        mpatches.Patch(color='lightgray', label='Lunch Break')
    ]
    ax.legend(handles=legend_patches, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=12)
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])

    if save_to_file:
        filename = f"route_timeline_{day_name.lower()}_alpha_{ALPHA}.png"
        plt.savefig(filename, dpi=VISUALIZATION_DPI, bbox_inches='tight')

def visualize_routes(all_routes: List[List[Customer]], all_arrival_times: List[List[float]], day_names: List[str],
                    buffer_minutes: float = BUFFER_MINUTES, save_to_file: bool = True, show_plot: bool = False):
    #separate visualizations for each day
    for day_idx, day_name in enumerate(day_names):
        if day_idx >= len(all_routes) or not all_routes[day_idx]:
            continue
        route = all_routes[day_idx]
        arrival_times = all_arrival_times[day_idx]

        visualize_route_map(route, arrival_times, day_name, day_idx, buffer_minutes, save_to_file, show_plot)
        visualize_time_windows(route, arrival_times, day_name, day_idx, buffer_minutes, save_to_file, show_plot)

def visualize_weekly_stats(all_routes: List[List[Customer]], all_arrival_times: List[List[float]], day_names: List[str],
                          buffer_minutes: float = BUFFER_MINUTES, save_to_file: bool = True, show_plot: bool = False):
    #weekly statistics visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    fig.suptitle(f'Weekly Route Statistics (Alpha = {ALPHA})', fontsize=18)
    customers_served = []
    avg_satisfaction = []
    working_hours = []

    for day_idx, day_name in enumerate(day_names):
        if day_idx >= len(all_routes) or not all_routes[day_idx]:
            customers_served.append(0)
            avg_satisfaction.append(0)
            working_hours.append(0)
            continue

        route = all_routes[day_idx]
        arrival_times = all_arrival_times[day_idx]
        num_customers = len(route) - 2
        customers_served.append(num_customers)

        total_satisfaction = sum(calculate_customer_satisfaction(arrival_times[i], route[i].time_windows[day_idx], buffer_minutes)
                                for i in range(1, len(route) - 1))
        day_avg_satisfaction = total_satisfaction / num_customers if num_customers > 0 else 0
        avg_satisfaction.append(day_avg_satisfaction)

        working_time = (arrival_times[-1] - arrival_times[0]) / 60 if len(arrival_times) >= 2 else 0
        working_hours.append(working_time)

    ax1.bar(day_names, customers_served, color='blue', alpha=0.7) #customers served
    ax1.set_title(f'Customers Served by Day (Customer Weight: {ALPHA}, Driver Weight: {1.0 - ALPHA})', fontsize=14)
    ax1.set_ylabel('Number of Customers', fontsize=12)
    for i, v in enumerate(customers_served):
        ax1.text(i, v + 0.5, str(v), ha='center', fontsize=10)
    ax1.grid(axis='y', alpha=0.3)

    color1, color2 = 'green', 'red' #satisfaction and working hours
    line1 = ax2.plot(day_names, avg_satisfaction, marker='o', color=color1, label='Avg. Satisfaction')
    ax2.set_ylim(0, 1.1)
    ax2.set_title('Satisfaction and Working Hours', fontsize=14)
    ax2.set_ylabel('Satisfaction (0-1)', fontsize=12, color=color1)
    ax2.tick_params(axis='y', labelcolor=color1)
    ax2.grid(axis='y', alpha=0.3)

    ax3 = ax2.twinx()
    line2 = ax3.plot(day_names, working_hours, marker='s', color=color2, label='Working Hours')
    ax3.set_ylabel('Working Hours', fontsize=12, color=color2)
    ax3.tick_params(axis='y', labelcolor=color2)

    lines = line1 + line2
    labels = [line.get_label() for line in lines]
    ax2.legend(lines, labels, loc='upper right')

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    if save_to_file:
        filename = f"weekly_statistics_alpha_{ALPHA}.png"
        plt.savefig(filename, dpi=VISUALIZATION_DPI, bbox_inches='tight')

if __name__ == "__main__":
        results = get_optimization_results()
        all_routes = results['routes']
        all_arrival_times = results['arrival_times']
        days = results['days']

        print("Creating daily visualizations.")
        visualize_routes(all_routes, all_arrival_times, days)
        print("Creating weekly statistics.")
        visualize_weekly_stats(all_routes, all_arrival_times, days)
        print("Completed.")

