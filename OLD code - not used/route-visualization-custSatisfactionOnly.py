import matplotlib

# Force the use of a different backend before importing pyplot
matplotlib.use('Agg')  # Use the non-interactive Agg backend

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from typing import List
import math
import os
import sys
import io
import traceback

# Import from custom modules
from customer_data import Customer, TimeWindow
from main_custSatisfactionOnly_bestImprov import (
    calculate_satisfaction,
    insertion_heuristic,
    two_opt_improvement,
    all_customers
)

# Import constants
from constants import (
    BUFFER_MINUTES, DRIVER_START_TIME, DRIVER_FINISH_TIME,
    LUNCH_BREAK_START, LUNCH_BREAK_END,
    VISUALIZATION_DPI, VISUALIZATION_SIZE
)

# Define larger figure sizes for individual plots
MAP_SIZE = (16, 12)
TIMELINE_SIZE = (16, 12)


def get_optimization_results():
    """Get route data from running the optimization"""
    print("Generating route data...")

    # Run for 5 days (Mon-Fri)
    all_routes = []
    all_arrival_times = []
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

    for day_idx in range(5):
        print(f"Processing {days[day_idx]}...")

        # Step 1: Create initial route using insertion heuristic
        initial_route, initial_cost, _ = insertion_heuristic(
            customers=all_customers,
            day_index=day_idx
        )

        # Step 2: Improve route using 2-opt
        improved_route, improved_cost, arrival_times = two_opt_improvement(
            route=initial_route,
            day_index=day_idx
        )

        # Store results
        all_routes.append(improved_route)
        all_arrival_times.append(arrival_times)

        print(f"  Route created with {len(improved_route) - 2} customers")

    return all_routes, all_arrival_times, days


def visualize_route_map(
        route: List[Customer],
        arrival_times: List[float],
        day_name: str,
        day_idx: int,
        buffer_minutes: float = BUFFER_MINUTES,
        save_to_file: bool = True,
        show_plot: bool = False
):
    """Create and save the geographical route map"""

    # Define color for satisfaction score
    def get_satisfaction_color(satisfaction):
        """Get color gradient based on satisfaction score"""
        if satisfaction >= 0.9:
            return 'darkblue'
        elif satisfaction >= 0.7:
            return 'blue'
        elif satisfaction >= 0.5:
            return 'lightblue'
        elif satisfaction >= 0.3:
            return 'orange'
        else:
            return 'red'

    # Format minutes to HH:MM
    def format_time(minutes):
        hours = int(minutes // 60)
        mins = int(minutes % 60)
        return f"{hours:02d}:{mins:02d}"

    try:
        # Create figure for the map
        fig, ax = plt.subplots(figsize=MAP_SIZE)
        fig.suptitle(f'Route Map for {day_name}', fontsize=18)

        # Set up the axes
        ax.set_title('Geographical Route', fontsize=16)
        ax.set_xlabel('X coordinate (meters)', fontsize=14)
        ax.set_ylabel('Y coordinate (meters)', fontsize=14)

        # Plot depot
        depot = route[0]
        ax.scatter(depot.x, depot.y, c='black', marker='s', s=150, label='Depot')
        ax.text(depot.x, depot.y + 200, 'Depot', ha='center', fontsize=12)

        # Plot customers and connections
        for i in range(1, len(route) - 1):
            c = route[i]
            arrival_time = arrival_times[i]
            time_window = c.time_windows[day_idx]
            satisfaction = calculate_satisfaction(arrival_time, time_window, buffer_minutes)
            color = get_satisfaction_color(satisfaction)

            # Plot customer
            ax.scatter(c.x, c.y, c=color, s=80, alpha=0.8)

            # Add customer ID and arrival time with larger font
            ax.text(c.x + 100, c.y + 50, f"#{c.id}", fontsize=10, weight='bold')
            ax.text(c.x + 100, c.y - 150, f"{format_time(arrival_time)}", fontsize=10)

            # Connect customers in sequence
            ax.plot([route[i - 1].x, c.x], [route[i - 1].y, c.y], 'k-', alpha=0.3, linewidth=1.5)

            # Add sequence number
            ax.text(c.x - 100, c.y, f"{i}", fontsize=10, bbox=dict(facecolor='white', alpha=0.7))

        # Connect last customer to depot
        if len(route) > 2:
            ax.plot([route[-2].x, depot.x], [route[-2].y, depot.y], 'k-', alpha=0.3, linewidth=1.5)

        # Set limits with some padding
        ax.set_xlim(0, 10000)
        ax.set_ylim(0, 10000)
        ax.grid(True, alpha=0.3)

        # Add satisfaction legend
        very_satisfied = mpatches.Patch(color='darkblue', label='Very Satisfied (≥ 0.9)')
        satisfied = mpatches.Patch(color='blue', label='Satisfied (≥ 0.7)')
        neutral = mpatches.Patch(color='lightblue', label='Neutral (≥ 0.5)')
        dissatisfied = mpatches.Patch(color='orange', label='Dissatisfied (≥ 0.3)')
        very_dissatisfied = mpatches.Patch(color='red', label='Very Dissatisfied (< 0.3)')

        ax.legend(handles=[very_satisfied, satisfied, neutral, dissatisfied, very_dissatisfied],
                  loc='lower right', fontsize=12)

        # Add route summary
        customers_served = len(route) - 2  # Excluding depot at start and end
        total_satisfaction = 0
        for i in range(1, len(route) - 1):
            satisfaction = calculate_satisfaction(
                arrival_times[i],
                route[i].time_windows[day_idx],
                buffer_minutes
            )
            total_satisfaction += satisfaction

        avg_satisfaction = total_satisfaction / customers_served if customers_served > 0 else 0

        summary_text = (
            f"Customers served: {customers_served}\n"
            f"Average satisfaction: {avg_satisfaction:.2f}\n"
            f"Route start: {format_time(arrival_times[0])}\n"
            f"Route end: {format_time(arrival_times[-1])}"
        )

        fig.text(0.02, 0.02, summary_text, fontsize=12, bbox=dict(facecolor='white', alpha=0.7))

        # Adjust layout
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])

        # Save to file
        if save_to_file:
            filename = f"route_map_{day_name.lower()}.png"
            print(f"Saving map to {filename}")
            plt.savefig(filename, dpi=VISUALIZATION_DPI, bbox_inches='tight')

        # Show plot
        if show_plot:
            plt.show()
        else:
            plt.close(fig)

        return True

    except Exception as e:
        print(f"Error creating route map for {day_name}: {str(e)}")
        traceback.print_exc()
        plt.close('all')
        return False


def visualize_time_windows(
        route: List[Customer],
        arrival_times: List[float],
        day_name: str,
        day_idx: int,
        buffer_minutes: float = BUFFER_MINUTES,
        save_to_file: bool = True,
        show_plot: bool = False
):
    """Create and save the time window visualization"""

    # Define colors for time window satisfaction
    def get_time_window_color(arrival_time, time_window):
        """Get color based on arrival time relative to time window"""
        if time_window.start <= arrival_time <= time_window.end:
            return 'blue'  # Within window - blue
        elif arrival_time < time_window.start:
            time_deviation = time_window.start - arrival_time
            if time_deviation <= buffer_minutes:
                return 'green'  # Early but within buffer - green
            else:
                return 'darkgreen'  # Very early - dark green
        else:  # arrival_time > time_window.end
            time_deviation = arrival_time - time_window.end
            if time_deviation <= buffer_minutes:
                return 'orange'  # Late but within buffer - orange
            else:
                return 'red'  # Very late - red

    # Format minutes to HH:MM
    def format_time(minutes):
        hours = int(minutes // 60)
        mins = int(minutes % 60)
        return f"{hours:02d}:{mins:02d}"

    try:
        # Create figure for the time windows
        fig, ax = plt.subplots(figsize=TIMELINE_SIZE)
        fig.suptitle(f'Time Window Satisfaction for {day_name}', fontsize=18)

        # Set up the axes
        ax.set_title('Customer Time Windows and Arrival Times', fontsize=16)
        ax.set_xlabel('Customer (in route order)', fontsize=14)
        ax.set_ylabel('Time (minutes from midnight)', fontsize=14)

        customer_indices = np.arange(1, len(route) - 1)  # Skip depot at start and end

        # Plot lunch break as shaded area
        ax.axhspan(LUNCH_BREAK_START, LUNCH_BREAK_END,
                   color='lightgray', alpha=0.5, label='Lunch Break')

        # Add text for lunch break
        lunch_text = f"Lunch: {format_time(LUNCH_BREAK_START)} - {format_time(LUNCH_BREAK_END)}"
        ax.text(len(customer_indices) / 2, (LUNCH_BREAK_START + LUNCH_BREAK_END) / 2,
                lunch_text, ha='center', va='center', fontsize=12,
                bbox=dict(facecolor='white', alpha=0.7))

        # Plot horizontal lines for each time window
        for i, idx in enumerate(customer_indices):
            c = route[idx]
            tw = c.time_windows[day_idx]
            arrival = arrival_times[idx]

            # Plot time window as a horizontal line
            ax.plot([i, i], [tw.start, tw.end], 'b-', linewidth=8, alpha=0.3)

            # Plot arrival time
            color = get_time_window_color(arrival, tw)
            ax.scatter(i, arrival, c=color, s=120, zorder=3)

            # Add customer ID
            ax.text(i, tw.start - 25, f"#{c.id}", ha='center', fontsize=12)

            # Add arrival time
            ax.text(i, arrival + 15, format_time(arrival), ha='center', fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.7))

            # Add time window text
            tw_text = f"{format_time(tw.start)}-{format_time(tw.end)}"
            ax.text(i, tw.end + 15, tw_text, ha='center', fontsize=10)

        # Set y-axis limits to cover the workday
        ax.set_ylim(DRIVER_START_TIME - 60, DRIVER_FINISH_TIME + 60)  # Add 1 hour buffer on each end

        # Set x-axis
        if len(customer_indices) > 0:
            ax.set_xticks(np.arange(len(customer_indices)))
            ax.set_xticklabels([f"Stop {i + 1}" for i in range(len(customer_indices))], fontsize=10)

        # Convert times on y-axis to HH:MM format
        y_ticks = np.arange(DRIVER_START_TIME - 60, DRIVER_FINISH_TIME + 61, 60)  # Every hour
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([format_time(t) for t in y_ticks], fontsize=10)

        # Add grid
        ax.grid(True, alpha=0.3)

        # Add legend for time window satisfaction
        on_time = mpatches.Patch(color='blue', label='On time')
        early_buffer = mpatches.Patch(color='green', label=f'Early (within {buffer_minutes} min buffer)')
        very_early = mpatches.Patch(color='darkgreen', label=f'Very early (beyond {buffer_minutes} min buffer)')
        late_buffer = mpatches.Patch(color='orange', label=f'Late (within {buffer_minutes} min buffer)')
        very_late = mpatches.Patch(color='red', label=f'Very late (beyond {buffer_minutes} min buffer)')
        lunch_break = mpatches.Patch(color='lightgray', label=f'Lunch Break (12:00-12:30)')

        ax.legend(handles=[on_time, early_buffer, very_early, late_buffer, very_late, lunch_break],
                  loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=12)

        # Adjust layout
        plt.tight_layout(rect=[0, 0.1, 1, 0.95])

        # Save to file
        if save_to_file:
            filename = f"route_timeline_{day_name.lower()}.png"
            print(f"Saving timeline to {filename}")
            plt.savefig(filename, dpi=VISUALIZATION_DPI, bbox_inches='tight')

        # Show plot
        if show_plot:
            plt.show()
        else:
            plt.close(fig)

        return True

    except Exception as e:
        print(f"Error creating time window visualization for {day_name}: {str(e)}")
        traceback.print_exc()
        plt.close('all')
        return False


def visualize_routes(
        all_routes: List[List[Customer]],
        all_arrival_times: List[List[float]],
        day_names: List[str],
        buffer_minutes: float = BUFFER_MINUTES,
        save_to_file: bool = True,
        show_plot: bool = False
):
    """
    Create separate visualizations for each day:
    - One for the geographical route map
    - One for the time window satisfaction chart
    """
    # Process each day
    for day_idx, day_name in enumerate(day_names):
        # Skip if no route for this day
        if day_idx >= len(all_routes) or not all_routes[day_idx]:
            print(f"Skipping visualization for {day_name} - no route data")
            continue

        route = all_routes[day_idx]
        arrival_times = all_arrival_times[day_idx]

        print(f"Creating visualizations for {day_name} with {len(route) - 2} customers")

        # Create geographical route map
        map_success = visualize_route_map(
            route=route,
            arrival_times=arrival_times,
            day_name=day_name,
            day_idx=day_idx,
            buffer_minutes=buffer_minutes,
            save_to_file=save_to_file,
            show_plot=show_plot
        )

        # Create time window visualization
        timeline_success = visualize_time_windows(
            route=route,
            arrival_times=arrival_times,
            day_name=day_name,
            day_idx=day_idx,
            buffer_minutes=buffer_minutes,
            save_to_file=save_to_file,
            show_plot=show_plot
        )

        if map_success and timeline_success:
            print(f"Visualizations for {day_name} created successfully.")
        else:
            print(f"Warning: Some visualizations for {day_name} could not be created.")


if __name__ == "__main__":
    try:
        # Get optimization results
        print("Starting route optimization and visualization...")
        all_routes, all_arrival_times, days = get_optimization_results()

        # Create visualizations
        print("\nCreating visualizations...")
        visualize_routes(all_routes, all_arrival_times, days)
        print("Visualization completed successfully!")

    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc()