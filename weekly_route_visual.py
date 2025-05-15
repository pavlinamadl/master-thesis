import matplotlib

matplotlib.use('Agg')  # Use the non-interactive Agg backend

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np
from typing import List, Dict, Tuple
import math
import os
import traceback
from matplotlib.gridspec import GridSpec

# Import from custom modules
from customer_data import Customer, TimeWindow
from time_utils import calculate_customer_satisfaction
from constants import BUFFER_MINUTES, ALPHA, VISUALIZATION_DPI
from optimization_engine import get_optimization_results


def visualize_weekly_sequence_streamlined(
        all_routes: List[List[Customer]],
        all_arrival_times: List[List[float]],
        day_names: List[str],
        day_indices: List[int],
        buffer_minutes: float = BUFFER_MINUTES,
        save_to_file: bool = True,
        show_plot: bool = False,
        figsize: Tuple[int, int] = (24, 16)
):
    """
    Create a streamlined visualization showing daily routes as vertical sequences in columns,
    with larger customer bubbles and simplified text layout.

    Args:
        all_routes: List of routes for each day
        all_arrival_times: List of arrival times for each day
        day_names: Names of the days
        day_indices: Indices of the days (0-4 for Monday-Friday)
        buffer_minutes: Buffer time for calculating customer satisfaction
        save_to_file: Whether to save the visualization to a file
        show_plot: Whether to display the visualization
        figsize: Size of the figure
    """
    try:
        # Format minutes to HH:MM
        def format_time(minutes):
            hours = int(minutes // 60)
            mins = int(minutes % 60)
            return f"{hours:02d}:{mins:02d}"

        # Calculate deviation and satisfaction
        def calculate_deviation(arrival_time, time_window):
            if arrival_time < time_window.start:
                return time_window.start - arrival_time, "early"
            elif arrival_time > time_window.end:
                return arrival_time - time_window.end, "late"
            else:
                return 0, "on time"

        # Get color for deviation
        def get_deviation_color(deviation, deviation_type):
            if deviation == 0:
                return 'blue'  # On time
            elif deviation_type == "early":
                if deviation <= buffer_minutes:
                    return 'green'  # Early but within buffer
                else:
                    return 'darkgreen'  # Very early
            else:  # Late
                if deviation <= buffer_minutes:
                    return 'orange'  # Late but within buffer
                else:
                    return 'red'  # Very late

        # Get color for satisfaction
        def get_satisfaction_color(satisfaction):
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

        # Find the maximum number of customers in any route
        max_customers = max([len(route) - 2 for route in all_routes])  # -2 for depot at start and end
        num_days = len(all_routes)

        # Create a new figure
        fig = plt.figure(figsize=figsize)
        fig.suptitle(f'Weekly Route Sequences (Alpha = {ALPHA})', fontsize=24, y=0.98)

        # Create a grid layout with one column per day
        gs = GridSpec(1, num_days, figure=fig, width_ratios=[1] * num_days)

        # Create axes - one for each day (column)
        axes = []
        for i in range(num_days):
            axes.append(fig.add_subplot(gs[0, i]))

        # Process each day
        for day_idx, (ax, day_name, day_index) in enumerate(zip(axes, day_names, day_indices)):
            route = all_routes[day_idx]
            arrival_times = all_arrival_times[day_idx]

            # Skip depot at start and end
            customer_route = route[1:-1]
            customer_arrivals = arrival_times[1:-1]

            # Set up the axis
            ax.set_title(f'{day_name} Route', fontsize=20, pad=15)

            # If no customers, show a message
            if not customer_route:
                ax.text(0.5, 0.5, "No route data", ha='center', va='center',
                        fontsize=16, transform=ax.transAxes)
                ax.axis('off')
                continue

            # Set axis limits - increase bottom margin for summary
            ax.set_xlim(-1.5, 2.5)  # Give more space for text
            ax.set_ylim(-2, len(customer_route))  # One unit per customer, plus space at bottom

            # Draw vertical line representing the route
            ax.plot([0, 0], [0, len(customer_route) - 1], 'k-', linewidth=3)

            # Plot each customer as a point on the route
            for i, (customer, arrival_time) in enumerate(zip(customer_route, customer_arrivals)):
                time_window = customer.time_windows[day_index]

                # Calculate deviation and satisfaction
                deviation, deviation_type = calculate_deviation(arrival_time, time_window)
                satisfaction = calculate_customer_satisfaction(arrival_time, time_window, buffer_minutes)

                # Get colors
                dev_color = get_deviation_color(deviation, deviation_type)
                sat_color = get_satisfaction_color(satisfaction)

                # Plot larger customer node
                circle = plt.Circle((0, i), 0.3, color=sat_color, zorder=3, edgecolor='black')
                ax.add_patch(circle)

                # Add larger customer ID within the node
                ax.text(0, i, str(customer.id), ha='center', va='center',
                        fontsize=14, weight='bold', color='white')

                # Add must-serve marker if applicable
                if customer.must_serve:
                    ax.text(0.35, i + 0.25, "*", ha='center', va='center', fontsize=20,
                            weight='bold', color='black')

                # Format time window and arrival time - right next to node
                tw_text = f"{format_time(time_window.start)}-{format_time(time_window.end)}"
                arr_text = f"{format_time(arrival_time)}"

                # Format deviation text in brackets
                if deviation > 0:
                    dev_hours = int(deviation // 60)
                    dev_minutes = int(deviation % 60)
                    if dev_hours > 0:
                        dev_text = f"({deviation_type.capitalize()}: {dev_hours}h {dev_minutes}m)"
                    else:
                        dev_text = f"({deviation_type.capitalize()}: {dev_minutes}m)"
                else:
                    dev_text = "(On Time)"

                # Add combined text right next to bubble
                combined_text = f"TW: {tw_text}\nArr: {arr_text} {dev_text}"
                text = ax.text(0.4, i, combined_text, ha='left', va='center', fontsize=11)

                # Make arrival time and deviation text bold and colored
                txt_bbox = dict(boxstyle="round,pad=0.3", fc='white', ec=dev_color, alpha=0.7)
                text.set_bbox(txt_bbox)

                # Add satisfaction score as small text at the bottom of the bubble
                sat_text = f"{satisfaction:.2f}"
                ax.text(0, i - 0.35, sat_text, ha='center', va='center', fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.7))

            # Hide axis ticks and labels
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

            # Add day summary at the bottom - moved further down to avoid overlap
            satisfaction_avg = sum(calculate_customer_satisfaction(
                arrival_times[i], route[i].time_windows[day_index], buffer_minutes)
                                   for i in range(1, len(route) - 1)) / len(customer_route) if customer_route else 0

            summary_text = (
                f"Customers: {len(customer_route)}\n"
                f"Avg. Satisfaction: {satisfaction_avg:.2f}\n"
                f"Start: {format_time(arrival_times[0])}\n"
                f"End: {format_time(arrival_times[-1])}"
            )

            # Place the summary at the bottom of the column with extra space
            ax.text(0, -1.5, summary_text, ha='center', va='center', fontsize=12,
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.4'))

        # Add legend at the bottom
        legend_ax = fig.add_axes([0.1, 0.02, 0.8, 0.05], frameon=False)
        legend_ax.axis('off')

        # Create legend items
        on_time = mpatches.Patch(color='blue', label='On time')
        early_buffer = mpatches.Patch(color='green', label=f'Early (≤{buffer_minutes}m)')
        very_early = mpatches.Patch(color='darkgreen', label=f'Very early (>{buffer_minutes}m)')
        late_buffer = mpatches.Patch(color='orange', label=f'Late (≤{buffer_minutes}m)')
        very_late = mpatches.Patch(color='red', label=f'Very late (>{buffer_minutes}m)')
        must_serve = mpatches.Patch(color='white', label='* Must-serve customer', edgecolor='black')

        # Add satisfaction legend items
        very_satisfied = mpatches.Patch(color='darkblue', label='Very Satisfied (≥ 0.9)')
        satisfied = mpatches.Patch(color='blue', label='Satisfied (≥ 0.7)')
        neutral = mpatches.Patch(color='lightblue', label='Neutral (≥ 0.5)')
        dissatisfied = mpatches.Patch(color='orange', label='Dissatisfied (≥ 0.3)')
        very_dissatisfied = mpatches.Patch(color='red', label='Very Dissatisfied (< 0.3)')

        # Add the legends in two rows
        legend1 = legend_ax.legend(handles=[on_time, early_buffer, very_early, late_buffer, very_late, must_serve],
                                   loc='upper center', ncol=6, fontsize=10, bbox_to_anchor=(0.5, 0.9))
        legend_ax.add_artist(legend1)

        legend_ax.legend(handles=[very_satisfied, satisfied, neutral, dissatisfied, very_dissatisfied],
                         loc='upper center', ncol=5, fontsize=10, bbox_to_anchor=(0.5, 0.4))

        # Add alpha information
        fig.text(0.5, 0.01, f"Alpha = {ALPHA} (Customer weight: {ALPHA}, Driver weight: {1.0 - ALPHA})",
                 ha='center', fontsize=12)

        # Adjust layout
        plt.tight_layout(rect=[0, 0.08, 1, 0.95])

        # Save to file
        if save_to_file:
            filename = f"weekly_route_streamlined_alpha_{ALPHA}.png"
            print(f"Saving weekly route sequences to {filename}")
            plt.savefig(filename, dpi=VISUALIZATION_DPI, bbox_inches='tight')

        # Show plot
        if show_plot:
            plt.show()
        else:
            plt.close(fig)

        return True

    except Exception as e:
        print(f"Error creating weekly route sequence visualization: {str(e)}")
        traceback.print_exc()
        plt.close('all')
        return False


def visualize_customer_consistency(
        all_routes: List[List[Customer]],
        day_names: List[str],
        day_indices: List[int],
        save_to_file: bool = True,
        show_plot: bool = False,
        figsize: Tuple[int, int] = (16, 14)  # Increased height for more space at bottom
):
    """
    Create a visualization showing which customers are served on which days
    and their position in each route.
    """
    try:
        # Get all unique customer IDs
        all_customer_ids = set()
        for route in all_routes:
            for customer in route:
                if customer.id != 0:  # Skip depot
                    all_customer_ids.add(customer.id)

        # Sort customer IDs
        customer_ids = sorted(all_customer_ids)

        # Create a matrix showing customers per day and their positions
        customer_matrix = {}
        for customer_id in customer_ids:
            customer_matrix[customer_id] = {}
            for day_idx, day_name in enumerate(day_names):
                route = all_routes[day_idx]
                # Find customer in route
                positions = [i for i, c in enumerate(route) if c.id == customer_id]
                if positions:
                    customer_matrix[customer_id][day_name] = positions[0]  # Position in route
                else:
                    customer_matrix[customer_id][day_name] = None  # Not in route

        # Find if any customer is must-serve
        must_serve = {}
        for route in all_routes:
            for customer in route:
                if customer.id != 0:  # Skip depot
                    must_serve[customer.id] = getattr(customer, 'must_serve', False)

        # Create figure with extra space at the bottom
        fig, ax = plt.subplots(figsize=figsize)

        # Adjust the position of the main axes to leave room at the bottom
        ax.set_position([0.1, 0.2, 0.8, 0.7])  # [left, bottom, width, height]

        fig.suptitle(f'Customer Visit Consistency Across Week (Alpha = {ALPHA})', fontsize=20)

        # Set up the grid
        num_customers = len(customer_ids)
        num_days = len(day_names)

        # Create the heatmap
        heatmap_data = np.zeros((num_customers, num_days))
        for i, customer_id in enumerate(customer_ids):
            for j, day_name in enumerate(day_names):
                position = customer_matrix[customer_id][day_name]
                heatmap_data[i, j] = position if position is not None else -1

        # Create custom colormap with "not in route" color
        cmap = plt.cm.viridis.copy()
        cmap.set_under('lightgray')  # Color for "not in route"

        # Plot the heatmap
        im = ax.imshow(heatmap_data, cmap=cmap, aspect='auto')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Position in Route')

        # Set ticks and labels
        ax.set_xticks(np.arange(num_days))
        ax.set_yticks(np.arange(num_customers))

        # Set horizontal day labels without rotation
        ax.set_xticklabels(day_names, fontsize=12, fontweight='bold', rotation=0)
        ax.set_yticklabels([f"{cid} {'*' if must_serve.get(cid, False) else ''}" for cid in customer_ids], fontsize=10)

        # Loop over data dimensions and create text annotations
        for i in range(num_customers):
            for j in range(num_days):
                position = heatmap_data[i, j]
                if position >= 0:
                    text = ax.text(j, i, f"{int(position)}",
                                   ha="center", va="center",
                                   color="white" if position > 10 else "black")
                else:
                    text = ax.text(j, i, "X", ha="center", va="center", color="red")

        # Add title
        ax.set_title("Position of each customer in daily routes (X = not in route)", pad=20)

        # Add explanatory notes in the space below the graph
        notes_text = "* indicates must-serve customer\n"
        notes_text += f"Alpha = {ALPHA} (Customer weight: {ALPHA}, Driver weight: {1.0 - ALPHA})"

        # Add notes at the bottom of the figure (not the axis)
        fig.text(0.5, 0.05, notes_text, ha='center', fontsize=12,
                 bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

        # Save to file
        if save_to_file:
            filename = f"customer_consistency_alpha_{ALPHA}.png"
            print(f"Saving customer consistency visualization to {filename}")
            plt.savefig(filename, dpi=VISUALIZATION_DPI, bbox_inches='tight')

        # Show plot
        if show_plot:
            plt.show()
        else:
            plt.close(fig)

        return True

    except Exception as e:
        print(f"Error creating customer consistency visualization: {str(e)}")
        traceback.print_exc()
        plt.close('all')
        return False


# Add these visualizations to the route-visualization-custAndDriverSatisfaction.py file
if __name__ == "__main__":
    try:
        # Get optimization results from central engine
        print("Starting route optimization and visualization...")
        print(f"Using alpha = {ALPHA} (customer weight: {ALPHA}, driver weight: {1.0 - ALPHA})")

        results = get_optimization_results()
        all_routes = results['routes']
        all_arrival_times = results['arrival_times']
        days = results['days']
        day_indices = results['day_indices']

        # Create the new visualizations
        print("\nCreating streamlined weekly route visualization...")
        visualize_weekly_sequence_streamlined(all_routes, all_arrival_times, days, day_indices)

        print("\nCreating customer consistency visualization...")
        visualize_customer_consistency(all_routes, days, day_indices)

        print("Visualization completed successfully!")

    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc()