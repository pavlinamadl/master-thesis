import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from typing import List, Tuple
import traceback
from matplotlib.gridspec import GridSpec
from customer_data import Customer
from time_utils import calculate_customer_satisfaction
from constants import BUFFER_MINUTES, ALPHA, VISUALIZATION_DPI
from optimization_engine import get_optimization_results

def format_time(minutes): #format time
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    return f"{hours:02d}:{mins:02d}"

def calculate_deviation(arrival_time, time_window): #calc. deviation and type
    if arrival_time < time_window.start:
        return time_window.start - arrival_time, "early"
    elif arrival_time > time_window.end:
        return arrival_time - time_window.end, "late"
    else:
        return 0, "on time"

def get_deviation_color(deviation, deviation_type): #color for deviation
    if deviation == 0:
        return 'blue'
    elif deviation_type == "early":
        return 'green' if deviation <= BUFFER_MINUTES else 'darkgreen'
    else: return 'orange' if deviation <= BUFFER_MINUTES else 'red'

def get_satisfaction_color(satisfaction): #color for satisfaction

    if satisfaction >= 0.9: return 'darkblue'
    elif satisfaction >= 0.7: return 'blue'
    elif satisfaction >= 0.5: return 'lightblue'
    elif satisfaction >= 0.3: return 'orange'
    else: return 'red'

def visualize_weekly_sequence_streamlined(all_routes: List[List[Customer]], all_arrival_times: List[List[float]],
                                        day_names: List[str], day_indices: List[int],
                                        buffer_minutes: float = BUFFER_MINUTES, save_to_file: bool = True,
                                        show_plot: bool = False, figsize: Tuple[int, int] = (24, 16)):
    #streamlined visualization
    max_customers = max([len(route) - 2 for route in all_routes])
    num_days = len(all_routes)

    fig = plt.figure(figsize=figsize)
    fig.suptitle(f'Weekly Route Sequences (Alpha = {ALPHA})', fontsize=24, y=0.98)
    gs = GridSpec(1, num_days, figure=fig, width_ratios=[1] * num_days)
    axes = [fig.add_subplot(gs[0, i]) for i in range(num_days)]

    for day_idx, (ax, day_name, day_index) in enumerate(zip(axes, day_names, day_indices)):
        route = all_routes[day_idx]
        arrival_times = all_arrival_times[day_idx]

        customer_route = route[1:-1] #without depot
        customer_arrivals = arrival_times[1:-1]
        ax.set_title(f'{day_name} Route', fontsize=20, pad=15)

        if not customer_route:
            ax.text(0.5, 0.5, "No route data", ha='center', va='center', fontsize=16, transform=ax.transAxes)
            ax.axis('off')
            continue

        ax.set_xlim(-1.5, 2.5)
        ax.set_ylim(-2, len(customer_route))

        ax.plot([0, 0], [0, len(customer_route) - 1], 'k-', linewidth=3) #route line

        for i, (customer, arrival_time) in enumerate(zip(customer_route, customer_arrivals)): #plot customers
            time_window = customer.time_windows[day_index]
            deviation, deviation_type = calculate_deviation(arrival_time, time_window)
            satisfaction = calculate_customer_satisfaction(arrival_time, time_window, buffer_minutes)
            dev_color = get_deviation_color(deviation, deviation_type)
            sat_color = get_satisfaction_color(satisfaction)
            # Customer node
            circle = plt.Circle((0, i), 0.3, color=sat_color, zorder=3, edgecolor='black')
            ax.add_patch(circle)
            ax.text(0, i, str(customer.id), ha='center', va='center', fontsize=14, weight='bold', color='white')

            tw_text = f"{format_time(time_window.start)}-{format_time(time_window.end)}" #time information
            arr_text = f"{format_time(arrival_time)}"

            if deviation > 0:
                dev_hours = int(deviation // 60)
                dev_minutes = int(deviation % 60)
                dev_text = f"({deviation_type.capitalize()}: {dev_hours}h {dev_minutes}m)" if dev_hours > 0 else f"({deviation_type.capitalize()}: {dev_minutes}m)"
            else: dev_text = "(On Time)"

            combined_text = f"TW: {tw_text}\nArr: {arr_text} {dev_text}"
            text = ax.text(0.4, i, combined_text, ha='left', va='center', fontsize=11)
            txt_bbox = dict(boxstyle="round,pad=0.3", fc='white', ec=dev_color, alpha=0.7)
            text.set_bbox(txt_bbox)

            ax.text(0, i - 0.35, f"{satisfaction:.2f}", ha='center', va='center', fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.7)) #satisfaction score

        ax.set_xticks([]) #hide axes
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        satisfaction_avg = sum(calculate_customer_satisfaction(arrival_times[i], route[i].time_windows[day_index], buffer_minutes)
                                for i in range(1, len(route) - 1)) / len(customer_route) if customer_route else 0 #day summary

        summary_text = (f"Customers: {len(customer_route)}\n"
                        f"Avg. Satisfaction: {satisfaction_avg:.2f}\n"
                        f"Start: {format_time(arrival_times[0])}\n"
                        f"End: {format_time(arrival_times[-1])}")
        ax.text(0, -1.5, summary_text, ha='center', va='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.4'))

    legend_ax = fig.add_axes([0.1, 0.02, 0.8, 0.05], frameon=False) #legend
    legend_ax.axis('off')

    timing_patches = [ #timing legend
        mpatches.Patch(color='blue', label='On time'),
        mpatches.Patch(color='green', label=f'Early (≤{buffer_minutes}m)'),
        mpatches.Patch(color='darkgreen', label=f'Very early (>{buffer_minutes}m)'),
        mpatches.Patch(color='orange', label=f'Late (≤{buffer_minutes}m)'),
        mpatches.Patch(color='red', label=f'Very late (>{buffer_minutes}m)'),
        mpatches.Patch(color='white', label='* Must-serve customer', edgecolor='black')
    ]

    satisfaction_patches = [ #satisfaction legend
        mpatches.Patch(color='darkblue', label='Very Satisfied (≥ 0.9)'),
        mpatches.Patch(color='blue', label='Satisfied (≥ 0.7)'),
        mpatches.Patch(color='lightblue', label='Neutral (≥ 0.5)'),
        mpatches.Patch(color='orange', label='Dissatisfied (≥ 0.3)'),
        mpatches.Patch(color='red', label='Very Dissatisfied (< 0.3)')
    ]

    legend1 = legend_ax.legend(handles=timing_patches, loc='upper center', ncol=6, fontsize=10, bbox_to_anchor=(0.5, 0.9))
    legend_ax.add_artist(legend1)
    legend_ax.legend(handles=satisfaction_patches, loc='upper center', ncol=5, fontsize=10, bbox_to_anchor=(0.5, 0.4))

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])

    if save_to_file:
        filename = f"weekly_route_streamlined_alpha_{ALPHA}.png"
        plt.savefig(filename, dpi=VISUALIZATION_DPI, bbox_inches='tight')

def visualize_customer_consistency(all_routes: List[List[Customer]], day_names: List[str], day_indices: List[int],
                                  save_to_file: bool = True, show_plot: bool = False,
                                  figsize: Tuple[int, int] = (16, 14)):
    #which customers are served on which days
    all_customer_ids = set() #get all customers ID
    for route in all_routes:
        for customer in route:
            if customer.id != 0:
                all_customer_ids.add(customer.id)
    customer_ids = sorted(all_customer_ids)
    customer_matrix = {} #customer matrix
    must_serve = {}
    for customer_id in customer_ids:
        customer_matrix[customer_id] = {}
        for day_idx, day_name in enumerate(day_names):
            route = all_routes[day_idx]
            positions = [i for i, c in enumerate(route) if c.id == customer_id]
            customer_matrix[customer_id][day_name] = positions[0] if positions else None

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_position([0.1, 0.2, 0.8, 0.7])
    fig.suptitle(f'Customer Visit Consistency Across Week (Alpha = {ALPHA})', fontsize=20)
    num_customers = len(customer_ids)
    num_days = len(day_names)

    heatmap_data = np.zeros((num_customers, num_days)) #create heatmap data
    for i, customer_id in enumerate(customer_ids):
        for j, day_name in enumerate(day_names):
            position = customer_matrix[customer_id][day_name]
            heatmap_data[i, j] = position if position is not None else -1

    cmap = plt.cm.viridis.copy() #plot heatmap
    cmap.set_under('lightgray')
    im = ax.imshow(heatmap_data, cmap=cmap, aspect='auto')

    cbar = plt.colorbar(im, ax=ax) #colorbar
    cbar.set_label('Position in Route')
    ax.set_xticks(np.arange(num_days)) #labels
    ax.set_yticks(np.arange(num_customers))
    ax.set_xticklabels(day_names, fontsize=12, fontweight='bold', rotation=0)
    ax.set_yticklabels([f"{cid} {'*' if must_serve.get(cid, False) else ''}" for cid in customer_ids], fontsize=10)

    for i in range(num_customers): #text annotations
        for j in range(num_days):
            position = heatmap_data[i, j]
            if position >= 0:
                text = ax.text(j, i, f"{int(position)}", ha="center", va="center", color="white" if position > 10 else "black")
            else: text = ax.text(j, i, "X", ha="center", va="center", color="red")
    ax.set_title("Position of each customer in daily routes", pad=20)

    if save_to_file:
        filename = f"customer_consistency_alpha_{ALPHA}.png"
        plt.savefig(filename, dpi=VISUALIZATION_DPI, bbox_inches='tight')

if __name__ == "__main__":
    print(f"Creating weekly route visualization for alpha = {ALPHA}.")

    results = get_optimization_results()
    all_routes = results['routes']
    all_arrival_times = results['arrival_times']
    days = results['days']
    day_indices = results['day_indices']

    visualize_weekly_sequence_streamlined(all_routes, all_arrival_times, days, day_indices)
    visualize_customer_consistency(all_routes, days, day_indices)
    print("Completed.")