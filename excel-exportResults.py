import pandas as pd
import numpy as np
from typing import List, Dict
import os
import traceback
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

# Import from custom modules
from customer_data import Customer, TimeWindow, all_customers

# Import from modular structure
from route_construction import insertion_heuristic
from route_optimization import tabu_enhanced_two_opt
from route_enhancement import attempt_additional_insertions
from time_utils import calculate_customer_satisfaction
from satisfaction_metrics import calculate_working_time

# Import constants
from constants import (
    BUFFER_MINUTES, DRIVER_START_TIME, DRIVER_FINISH_TIME,
    W_CUSTOMER, W_DRIVER
)


def format_time(minutes):
    """Format minutes to HH:MM"""
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    return f"{hours:02d}:{mins:02d}"


def get_optimization_results():
    """Run the optimization and get route data for all days"""
    print("Generating route data...")

    # Run for 5 days (Mon-Fri)
    all_routes = []
    all_arrival_times = []
    all_working_times = []
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    all_unvisited_customers = []

    for day_idx in range(5):
        print(f"Processing {days[day_idx]}...")

        # Previous routes and working times for driver satisfaction calculation
        previous_routes = all_routes.copy()
        previous_working_times = all_working_times.copy()

        # Step 1: Create initial route using insertion heuristic
        initial_route, initial_cost, _, initial_cust_sat, initial_driver_sat = insertion_heuristic(
            customers=all_customers,
            day_index=day_idx,
            previous_routes=previous_routes,
            previous_working_times=previous_working_times
        )

        # Identify unvisited customers from the initial route
        visited_customer_ids = {customer.id for customer in initial_route}
        unvisited_customers = [customer for customer in all_customers if
                               customer.id not in visited_customer_ids and customer.id != 0]  # Exclude depot

        # Step 2: Improve route using tabu-enhanced 2-opt
        improved_route, improved_cost, improved_arrival_times, improved_cust_sat, improved_driver_sat = tabu_enhanced_two_opt(
            route=initial_route,
            day_index=day_idx,
            previous_routes=previous_routes,
            previous_working_times=previous_working_times
        )

        # Step 3: Attempt to insert additional unvisited customers
        final_route, final_cost, arrival_times, customer_sat, driver_sat = attempt_additional_insertions(
            route=improved_route,
            unvisited_customers=unvisited_customers,
            day_index=day_idx,
            previous_routes=previous_routes,
            previous_working_times=previous_working_times
        )

        # Calculate working time for this route
        working_time = calculate_working_time(final_route)

        # Store results
        all_routes.append(final_route)
        all_arrival_times.append(arrival_times)
        all_working_times.append(working_time)
        all_unvisited_customers.append(unvisited_customers)

        print(f"  Route created with {len(final_route) - 2} customers")

    return all_routes, all_arrival_times, days, all_unvisited_customers


def create_routes_comparison_sheet(wb, all_routes, all_arrival_times, days):
    """Create a sheet showing all five routes side by side"""
    print("Creating routes comparison sheet...")

    ws = wb.create_sheet("Routes Comparison")
    ws.title = "Routes Comparison"

    # Set column widths
    for col in range(1, 16):  # Assuming 3 columns per day (ID, Time Window, Arrival)
        ws.column_dimensions[get_column_letter(col)].width = 12

    # Create header row with day names
    header_fill = PatternFill(start_color="D3D3D3", end_color="D3D3D3", fill_type="solid")
    header_font = Font(bold=True)

    # Define header rows
    ws.cell(row=1, column=1, value="Route Comparison").font = Font(bold=True, size=14)

    col = 1
    for day in days:
        ws.merge_cells(start_row=2, start_column=col, end_row=2, end_column=col + 2)
        ws.cell(row=2, column=col, value=day).font = header_font
        ws.cell(row=2, column=col).alignment = Alignment(horizontal='center')
        ws.cell(row=2, column=col).fill = header_fill

        ws.cell(row=3, column=col, value="Stop #").font = header_font
        ws.cell(row=3, column=col).fill = header_fill

        ws.cell(row=3, column=col + 1, value="Customer ID").font = header_font
        ws.cell(row=3, column=col + 1).fill = header_fill

        ws.cell(row=3, column=col + 2, value="Arrival Time").font = header_font
        ws.cell(row=3, column=col + 2).fill = header_fill

        col += 3

    # Find the maximum number of stops in any route
    max_stops = max([len(route) for route in all_routes])

    # Fill in the data for each day
    for day_idx, day in enumerate(days):
        route = all_routes[day_idx]
        arrival_times = all_arrival_times[day_idx]

        col = day_idx * 3 + 1

        # Fill in stop information
        for stop_idx in range(len(route)):
            customer = route[stop_idx]
            row = stop_idx + 4  # Starting from row 4 (after headers)

            # Stop number (0 is depot, so use "D" for clarity)
            ws.cell(row=row, column=col, value="D" if customer.id == 0 else stop_idx)

            # Customer ID
            ws.cell(row=row, column=col + 1, value=customer.id)

            # Arrival time
            if stop_idx < len(arrival_times):
                arrival_time = arrival_times[stop_idx]
                ws.cell(row=row, column=col + 2, value=format_time(arrival_time))

    # Add summary at the bottom
    summary_row = max_stops + 5
    ws.cell(row=summary_row, column=1, value="Summary").font = Font(bold=True)

    for day_idx, day in enumerate(days):
        col = day_idx * 3 + 1
        route = all_routes[day_idx]

        # Number of customers served (excluding depot at start and end)
        customers_served = len(route) - 2
        ws.cell(row=summary_row + 1, column=col, value="Customers:")
        ws.cell(row=summary_row + 1, column=col + 1, value=customers_served)

        # Route duration
        if len(all_arrival_times[day_idx]) >= 2:
            start_time = all_arrival_times[day_idx][0]
            end_time = all_arrival_times[day_idx][-1]
            duration_mins = end_time - start_time
            duration_hrs = duration_mins / 60

            ws.cell(row=summary_row + 2, column=col, value="Duration:")
            ws.cell(row=summary_row + 2, column=col + 1, value=f"{duration_hrs:.2f} hrs")

            ws.cell(row=summary_row + 3, column=col, value="Start:")
            ws.cell(row=summary_row + 3, column=col + 1, value=format_time(start_time))

            ws.cell(row=summary_row + 4, column=col, value="End:")
            ws.cell(row=summary_row + 4, column=col + 1, value=format_time(end_time))

    return ws


def create_customer_details_sheet(wb, all_routes, all_arrival_times, days):
    """Create a sheet with customer service details"""
    print("Creating customer service details sheet...")

    ws = wb.create_sheet("Customer Service Details")
    ws.title = "Customer Service Details"

    # Set column widths
    columns = ["Customer ID", "Day", "Time Window", "Arrival Time", "Status", "Deviation", "Satisfaction"]
    for col in range(1, len(columns) + 1):
        ws.column_dimensions[get_column_letter(col)].width = 15

    # Create header row
    header_fill = PatternFill(start_color="D3D3D3", end_color="D3D3D3", fill_type="solid")
    header_font = Font(bold=True)

    ws.cell(row=1, column=1, value="Customer Service Details").font = Font(bold=True, size=14)

    for col, header in enumerate(columns, 1):
        ws.cell(row=2, column=col, value=header).font = header_font
        ws.cell(row=2, column=col).fill = header_fill

    # Fill in the data
    row = 3
    for day_idx, day in enumerate(days):
        route = all_routes[day_idx]
        arrival_times = all_arrival_times[day_idx]

        # Skip depot at start and end
        for i in range(1, len(route) - 1):
            customer = route[i]
            arrival_time = arrival_times[i]
            time_window = customer.time_windows[day_idx]

            # Calculate satisfaction
            satisfaction = calculate_customer_satisfaction(arrival_time, time_window, BUFFER_MINUTES)

            # Determine status and deviation
            if arrival_time < time_window.start:
                deviation = time_window.start - arrival_time
                status = "Early"
            elif arrival_time > time_window.end:
                deviation = arrival_time - time_window.end
                status = "Late"
            else:
                deviation = 0
                status = "On Time"

            # Format time window and deviation
            time_window_str = f"{format_time(time_window.start)}-{format_time(time_window.end)}"
            deviation_str = f"{deviation // 60}h {deviation % 60}m" if deviation > 0 else "0m"

            # Color-code satisfaction
            if satisfaction >= 0.9:
                sat_fill = PatternFill(start_color="90EE90", end_color="90EE90", fill_type="solid")  # Light green
            elif satisfaction >= 0.7:
                sat_fill = PatternFill(start_color="C1FFC1", end_color="C1FFC1", fill_type="solid")  # Pale green
            elif satisfaction >= 0.5:
                sat_fill = PatternFill(start_color="FFFF99", end_color="FFFF99", fill_type="solid")  # Light yellow
            elif satisfaction >= 0.3:
                sat_fill = PatternFill(start_color="FFB347", end_color="FFB347", fill_type="solid")  # Orange
            else:
                sat_fill = PatternFill(start_color="FF6961", end_color="FF6961", fill_type="solid")  # Red

            # Write data to sheet
            ws.cell(row=row, column=1, value=customer.id)
            ws.cell(row=row, column=2, value=day)
            ws.cell(row=row, column=3, value=time_window_str)
            ws.cell(row=row, column=4, value=format_time(arrival_time))
            ws.cell(row=row, column=5, value=status)
            ws.cell(row=row, column=6, value=deviation_str)
            ws.cell(row=row, column=7, value=f"{satisfaction:.2f}")
            ws.cell(row=row, column=7).fill = sat_fill

            row += 1

    return ws


def create_customers_by_day_sheet(wb, all_routes, all_arrival_times, all_unvisited_customers, days):
    """Create a sheet showing which customers are served on which days"""
    print("Creating customers by day sheet...")

    ws = wb.create_sheet("Customers By Day")
    ws.title = "Customers By Day"

    # Set column widths
    ws.column_dimensions['A'].width = 15  # Customer ID
    for col in range(2, 7):  # 5 days
        ws.column_dimensions[get_column_letter(col)].width = 15

    # Create header row
    header_fill = PatternFill(start_color="D3D3D3", end_color="D3D3D3", fill_type="solid")
    header_font = Font(bold=True)

    ws.cell(row=1, column=1, value="Customer Visits By Day").font = Font(bold=True, size=14)

    ws.cell(row=2, column=1, value="Customer ID").font = header_font
    ws.cell(row=2, column=1).fill = header_fill

    for col, day in enumerate(days, 2):
        ws.cell(row=2, column=col, value=day).font = header_font
        ws.cell(row=2, column=col).fill = header_fill

    # Collect all unique customer IDs (excluding depot)
    all_customer_ids = set()
    for day_idx in range(len(days)):
        route = all_routes[day_idx]
        for customer in route:
            if customer.id != 0:  # Exclude depot
                all_customer_ids.add(customer.id)

        # Include unvisited customers
        for customer in all_unvisited_customers[day_idx]:
            all_customer_ids.add(customer.id)

    # Sort customer IDs
    sorted_customer_ids = sorted(all_customer_ids)

    # Create dictionary to map customer IDs to rows
    customer_rows = {}
    for i, customer_id in enumerate(sorted_customer_ids):
        row = i + 3  # Starting from row 3
        customer_rows[customer_id] = row
        ws.cell(row=row, column=1, value=customer_id)

    # Fill in the data for each day
    for day_idx, day in enumerate(days):
        route = all_routes[day_idx]
        arrival_times = all_arrival_times[day_idx]
        col = day_idx + 2  # Column for this day

        # Mark visited customers
        for i in range(1, len(route) - 1):  # Skip depot
            customer = route[i]
            row = customer_rows[customer.id]
            arrival_time = arrival_times[i]

            # Calculate satisfaction
            satisfaction = calculate_customer_satisfaction(
                arrival_time,
                customer.time_windows[day_idx],
                BUFFER_MINUTES
            )

            # Color code cell by satisfaction
            if satisfaction >= 0.9:
                fill = PatternFill(start_color="90EE90", end_color="90EE90", fill_type="solid")  # Light green
            elif satisfaction >= 0.7:
                fill = PatternFill(start_color="C1FFC1", end_color="C1FFC1", fill_type="solid")  # Pale green
            elif satisfaction >= 0.5:
                fill = PatternFill(start_color="FFFF99", end_color="FFFF99", fill_type="solid")  # Light yellow
            elif satisfaction >= 0.3:
                fill = PatternFill(start_color="FFB347", end_color="FFB347", fill_type="solid")  # Orange
            else:
                fill = PatternFill(start_color="FF6961", end_color="FF6961", fill_type="solid")  # Red

            # Add arrival time and format the cell
            ws.cell(row=row, column=col, value=format_time(arrival_time))
            ws.cell(row=row, column=col).fill = fill

        # Mark unvisited customers as "Not Visited"
        for customer in all_unvisited_customers[day_idx]:
            row = customer_rows[customer.id]
            if ws.cell(row=row, column=col).value is None:
                ws.cell(row=row, column=col, value="Not Visited")
                ws.cell(row=row, column=col).fill = PatternFill(start_color="D3D3D3", end_color="D3D3D3",
                                                                fill_type="solid")  # Gray

    # Add a summary at the bottom
    summary_row = len(sorted_customer_ids) + 4
    ws.cell(row=summary_row, column=1, value="Summary").font = Font(bold=True)

    # Count visits by day
    for day_idx, day in enumerate(days):
        col = day_idx + 2
        route = all_routes[day_idx]

        visits = len(route) - 2  # Exclude depot at start and end
        total_customers = len(sorted_customer_ids)
        percent_served = (visits / total_customers) * 100 if total_customers > 0 else 0

        ws.cell(row=summary_row + 1, column=1, value="Customers Served:")
        ws.cell(row=summary_row + 1, column=col, value=visits)

        ws.cell(row=summary_row + 2, column=1, value="Total Customers:")
        ws.cell(row=summary_row + 2, column=col, value=total_customers)

        ws.cell(row=summary_row + 3, column=1, value="Percent Served:")
        ws.cell(row=summary_row + 3, column=col, value=f"{percent_served:.1f}%")

    return ws


def create_excel_report():
    """Create an Excel workbook with multiple sheets for the route optimization results"""
    try:
        print("Starting optimization and Excel report generation...")

        # Get optimization results
        all_routes, all_arrival_times, days, all_unvisited_customers = get_optimization_results()

        # Create a new workbook
        wb = Workbook()

        # Remove the default sheet
        default_sheet = wb.active
        wb.remove(default_sheet)

        # Create the comparison sheet (routes side by side)
        create_routes_comparison_sheet(wb, all_routes, all_arrival_times, days)

        # Create the customer service details sheet
        create_customer_details_sheet(wb, all_routes, all_arrival_times, days)

        # Create the customers by day sheet
        create_customers_by_day_sheet(wb, all_routes, all_arrival_times, all_unvisited_customers, days)

        # Save the workbook
        filename = "route_optimization_report.xlsx"
        wb.save(filename)
        print(f"Excel report saved to {filename}")

        return True

    except Exception as e:
        print(f"Error generating Excel report: {str(e)}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    create_excel_report()