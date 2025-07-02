"""
Customer Data Visualization Script
Add this file to your existing project to visualize customer locations and create data table.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Import your existing customer data
from customer_data import all_customers
from constants import AREA_SIZE, DEPOT_X, DEPOT_Y


def format_time(minutes):
    """Format minutes to HH:MM"""
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    return f"{hours:02d}:{mins:02d}"


def create_customer_location_plot(customers_list, save_plot=True, show_plot=True):
    """
    Create a matplotlib plot showing customer locations and depot using your existing data

    Args:
        customers_list: Your existing all_customers list
        save_plot: Whether to save the plot to file
        show_plot: Whether to display the plot
    """

    # Create figure
    plt.figure(figsize=(12, 10))

    # Separate depot and customers
    depot = customers_list[0]
    customers = customers_list[1:]

    # Plot customers with bigger circles
    customer_x = [c.x for c in customers]
    customer_y = [c.y for c in customers]

    plt.scatter(customer_x, customer_y, c='blue', s=100, alpha=0.7)

    # Plot depot as red circle (same shape as customers)
    plt.scatter(depot.x, depot.y, c='red', s=150, alpha=0.8,
                edgecolors='black', linewidth=2)

    # Add customer ID labels
    for customer in customers:
        plt.annotate(str(customer.id), (customer.x, customer.y),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, alpha=0.8)


    # Set plot properties
    plt.xlim(0, AREA_SIZE)
    plt.ylim(0, AREA_SIZE)
    plt.xlabel('X Coordinate (meters)', fontsize=12)
    plt.ylabel('Y Coordinate (meters)', fontsize=12)
    plt.title('Customer Locations and Depot - 10km × 10km Service Area', fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_plot:
        plt.savefig('customer_locations_plot.png', dpi=300, bbox_inches='tight')
        print("Plot saved as 'customer_locations_plot.png'")

    if show_plot:
        plt.show()
    else:
        plt.close()


def create_customer_dataframe(customers_list):
    """
    Create a pandas DataFrame with all customer information from your existing data

    Args:
        customers_list: Your existing all_customers list

    Returns:
        pandas.DataFrame: Complete customer data table
    """

    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    data = []

    for customer in customers_list:
        if customer.id == 0:  # Depot
            row = {
                'Customer_ID': customer.id,
                'Type': 'Depot',
                'X_Coordinate': customer.x,
                'Y_Coordinate': customer.y,
                'Service_Time_min': customer.service_time,
                'Must_Serve': customer.must_serve,
                'Monday_TW': 'N/A',
                'Tuesday_TW': 'N/A',
                'Wednesday_TW': 'N/A',
                'Thursday_TW': 'N/A',
                'Friday_TW': 'N/A'
            }
        else:  # Regular customer
            row = {
                'Customer_ID': customer.id,
                'Type': 'Customer',
                'X_Coordinate': round(customer.x, 1),
                'Y_Coordinate': round(customer.y, 1),
                'Service_Time_min': round(customer.service_time, 1),
                'Must_Serve': customer.must_serve,
            }

            # Add time windows for each day
            for i, day in enumerate(days):
                tw = customer.time_windows[i]
                tw_str = f"{format_time(tw.start)}-{format_time(tw.end)}"
                row[f'{day}_TW'] = tw_str

        data.append(row)

    # Create DataFrame
    df = pd.DataFrame(data)
    return df


def display_customer_summary():
    """Display comprehensive customer data summary with plot and DataFrame"""

    customers = all_customers[1:]  # Exclude depot

    print("="*80)
    print("CUSTOMER DATA VISUALIZATION AND TABLE")
    print("="*80)

    # Basic info
    print(f"Total customers: {len(customers)}")
    print(f"Depot location: ({DEPOT_X}, {DEPOT_Y})")
    print(f"Service area: {AREA_SIZE/1000}km × {AREA_SIZE/1000}km")
    print(f"All customers must-serve: {all(c.must_serve for c in customers)}")

    # Service time stats
    service_times = [c.service_time for c in customers]
    print(f"\nService Time Statistics:")
    print(f"Range: {min(service_times):.1f} - {max(service_times):.1f} minutes")
    print(f"Mean: {np.mean(service_times):.1f} ± {np.std(service_times):.1f} minutes")

    # Create plot
    print(f"\nCreating customer location plot...")
    create_customer_location_plot(all_customers)

    # Create DataFrame
    print(f"\nCreating customer data table...")
    df = create_customer_dataframe(all_customers)

    # Display DataFrame with proper formatting
    print(f"\nCustomer Data Table:")
    print("="*120)

    # Set pandas display options
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_rows', None)

    print(df.to_string(index=False))

    print(f"\nDataFrame Info:")
    print(f"Shape: {df.shape} (rows × columns)")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024:.1f} KB")

    print("="*80)

    return df


def main():
    """Main function to generate plot and DataFrame from your existing customer data"""

    print("Loading customer data from your existing system...")
    print(f"Found {len(all_customers)} total customers (including depot)")

    # Generate comprehensive display
    customer_df = display_customer_summary()

    print("\nVisualization complete!")
    print("\nAccess the data using:")
    print("- all_customers: Your original customer list")
    print("- customer_df: Pandas DataFrame with formatted data")
    print("- Plot saved as 'customer_locations_plot.png'")

    return customer_df


if __name__ == "__main__":
    # Run the visualization
    df = main()

    # Optional: Save DataFrame to CSV
    save_csv = input("\nSave DataFrame to CSV? (y/n): ").lower().strip() == 'y'
    if save_csv:
        df.to_csv('customer_data_table.csv', index=False)
        print("DataFrame saved as 'customer_data_table.csv'")