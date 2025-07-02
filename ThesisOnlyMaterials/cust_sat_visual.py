import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')  # Use Agg backend (saves to file)


def customer_satisfaction(arrival_time, time_window_start, time_window_end, beta=45, tau=15):
    """
    Calculate customer satisfaction score based on arrival time and time window.

    Parameters:
    - arrival_time: actual arrival time (minutes from midnight)
    - time_window_start: start of customer's time window (a_i)
    - time_window_end: end of customer's time window (b_i)
    - beta: buffer tolerance (45 minutes)
    - tau: steep threshold (15 minutes)

    Returns:
    - satisfaction score S_C ∈ [0,1]
    """

    # Calculate time deviation δ
    if arrival_time < time_window_start:
        # Early arrival
        delta = time_window_start - arrival_time
    elif arrival_time > time_window_end:
        # Late arrival
        delta = arrival_time - time_window_end
    else:
        # On-time arrival
        delta = 0

    # Calculate satisfaction score based on piecewise function
    if delta == 0:
        # Perfect satisfaction
        return 1.0
    elif 0 < delta <= beta:
        # Gentle decline
        return 1.0 - delta / (2 * beta)
    else:  # delta > beta
        # Steep decline
        return 0.25 / (1 + 0.1 * (delta - tau))


# Set up parameters
beta = 45  # buffer tolerance (minutes)
tau = 15  # steep threshold (minutes)

# Define a customer time window (10:00 AM to 10:30 AM)
time_window_start = 10 * 60  # 10:00 AM = 600 minutes from midnight
time_window_end = 10.5 * 60  # 10:30 AM = 630 minutes from midnight

# Create array of arrival times (from 1.5 hours before to 1.5 hours after the time window)
arrival_times = np.linspace(time_window_start - 90, time_window_end + 90, 1000)

# Calculate satisfaction scores
satisfaction_scores = [
    customer_satisfaction(arrival, time_window_start, time_window_end, beta, tau)
    for arrival in arrival_times
]

# Convert minutes to hours for better readability
arrival_hours = arrival_times / 60
window_start_hours = time_window_start / 60
window_end_hours = time_window_end / 60

# Create the plot
plt.figure(figsize=(12, 8))
plt.plot(arrival_hours, satisfaction_scores, 'b-', linewidth=2.5, label='Customer Satisfaction')

# Highlight the time window
plt.axvspan(window_start_hours, window_end_hours, alpha=0.2, color='green',
            label=f'Time Window [{window_start_hours:.1f}h - {window_end_hours:.1f}h]')

# Add vertical lines to show different zones
plt.axvline(window_start_hours - beta / 60, color='orange', linestyle='--', alpha=0.7,
            label=f'Buffer Start (-{beta} min)')
plt.axvline(window_end_hours + beta / 60, color='orange', linestyle='--', alpha=0.7,
            label=f'Buffer End (+{beta} min)')

# Formatting
plt.xlabel('Arrival Time (hours from midnight)', fontsize=12)
plt.ylabel('Customer Satisfaction Score', fontsize=12)
plt.title('Customer Satisfaction Function\n30-Minute Time Window (10:00 AM - 10:30 AM)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

# Set y-axis limits
plt.ylim(0, 1.1)

# Add annotations for different zones
plt.annotate('Perfect Satisfaction\n(δ = 0)',
             xy=(window_start_hours + 0.15, 1.0),
             xytext=(window_start_hours + 0.15, 1.05),
             ha='center', fontsize=10, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='green', lw=1.5))

plt.annotate('Tolerance Zone\n(gentle decline)',
             xy=(window_end_hours + 0.5, 0.75),
             xytext=(window_end_hours + 1, 0.85),
             ha='center', fontsize=10,
             arrowprops=dict(arrowstyle='->', color='orange', lw=1))

plt.annotate('Dissatisfaction Zone\n(steep decline)',
             xy=(window_end_hours + 1.2, 0.2),
             xytext=(window_end_hours + 1.5, 0.35),
             ha='center', fontsize=10,
             arrowprops=dict(arrowstyle='->', color='red', lw=1))

plt.tight_layout()

# Save the plot instead of showing it (due to PyCharm backend issues)
plt.savefig('customer_satisfaction_function.png', dpi=300, bbox_inches='tight')
print("Plot saved as 'customer_satisfaction_function.png'")
print("You can view the image in your project directory.")

# Print some example values
print("Customer Satisfaction Function Examples:")
print("=" * 50)
print(f"Time Window: {window_start_hours:.1f}h - {window_end_hours:.1f}h")
print(f"Buffer tolerance (β): {beta} minutes")
print(f"Steep threshold (τ): {tau} minutes")
print()

# Test specific arrival times
test_times = [
    (window_start_hours - 1, "1 hour early"),
    (window_start_hours - 0.5, "30 min early"),
    (window_start_hours + 0.15, "within window (10:09 AM)"),
    (window_end_hours + 0.5, "30 min late"),
    (window_end_hours + 1, "1 hour late"),
    (window_end_hours + 2, "2 hours late")
]

for arrival_hour, description in test_times:
    arrival_min = arrival_hour * 60
    satisfaction = customer_satisfaction(arrival_min, time_window_start, time_window_end, beta, tau)
    print(f"Arrival at {arrival_hour:4.1f}h ({description:12s}): Satisfaction = {satisfaction:.3f}")