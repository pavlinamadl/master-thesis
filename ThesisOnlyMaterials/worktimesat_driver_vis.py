import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')  # Use Agg backend (saves to file)


def driver_work_satisfaction(working_hours, H_ideal=8.5, H_max=9.5):
    """
    Calculate driver work-time satisfaction based on working hours.

    Parameters:
    - working_hours: working time in hours (h)
    - H_ideal: ideal working hours (8.5)
    - H_max: maximum acceptable working hours (9.5)

    Returns:
    - satisfaction score S_WT ∈ [0,1]
    """
    h = working_hours

    if h <= H_ideal:
        # Building satisfaction: linear increase from 50% to 100%
        return 0.5 + 0.5 * (h / H_ideal)
    elif H_ideal < h <= H_max:
        # Declining satisfaction: linear decrease from 100% to 50%
        return 1.0 - 0.5 * (h - H_ideal) / (H_max - H_ideal)
    else:  # h > H_max
        # Excessive hours: asymptotic decline toward zero
        return 0.5 / (1 + (h - H_max))


# Parameters
H_ideal = 8.5  # ideal working hours
H_max = 9.5  # maximum acceptable working hours

# Create array of working hours (from 0 to 12 hours)
working_hours = np.linspace(0, 12, 1000)

# Calculate satisfaction scores
satisfaction_scores = [driver_work_satisfaction(h, H_ideal, H_max) for h in working_hours]

# Create the plot
plt.figure(figsize=(10, 7))
plt.plot(working_hours, satisfaction_scores, 'b-', linewidth=3, label='Driver Work-Time Satisfaction')

# Add vertical lines for key thresholds
plt.axvline(H_ideal, color='green', linestyle='--', alpha=0.8, linewidth=2,
            label=f'Ideal Hours ({H_ideal}h)')
plt.axvline(H_max, color='orange', linestyle='--', alpha=0.8, linewidth=2,
            label=f'Max Hours ({H_max}h)')

# Shade different zones
plt.axvspan(0, H_ideal, alpha=0.1, color='green', label='Building Satisfaction Zone')
plt.axvspan(H_ideal, H_max, alpha=0.1, color='yellow', label='Declining Satisfaction Zone')
plt.axvspan(H_max, 12, alpha=0.1, color='red', label='Excessive Hours Zone')

# Formatting
plt.xlabel('Working Hours per Day', fontsize=12, fontweight='bold')
plt.ylabel('Driver Satisfaction Score', fontsize=12, fontweight='bold')
plt.title('Driver Work-Time Satisfaction Function', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)

# Set axis limits
plt.xlim(0, 12)
plt.ylim(0, 1.1)

# Add text annotations for the three zones
plt.text(4, 0.75, 'Building\nSatisfaction\n(Linear increase)',
         ha='center', va='center', fontsize=10, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))

plt.text(9, 0.75, 'Declining\nSatisfaction\n(Linear decrease)',
         ha='center', va='center', fontsize=10, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.7))

plt.text(11, 0.25, 'Excessive\nHours\n(Asymptotic decline)',
         ha='center', va='center', fontsize=10, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7))

plt.tight_layout()

# Save the plot
plt.savefig('driver_work_satisfaction.png', dpi=300, bbox_inches='tight')
print("Plot saved as 'driver_work_satisfaction.png'")

# Print some example values
print("\nDriver Work-Time Satisfaction Examples:")
print("=" * 45)
print(f"Ideal working hours: {H_ideal}h")
print(f"Maximum acceptable hours: {H_max}h")
print()

# Test specific working hours
test_hours = [4, 6, 8, 8.5, 9, 9.5, 10, 11, 12]

for hours in test_hours:
    satisfaction = driver_work_satisfaction(hours, H_ideal, H_max)
    print(f"Working {hours:4.1f} hours: Satisfaction = {satisfaction:.3f}")