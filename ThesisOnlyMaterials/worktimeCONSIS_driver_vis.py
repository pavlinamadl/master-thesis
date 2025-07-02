import numpy as np
import matplotlib.pyplot as plt

def work_time_consistency_satisfaction(relative_deviation):
    """
    Calculate work-time consistency satisfaction based on relative deviation.

    Parameters:
    - relative_deviation: d = |T_w - T̄| / |T̄| (relative deviation from average)

    Returns:
    - satisfaction score S_WTC ∈ [0,1]
    """
    d = relative_deviation

    if d <= 0.2:
        # Linear decrease from 100% to 50% satisfaction
        return 1.0 - 2.5 * d
    else:  # d > 0.2
        # Continue decreasing linearly to zero at d = 0.7
        return max(0.5 - (d - 0.2), 0.0)

# Create array of relative deviations (from 0% to 100%)
relative_deviations = np.linspace(0, 1.0, 1000)

# Calculate satisfaction scores
satisfaction_scores = [work_time_consistency_satisfaction(d) for d in relative_deviations]

# Create the plot
plt.figure(figsize=(10, 7))
plt.plot(relative_deviations * 100, satisfaction_scores, 'b-', linewidth=3,
         label='Work-Time Consistency Satisfaction')

# Add vertical lines for key thresholds
plt.axvline(20, color='orange', linestyle='--', alpha=0.8, linewidth=2,
            label='Acceptable Variation Limit (20%)')
plt.axvline(70, color='red', linestyle='--', alpha=0.8, linewidth=2,
            label='Zero Satisfaction Threshold (70%)')

# Shade different zones (no legend entries)
plt.axvspan(0, 20, alpha=0.1, color='green')
plt.axvspan(20, 70, alpha=0.1, color='yellow')
plt.axvspan(70, 100, alpha=0.1, color='red')

# Formatting
plt.xlabel('Relative Deviation from Average Working Time (%)', fontsize=12, fontweight='bold')
plt.ylabel('Work-Time Consistency Satisfaction', fontsize=12, fontweight='bold')
plt.title('Driver Work-Time Consistency Satisfaction Function', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)

# Set axis limits
plt.xlim(0, 100)
plt.ylim(0, 1.1)

plt.tight_layout()

# Instead of plt.show(), save directly to a file:
plt.savefig('work_time_consistency_satisfaction.png', dpi=300, bbox_inches='tight')
print("Plot saved as 'work_time_consistency_satisfaction.png'")
