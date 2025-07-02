import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib

matplotlib.use('Agg')  # Use Agg backend (saves to file)

# Create figure with subplots for different examples
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Route Consistency Examples: Jaccard Similarity Calculation', fontsize=16, fontweight='bold')


def draw_route_comparison(ax, current_route, previous_route, title):
    """Draw a visual comparison of two routes showing Jaccard similarity."""

    # Calculate sets
    intersection = current_route.intersection(previous_route)
    union = current_route.union(previous_route)
    current_only = current_route - previous_route
    previous_only = previous_route - current_route

    # Calculate Jaccard similarity
    jaccard = len(intersection) / len(union) if len(union) > 0 else 0

    # Clear the axis
    ax.clear()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.set_aspect('equal')

    # Draw circles to represent the sets (Venn diagram style)
    circle1 = patches.Circle((3, 4), 2.5, linewidth=3, edgecolor='blue', facecolor='lightblue', alpha=0.3)
    circle2 = patches.Circle((7, 4), 2.5, linewidth=3, edgecolor='red', facecolor='lightcoral', alpha=0.3)

    ax.add_patch(circle1)
    ax.add_patch(circle2)

    # Add labels for the circles
    ax.text(1.5, 6.5, 'Current Route', fontsize=12, fontweight='bold', color='blue')
    ax.text(7.5, 6.5, 'Previous Route', fontsize=12, fontweight='bold', color='red')

    # Add customer numbers
    # Current route only customers (left side)
    y_pos = 5.5
    for i, customer in enumerate(sorted(current_only)):
        ax.text(2, y_pos - i * 0.4, f'C{customer}', fontsize=10, ha='center',
                bbox=dict(boxstyle='circle', facecolor='lightblue'))

    # Previous route only customers (right side)
    y_pos = 5.5
    for i, customer in enumerate(sorted(previous_only)):
        ax.text(8, y_pos - i * 0.4, f'C{customer}', fontsize=10, ha='center',
                bbox=dict(boxstyle='circle', facecolor='lightcoral'))

    # Intersection customers (middle)
    y_pos = 5.5
    for i, customer in enumerate(sorted(intersection)):
        ax.text(5, y_pos - i * 0.4, f'C{customer}', fontsize=10, ha='center',
                bbox=dict(boxstyle='circle', facecolor='yellow', edgecolor='green', linewidth=2))

    # Add calculation text
    calc_text = f"""
Current: {{{', '.join(f'C{c}' for c in sorted(current_route))}}}
Previous: {{{', '.join(f'C{c}' for c in sorted(previous_route))}}}

Intersection: {{{', '.join(f'C{c}' for c in sorted(intersection))}}} → {len(intersection)} customers
Union: {{{', '.join(f'C{c}' for c in sorted(union))}}} → {len(union)} customers

Jaccard = |Intersection| / |Union| = {len(intersection)}/{len(union)} = {jaccard:.3f}
    """

    ax.text(5, 1.5, calc_text, fontsize=9, ha='center', va='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

    ax.set_title(f'{title}\nJaccard Similarity = {jaccard:.3f}', fontsize=12, fontweight='bold')
    ax.axis('off')

    return jaccard


# Example 1: High similarity
current1 = {1, 2, 3, 4, 5}
previous1 = {1, 2, 3, 6, 7}
j1 = draw_route_comparison(axes[0, 0], current1, previous1, "Example 1: Good Consistency")

# Example 2: Perfect similarity
current2 = {1, 2, 3, 4}
previous2 = {1, 2, 3, 4}
j2 = draw_route_comparison(axes[0, 1], current2, previous2, "Example 2: Perfect Consistency")

# Example 3: Low similarity
current3 = {1, 2, 3, 4, 5}
previous3 = {6, 7, 8, 9, 10}
j3 = draw_route_comparison(axes[1, 0], current3, previous3, "Example 3: No Consistency")

# Example 4: Moderate similarity
current4 = {1, 2, 3, 4, 5, 6}
previous4 = {3, 4, 7, 8, 9, 10}
j4 = draw_route_comparison(axes[1, 1], current4, previous4, "Example 4: Moderate Consistency")

plt.tight_layout()

# Save the plot
plt.savefig('route_consistency_examples.png', dpi=300, bbox_inches='tight')
print("Plot saved as 'route_consistency_examples.png'")

# Print summary
print("\nRoute Consistency Summary:")
print("=" * 40)
print(f"Example 1 - Good Consistency:     J = {j1:.3f}")
print(f"Example 2 - Perfect Consistency:  J = {j2:.3f}")
print(f"Example 3 - No Consistency:       J = {j3:.3f}")
print(f"Example 4 - Moderate Consistency: J = {j4:.3f}")
print()
print("Legend:")
print("• Blue circle = Current route customers")
print("• Red circle = Previous route customers")
print("• Yellow circles (green border) = Common customers (intersection)")
print("• Jaccard = (Common customers) / (All unique customers)")