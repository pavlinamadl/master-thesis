# Consistent TSP: Balancing Customer and Driver Satisfaction
# Pavlina Madl - Master Thesis

This algorithm solves the Consistent Traveling Salesman Problem with Time Windows using a three-phase optimization algorithm that balances customer satisfaction and driver satisfaction across a weekly schedule.

The solution is divided into three phases:
1. **Construction Phase**: Initial route building using insertion heuristic
2. **Optimization Phase**: Route improvement via tabu-enhanced 2-opt local search  
3. **Enhancement Phase**: Additional customer insertion post-optimization

The algorithm optimizes routes for five weekdays while maintaining consistency between days to reduce driver cognitive load and improve operational efficiency by using an alpha parameter (α) where α controls customer satisfaction weight and (1-α) controls driver satisfaction weight.

Run `python main1.py` to execute the optimization. Modify `ALPHA` in `constants.py` to adjust the customer-driver satisfaction balance (0.0 = driver-focused, 1.0 = customer-focused).

## File Descriptions

**`main1.py`** - Main execution script that runs the optimization and displays detailed results and metrics.

**`optimization_engine.py`** - Coordinates the three-phase optimization process and caches results for multiple algorithm runs.

**`route_construction.py`** - Implements insertion heuristic to build initial feasible routes considering time windows and must-serve constraints.

**`route_optimization.py`** - Performs tabu-enhanced 2-opt local search to improve route quality while avoiding cycling through previously explored solutions.

**`route_enhancement.py`** - Attempts to insert additional unvisited customers into optimized routes without degrading solution quality.

**`customer_data.py`** - Generates base customer dataset with time windows and service requirements for the depot and 30 must-serve customers.

**`additional_customers.py`** - Creates extended customer set with 45 additional customers having varied service day preferences and time window constraints.

**`satisfaction_metrics.py`** - Calculates driver satisfaction based on working time, time consistency, and route consistency across multiple days.

**`time_utils.py`** - Handles time calculations including arrival time estimation, lunch break adjustments, and customer satisfaction scoring.

**`distance_utils.py`** - Provides Manhattan distance calculation between customer locations.

**`constants.py`** - Contains all system parameters including time settings, satisfaction weights, algorithm parameters, and area dimensions.

**`weekly_route_visual.py`** - Creates streamlined weekly route visualizations showing all five days side-by-side with satisfaction color coding.

**`route-visualization-custAndDriverSatisfaction.py`** - Generates detailed daily visualizations including geographic route maps and time window compliance charts.

**`excel-exportResults.py`** - Exports comprehensive optimization results to Excel with multiple sheets for route comparison and customer service details.

**`cheapest_solution.py`** - Provides baseline nearest-neighbor solution for comparison with the optimized multi-objective approach.