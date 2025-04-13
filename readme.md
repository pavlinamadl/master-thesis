# Customer and Driver Satisfaction Consistent Traveling Salesman Proble

## Overview
This system implements a sophisticated traveling salesman problem solution that simultaneously optimizes for dual objectives: customer satisfaction and driver satisfaction. The algorithm addresses the Consistent Traveling Salesman Problem with additional constraints for driver work consistency across multiple days. 

The code generates a weekly schedule (Monday through Friday) for drivers to visit customers, ensuring time windows are respected, service times are appropriate, and routes maintain consistency throughout the week. This solution is particularly useful for recurring delivery or service scenarios where both customer expectations and driver working conditions are critical factors.

## Problem Definition
The system solves a complex optimization problem with the following components:

### Entities
- **Depot**: Starting and ending point for all routes
- **Customers**: Locations that must be visited within specific time windows
- **Routes**: Sequences of customer visits for each day of the week
- **Driver**: Entity that traverses routes with specific working constraints

### Constraints
- **Time Windows**: Each customer specifies preferred visit time windows for each day
- **Service Times**: Each customer requires a specific amount of service time
- **Working Hours**: Drivers have ideal and maximum working hour limits
- **Lunch Break**: Fixed lunch break period must be respected
- **Start/End Times**: Routes must start and end at specific times
- **Travel Speed**: Constant travel speed between locations

### Objectives
1. **Maximize Customer Satisfaction**:
   - Higher when arrivals are within time windows
   - Decreases with early or late arrivals using non-linear functions
   - Different penalty rates for minor vs. major deviations

2. **Maximize Driver Satisfaction**:
   - Consistent working hours across days
   - Consistent customers/routes across days
   - Working hours close to ideal duration
   - Reusing route segments from previous days

## File Structure

### Core Files

#### `__init__.py`
Empty file that marks the directory as a Python package, allowing imports between modules.

#### `constants.py`
Contains all configuration parameters and constants used throughout the system:
- Area settings (size, depot location) - depot located in the centre
- Time settings (driver working hours, service times, lunch break)
- Satisfaction parameters
- Time window settings
- Driver satisfaction settings
- Algorithm control parameters (weights, iterations)
- Edge consistency bonus parameters
- Visualization settings

#### `customer_data.py`
Defines the data structures and generates test data:
- `TimeWindow` class: Named tuple for time window constraints
- `Customer` class: Contains location, time windows, and service requirements
- Generates 30 random customers plus depot
- Creates time windows for each day
- Ensures balanced distribution of customers across time windows
- Assigns realistic service times to each customer

#### `distance_utils.py`
Provides distance calculation functionality:
- Implements Manhattan distance between two customers - manhattan distance chosen due to right angle solution based on the real life
- Manhattan distance = |x₁ - x₂| + |y₁ - y₂|
- Vehicles can't travel diagonally

#### `time_utils.py`
Handles all time-related calculations:
- `check_lunch_break_conflict()`: Detects conflicts with lunch break
- `adjust_for_lunch_break()`: Pushes times that fall in lunch break to after lunch
- `estimate_arrival_times()`: Calculates arrival times for each customer in a route
- `calculate_customer_satisfaction()`: Sophisticated non-linear function to quantify satisfaction based on arrival time relative to time window

#### `satisfaction_metrics.py`
Contains functions for calculating driver satisfaction metrics:
- `calculate_working_time()`: Computes total working time for a route
- `calculate_work_time_satisfaction()`: Evaluates satisfaction with working duration
- `calculate_work_time_consistency()`: Measures consistency of working times across days
- `calculate_route_consistency()`: Quantifies similarity between routes using Jaccard similarity
- `calculate_driver_satisfaction()`: Combines all components into overall driver satisfaction

### Algorithm Modules

#### `route_construction.py`
Implements the insertion heuristic for initial route creation:
- `insertion_heuristic()`: Builds routes by iteratively inserting customers at best positions
- Considers both customer satisfaction and driver satisfaction
- Special handling for edge consistency with previous days' routes
- Zero-weight mode for forcing consistent routes

#### `route_optimization.py`
Contains the tabu search implementation for route improvement:
- `tabu_enhanced_two_opt()`: Performs 2-opt swaps with tabu restrictions
- Dynamic tabu tenure based on problem size
- Edge consistency bonuses for reusing segments from previous days
- Diversification strategy for escaping local optima
- `diversify_route()`: Helper function for route diversification

#### `route_enhancement.py`
Post-optimization customer insertion:
- `attempt_additional_insertions()`: Tries to add unvisited customers to optimized route
- Special handling for consistency mode (skips insertions to maintain consistency)
- Considers satisfaction impact of additional insertions

### Execution and Output

#### `main.py`
Main execution script that orchestrates the algorithm:
- Imports all necessary modules
- Processes each day (Monday-Friday) sequentially
- Applies three-phase approach for each day:
  1. Constructs initial route using insertion heuristic
  2. Improves route using tabu-enhanced 2-opt
  3. Attempts additional insertions
- Tracks metrics and prints detailed results
- Handles day-specific weight adjustments for consistency

#### `excel-exportResults.py`
Exports optimization results to Excel:
- Creates workbook with multiple sheets
- `create_routes_comparison_sheet()`: Shows all routes side by side
- `create_customer_details_sheet()`: Detailed service information for each customer
- `create_customers_by_day_sheet()`: Matrix showing which customers are visited on which days
- Color coding based on satisfaction scores
- Summary statistics for each day and the entire week

#### `route-visualization-custAndDriverSatisfaction.py`
Generates visualizations of the routes and statistics:
- `visualize_route_map()`: Geographical map of the route
- `visualize_time_windows()`: Timeline showing time windows and arrival times
- `visualize_weekly_stats()`: Comparative statistics across the week
- Color coding based on satisfaction levels
- Detailed annotations with arrival times and customer IDs

## Algorithm Architecture

The solution employs a three-phase metaheuristic approach:

### Phase 1: Initial Route Construction via Insertion Heuristic
```python
def insertion_heuristic(customers, day_index, previous_routes, previous_working_times, w_cust, w_driver, edge_consistency_bonus, driver_start_time, driver_finish_time, buffer_minutes)
```

The insertion heuristic builds an initial feasible route through an intelligent, step-by-step construction process:

#### Initialization
1. **Empty Route Creation**
   - Start with depot-only route: `route = [depot, depot]`
   - Initialize arrival times: `arrival_times = estimate_arrival_times(route, driver_start_time)`
   - Create list of unrouted customers: `unrouted = [c for c in customers if c.id != 0]`

2. **Previous Edge Extraction**
   - For days after Monday, extract all edges from previous day's route:
   - `previous_edges = {(prev_route[i].id, prev_route[i+1].id) for i in range(len(prev_route)-1)}`
   - These edges represent route segments that would increase consistency if reused

#### Main Insertion Loop
The algorithm iteratively selects the best customer to insert at the best position:

1. **Candidate Evaluation**
   - For each unrouted customer and each possible position in the route:
     - Create temporary route with customer inserted: `temp_route = route[:i] + [customer] + route[i:]`
     - Calculate new arrival times: `temp_arrival_times = estimate_arrival_times(temp_route, driver_start_time)`
     - Check feasibility: Skip if route finishes after `driver_finish_time`
     - Calculate customer satisfaction based on time window adherence
     - Calculate driver satisfaction based on working time and consistency
     - Apply edge consistency bonus if the insertion reuses edges from previous days

2. **Insertion Cost Calculation**
   - Standard weighted cost formula: 
     `insertion_cost = (1.0 - customer_satisfaction) * w_cust + (1.0 - driver_satisfaction) * w_driver - edge_bonus`
   - Special "zero-weight mode" for maximum consistency:
     - If `w_cust = 0.0` and `w_driver = 0.0`, then:
     - `insertion_cost = -1000.0` for consistent edges (extremely favorable)
     - `insertion_cost = 1000.0` for new edges (extremely unfavorable)
     - This effectively forces the algorithm to prioritize edge consistency above all else

3. **Best Insertion Selection**
   - Track best insertion across all combinations: `best_insertion_cost = float('inf')`
   - Update when better insertion found: 
     ```python
     if insertion_cost < best_insertion_cost:
         best_insertion_cost = insertion_cost
         best_customer = customer
         best_position = i
         best_temp_route = temp_route
         best_temp_arrival_times = temp_arrival_times
         best_customer_sat = customer_satisfaction
         best_driver_sat = driver_satisfaction
     ```

4. **Route Update**
   - If feasible insertion found, update route: `route = best_temp_route`
   - Remove customer from unrouted list: `unrouted.remove(best_customer)`
   - Update arrival times: `arrival_times = best_temp_arrival_times`
   - Continue to next iteration

5. **Termination**
   - Loop terminates when either:
     - No unrouted customers remain
     - No feasible insertions possible (remaining customers would violate constraints)

#### Final Evaluation
After construction is complete, the algorithm calculates final metrics:

1. **Recalculate Customer Satisfaction**
   - For each customer in the final route, calculate satisfaction based on arrival times
   - Average across all customers: `customer_satisfaction = total_customer_sat / num_customers`

2. **Recalculate Driver Satisfaction**
   - Calculate final driver satisfaction: `driver_satisfaction, _, _ = calculate_driver_satisfaction(...)`

3. **Calculate Total Cost**
   - Weighted objective function: `total_cost = (1.0 - customer_satisfaction) * w_cust + (1.0 - driver_satisfaction) * w_driver`

4. **Return Values**
   - Returns tuple containing: (route, total_cost, arrival_times, customer_satisfaction, driver_satisfaction)

This construction heuristic effectively balances multiple competing objectives: maximizing customers served, maintaining time window adherence, ensuring driver satisfaction, and preserving route consistency across days. The edge consistency bonus mechanism is particularly important for multi-day planning scenarios.

### Phase 2: Route Improvement via Tabu Search with 2-Opt Moves
```python
def tabu_enhanced_two_opt(route, day_index, previous_routes, previous_working_times, w_cust, w_driver, edge_consistency_bonus, driver_start_time, buffer_minutes, max_iterations, diversification_threshold)
```

The tabu-enhanced 2-opt algorithm is a sophisticated metaheuristic that systematically improves routes through intelligent local search. This implementation includes advanced features such as dynamic tabu tenure, aspiration criteria, and strategic diversification.

#### Algorithm Initialization

1. **Base Case Handling**
   - If route has 3 or fewer nodes (depot-customer-depot), return without changes
   - This optimization avoids unnecessary computation for near-trivial routes

2. **Initial Solution Setup**
   - Create working copies: `best_route = route.copy()` and `current_route = route.copy()`
   - Calculate initial arrival times: `best_arrival_times = estimate_arrival_times(best_route, driver_start_time)`
   - Determine the number of customers (excluding depot): `num_customers = len(best_route) - 2`
   - Calculate dynamic tabu tenure based on problem size: `tabu_tenure = int(math.sqrt(num_customers)) + 1`
   ```python
   print(f"Dynamic tabu tenure: {tabu_tenure} (based on {num_customers} customers)")
   ```

3. **Initial Cost Calculation**
   - Calculate customer satisfaction for each served customer based on arrival times
   - Calculate driver satisfaction considering work time and consistency
   - Compute weighted objective function: `best_cost = (1.0 - best_customer_satisfaction) * w_cust + (1.0 - best_driver_satisfaction) * w_driver`
   - Set current cost equal to best: `current_cost = best_cost`

4. **Tabu Search Structures**
   - Initialize empty tabu list as dictionary: `tabu_list = {}`
   - Initialize counters: `iteration = 0` and `iterations_without_improvement = 0`

#### Main Tabu Search Loop

The search iterates until either the maximum iterations are reached or no further improvements are possible:

1. **Diversification Check**
   - If stuck for too long (`iterations_without_improvement >= diversification_threshold`):
     - Apply diversification: `diversified_route = diversify_route(current_route)`
     - Check if diversified route is feasible by calculating arrival times
     - If feasible, reset counter: `iterations_without_improvement = 0`
     - If route structure changes, recalculate tabu tenure based on new size

2. **Neighborhood Exploration (2-Opt Moves)**
   - For each pair of non-adjacent positions i, j in the route (excluding depot):
     - Create candidate solution with 2-opt move: `neighbor_route = current_route.copy()`
     - Reverse segment between i and j: `neighbor_route[i:j+1] = reversed(current_route[i:j+1])`
     - Calculate new arrival times: `neighbor_arrival_times = estimate_arrival_times(neighbor_route, driver_start_time)`
     - Skip if route finishes after working hours: `if neighbor_arrival_times[-1] > DRIVER_FINISH_TIME: continue`

3. **Candidate Evaluation**
   - Calculate customer satisfaction for each customer in the candidate route
   - Calculate driver satisfaction considering route and work time consistency
   - Special handling for edge consistency with previous days:
     ```python
     if day_index > 0:
         # Extract edges from previous day's route
         previous_edges = set()
         if previous_routes:
             prev_route = previous_routes[-1]
             for idx in range(len(prev_route) - 1):
                 previous_edges.add((prev_route[idx].id, prev_route[idx + 1].id))
         
         # Check how many edges are consistent with previous day's route
         consistent_edges = 0
         for idx in range(len(neighbor_route) - 1):
             if (neighbor_route[idx].id, neighbor_route[idx + 1].id) in previous_edges:
                 consistent_edges += 1
         
         # Add edge consistency bonus/penalty
         edge_consistency_penalty = -consistent_edges * edge_consistency_bonus
         
         # Calculate neighbor cost with consistency consideration
         if w_cust == 0.0 and w_driver == 0.0:
             neighbor_cost = -consistent_edges  # More consistent = better
         else:
             neighbor_cost = (1.0 - neighbor_customer_satisfaction) * w_cust + \
                            (1.0 - neighbor_driver_satisfaction) * w_driver + \
                            edge_consistency_penalty
     else:
         # For Monday, use regular calculation
         neighbor_cost = (1.0 - neighbor_customer_satisfaction) * w_cust + \
                         (1.0 - neighbor_driver_satisfaction) * w_driver
     ```

4. **Tabu Status and Aspiration Criterion**
   - Define the move: `move = (i, j)`
   - Check if move is allowed: `(move not in tabu_list or neighbor_cost < best_cost) and neighbor_cost < best_neighbor_cost`
   - The aspiration criterion (`neighbor_cost < best_cost`) allows tabu moves if they lead to the best solution overall
   - If allowed and better than current best neighbor, update best neighbor variables

5. **Termination Check**
   - If no feasible moves found: `if best_neighbor_route is None: break`

6. **Solution Update**
   - Update current solution: `current_route = best_neighbor_route` and `current_cost = best_neighbor_cost`
   - If better than global best, update global best:
     ```python
     if best_neighbor_cost < best_cost:
         best_route = best_neighbor_route.copy()
         best_cost = best_neighbor_cost
         best_arrival_times = best_neighbor_arrival_times.copy()
         best_customer_satisfaction = best_neighbor_customer_satisfaction
         best_driver_satisfaction = best_neighbor_driver_satisfaction
         iterations_without_improvement = 0
     else:
         iterations_without_improvement += 1
     ```

7. **Tabu List Management**
   - Decrement tabu tenure for existing moves: `tabu_list[move] -= 1`
   - Remove expired tabu moves: `if tabu_list[move] <= 0: moves_to_remove.append(move)`
   - Add current move to tabu list: `tabu_list[best_move] = tabu_tenure`
   - Also add inverse move to prevent cycling: `tabu_list[(best_move[1], best_move[0])] = tabu_tenure`

#### Special Features

1. **Dynamic Tabu Tenure**
   - The tabu tenure (how long moves remain forbidden) scales with problem size
   - Formula: `tabu_tenure = int(math.sqrt(num_customers)) + 1`
   - Rationale: Larger problems require longer tabu restrictions to prevent cycling
   - This adaptive approach outperforms fixed tabu tenures across different problem sizes

2. **Strategic Diversification**
   - The `diversify_route()` function performs multiple random 2-opt moves
   - Triggered after `diversification_threshold` iterations without improvement
   - Number of random moves scales with route size: `num_moves = min(5, len(route) // 4)`
   - This helps escape deep local optima when the search stagnates

3. **Symmetric Tabu Management**
   - Both a move (i,j) and its inverse (j,i) are added to the tabu list
   - This prevents the algorithm from immediately reversing a move
   - Crucial for preventing cycles in the search process

4. **Edge Consistency Consideration**
   - For days after Monday, the algorithm counts edges consistent with previous day
   - Applies a bonus proportional to the number of consistent edges
   - In zero-weight mode, optimization focuses exclusively on maximizing consistent edges

The tabu-enhanced 2-opt algorithm significantly improves upon the initial route by strategically exploring the solution space while avoiding previously visited solutions. The dynamic tabu tenure and diversification mechanisms ensure effective exploration regardless of problem size.

### Phase 3: Post-Optimization Customer Insertion
```python
def attempt_additional_insertions(route, unvisited_customers, day_index, previous_routes, previous_working_times, w_cust, w_driver, edge_consistency_bonus, driver_start_time, driver_finish_time, buffer_minutes)
```

The post-optimization insertion phase attempts to maximize service capacity by intelligently inserting additional customers into the already optimized route. This phase is crucial for ensuring that all available capacity is utilized efficiently.

#### Consistency Preservation Mode
```python
# Skip additional insertions on days after Monday when using zero weights
# to maintain consistent routes
if day_index > 0 and w_cust == 0.0 and w_driver == 0.0:
    print("Skipping additional insertions to maintain route consistency")
```

## Core Functions and Components

### Time and Distance Calculations
```python
def distance(customer1: Customer, customer2: Customer) -> float
def check_lunch_break_conflict(arrival_time: float) -> bool
def adjust_for_lunch_break(time_value: float) -> float
def estimate_arrival_times(route: List[Customer], driver_start_time: float = DRIVER_START_TIME) -> List[float]
```

These functions handle the fundamental calculations needed for route evaluation:
- Manhattan distance between locations
- Detection and resolution of lunch break conflicts
- Accurate arrival time estimation accounting for travel time, service time, and lunch breaks

### Satisfaction Metrics

#### Customer Satisfaction
```python
def calculate_customer_satisfaction(
        arrival_time: float,
        time_window: TimeWindow,
        buffer_minutes: float = BUFFER_MINUTES
) -> float
```

The customer satisfaction function is a sophisticated multi-stage non-linear function that precisely quantifies how satisfied a customer would be with a particular arrival time:

1. **Perfect Satisfaction (1.0)**
   - When arrival time is exactly within the time window bounds
   - Mathematically: if `time_window.start ≤ arrival_time ≤ time_window.end`

2. **Minor Deviation (Linear Decrease)**
   - When arrival time is outside the window but within the buffer zone
   - Deviation calculated as: `deviation = |arrival_time - nearest_window_bound|`
   - Satisfaction decreases linearly from 1.0 to 0.5 as deviation increases from 0 to `buffer_minutes`
   - Formula: `satisfaction = 1.0 - (deviation / buffer_minutes) * 0.5`
   - At maximum buffer deviation, satisfaction is 0.5

3. **Medium Deviation (Steeper Linear Decrease)**
   - When deviation is beyond buffer but below the steep threshold
   - Satisfaction decreases from 0.5 to 0.25 as deviation increases from `buffer_minutes` to `SATISFACTION_STEEP_THRESHOLD`
   - Formula: 
     ```
     buffer_satisfaction = 0.5
     threshold_satisfaction = 0.25
     deviation_beyond_buffer = deviation - buffer_minutes
     max_deviation_beyond_buffer = SATISFACTION_STEEP_THRESHOLD - buffer_minutes
     satisfaction = buffer_satisfaction - (deviation_beyond_buffer / max_deviation_beyond_buffer) * (buffer_satisfaction - threshold_satisfaction)
     ```

4. **Severe Deviation (Asymptotic Decrease)**
   - When deviation exceeds the steep threshold
   - Satisfaction decreases asymptotically from 0.25 toward 0 (never quite reaching 0)
   - Formula: `satisfaction = base_satisfaction / (1.0 + 0.1 * extra_deviation)`
   - Where `base_satisfaction = 0.25` and `extra_deviation = deviation - SATISFACTION_STEEP_THRESHOLD`

This approach ensures:
- Perfect satisfaction within the window
- Reasonable tolerance for minor deviations
- Increasingly severe penalties for larger deviations
- A mathematical floor that prevents satisfaction from ever reaching absolute zero

#### Driver Satisfaction Components

Driver satisfaction is a composite metric built from three components:

##### 1. Work Time Satisfaction
```python
def calculate_work_time_satisfaction(working_time: float) -> float
```

This function evaluates how optimal the total working duration is:

1. **Conversion to Hours**
   - Working time is converted from minutes to hours for better readability:
   - `working_hours = working_time / 60.0`

2. **Below Ideal Hours (Linear Increase)**
   - When working hours are below ideal (underwork):
   - Satisfaction increases linearly from 0.5 to 1.0 as working hours increase from 0 to ideal
   - Formula: `satisfaction = 0.5 + (working_hours / IDEAL_WORKING_HOURS) * 0.5`
   - Rationale: Too little work is suboptimal but not as bad as excessive overwork

3. **Between Ideal and Maximum Hours (Linear Decrease)**
   - When working hours are between ideal and maximum (slight overwork):
   - Satisfaction decreases linearly from 1.0 to 0.5 as working hours increase from ideal to maximum
   - Formula: 
     ```
     overwork_ratio = (working_hours - IDEAL_WORKING_HOURS) / (MAX_WORKING_HOURS - IDEAL_WORKING_HOURS)
     satisfaction = 1.0 - overwork_ratio * 0.5
     ```

4. **Beyond Maximum Hours (Asymptotic Decrease)**
   - When working hours exceed maximum (severe overwork):
   - Satisfaction decreases asymptotically from 0.5 toward 0 as working hours increase
   - Formula: `satisfaction = 0.5 / (1.0 + excess_hours)`
   - Where `excess_hours = working_hours - MAX_WORKING_HOURS`

This approach creates a bell curve of satisfaction that peaks at the ideal working hours.

##### 2. Work Time Consistency
```python
def calculate_work_time_consistency(
        working_time: float,
        previous_working_times: List[float]
) -> float
```

This function measures how consistent today's working time is with previous days:

1. **Baseline Case**
   - If no previous days exist (i.e., Monday), consistency is perfect (1.0)

2. **Average Calculation**
   - Calculate average working time from all previous days:
   - `avg_previous_time = sum(previous_working_times) / len(previous_working_times)`

3. **Deviation Percentage**
   - Calculate deviation as a percentage of the average:
   - `deviation_percentage = abs(working_time - avg_previous_time) / avg_previous_time`

4. **Satisfaction Calculation**
   - For small deviations (≤ 20%):
     - Satisfaction decreases linearly from 1.0 to 0.5
     - Formula: `satisfaction = 1.0 - deviation_percentage * 2.5`
   - For larger deviations (> 20%):
     - Satisfaction continues to decrease but at a slower rate
     - Formula: `satisfaction = max(0.5 - (deviation_percentage - 0.2), 0.0)`

This approach prioritizes day-to-day consistency in working hours, with rapidly decreasing satisfaction for even small variations, reflecting the importance of predictable schedules for drivers.

##### 3. Route Consistency
```python
def calculate_route_consistency(
        route: List[Customer],
        previous_routes: List[List[Customer]]
) -> float
```

This function quantifies how similar today's route is to previous days' routes:

1. **Baseline Cases**
   - If no previous days exist (i.e., Monday), consistency is perfect (1.0)
   - If current route is empty, consistency is perfect (1.0)

2. **Customer ID Extraction**
   - Extract customer IDs from current route (excluding depot):
   - `current_ids = {customer.id for customer in route if customer.id != 0}`

3. **Jaccard Similarity Calculation**
   - For each previous route:
     - Extract customer IDs (excluding depot)
     - Calculate intersection size: `intersection = len(current_ids.intersection(prev_ids))`
     - Calculate union size: `union = len(current_ids.union(prev_ids))`
     - Compute Jaccard index: `consistency = intersection / union`
     - Handle edge case: if both routes are empty, consistency is 1.0

4. **Aggregation**
   - Final consistency is the average across all previous routes:
   - `final_consistency = sum(consistencies) / len(consistencies)`

The Jaccard similarity measure (intersection over union) provides an elegant mathematical foundation for comparing route similarity, focusing on which customers are shared between routes rather than just the total number of customers.

#### Overall Driver Satisfaction

```python
def calculate_driver_satisfaction(
        route: List[Customer],
        day_index: int,
        previous_routes: List[List[Customer]],
        previous_working_times: List[float],
        work_time_weight: float = WORK_TIME_WEIGHT,
        route_consistency_weight: float = ROUTE_CONSISTENCY_WEIGHT
) -> Tuple[float, float, float]
```

The final driver satisfaction combines all components:

1. **Working Time Calculation**
   - Calculate total working time for this route:
   - `working_time = calculate_working_time(route)`

2. **Component Calculation**
   - Work time satisfaction: `work_time_sat = calculate_work_time_satisfaction(working_time)`
   - Work time consistency: `work_time_consistency = calculate_work_time_consistency(working_time, previous_working_times)`
   - Route consistency: `route_consistency = calculate_route_consistency(route, previous_routes)`

3. **Weighted Consistency**
   - Combine consistency metrics with weights:
   - `consistency_sat = work_time_consistency * work_time_weight + route_consistency * route_consistency_weight`
   - Where weights sum to 1.0, allowing relative importance adjustment

4. **Final Satisfaction**
   - Average of time satisfaction and consistency satisfaction:
   - `overall_sat = (work_time_sat + consistency_sat) / 2.0`

5. **Return Value**
   - Returns tuple of (overall_satisfaction, work_time_satisfaction, consistency_satisfaction)
   - Allows tracking individual components for analysis

This approach gives equal importance to work quality (appropriate duration) and work consistency (predictability), resulting in a comprehensive driver satisfaction metric between 0 and 1.

### Utility Functions
```python
def calculate_working_time(route: List[Customer]) -> float
def diversify_route(route: List[Customer]) -> List[Customer]
```

These functions provide essential utilities:
- Accurate calculation of total working time including travel, service, and breaks
- Route diversification through random 2-opt moves to escape local optima

## Configuration Parameters

The algorithm's behavior is controlled through numerous parameters defined in `constants.py`:

### Time-Related Constants
- `DRIVER_START_TIME`: Minutes since midnight when driver begins work (e.g., 480 for 8:00 AM)
- `DRIVER_FINISH_TIME`: Minutes since midnight when driver must finish (e.g., 1020 for 5:00 PM)
- `BUFFER_MINUTES`: Allowed buffer zone around time windows before satisfaction drops
- `LUNCH_BREAK_START`: Minutes since midnight when lunch begins
- `LUNCH_BREAK_END`: Minutes since midnight when lunch ends
- `SATISFACTION_STEEP_THRESHOLD`: Deviation threshold where satisfaction penalty becomes steeper

### Performance Parameters
- `SPEED_METERS_PER_MINUTE`: Travel speed in meters per minute
- `W_CUSTOMER`: Weight for customer satisfaction (higher values prioritize customers)
- `W_DRIVER`: Weight for driver satisfaction (higher values prioritize drivers)
- `IDEAL_WORKING_HOURS`: Target duration of driver's working day in hours
- `MAX_WORKING_HOURS`: Maximum acceptable working day duration in hours

### Algorithm Control Parameters
- `WORK_TIME_WEIGHT`: Importance of work time component in driver satisfaction
- `ROUTE_CONSISTENCY_WEIGHT`: Importance of route consistency in driver satisfaction
- `MAX_TABU_ITERATIONS`: Maximum number of iterations for tabu search
- `TABU_DIVERSIFICATION_THRESHOLD`: Iterations without improvement before diversification
- `EDGE_CONSISTENCY_BONUS`: Bonus factor for maintaining consistent route edges

## Data Structures

### Customer Data
```python
class TimeWindow(NamedTuple):
    start: float  # Minutes since midnight
    end: float    # Minutes since midnight

class Customer:
    id: int       # Unique identifier
    x: float      # X coordinate
    y: float      # Y coordinate
    service_time: float  # Minutes required for service
    time_windows: List[TimeWindow]  # Time windows for each day
```

The system uses:
- **TimeWindow**: Named tuple representing preferred visit times
- **Customer**: Class containing location, service requirements, and time preferences
- **Depot**: Special customer with ID 0 representing start/end point

## Execution Flow

The `main()` function orchestrates the full execution:

1. **Initialization**: Loads customer data and prepares data structures
2. **Weekly Planning**: Iterates through five days (Monday-Friday)
   - Applies different weighting strategies based on day of week
   - Tracks routes and working times for consistency calculations
3. **Daily Route Construction**: For each day:
   - Creates initial route using insertion heuristic
   - Identifies unvisited customers
   - Improves route using tabu-enhanced 2-opt
   - Attempts additional insertions post-optimization
   - Calculates and stores performance metrics
4. **Results Reporting**: Generates detailed output including:
   - Route details with customer IDs
   - Working times and satisfaction metrics
   - Start and end times for each day
   - Individual customer service details with time windows, arrival times, deviations
   - Overall weekly performance statistics

## Advanced Features

### Route Consistency Mechanisms
The system employs multiple strategies to maintain consistency:

1. **Direct Route Similarity**:
   - Jaccard similarity calculation (intersection over union)
   - Higher weighting for customers that appear in multiple days

2. **Edge Consistency Bonus**:
   - Identifies route segments (edges) used in previous days
   - Applies bonus factor to favor reusing these edges
   - Special handling for zero-weight mode that prioritizes perfect consistency

3. **Working Time Consistency**:
   - Tracks daily working times across the week
   - Penalizes deviations from average of previous days
   - Non-linear penalty function for large deviations

### Adaptive Tabu Mechanisms
The tabu search component features several advanced techniques:

1. **Dynamic Tabu Tenure**:
   - Automatically scales with problem size (route length)
   - Based on square root of number of customers plus one
   - Prevents cycling in larger problems

2. **Strategic Diversification**:
   - Triggered after predefined number of non-improving iterations
   - Performs multiple random 2-opt moves to escape local optima
   - Recalculates tabu tenure if route size changes

3. **Symmetric Tabu Treatment**:
   - Both a move and its inverse are added to the tabu list
   - Prevents immediate reversal of moves

### Satisfaction Functions
The system uses sophisticated non-linear satisfaction functions:

1. **Customer Satisfaction**:
   - Multi-stage function with different slopes based on deviation severity
   - Perfect (1.0) within time window
   - Linear decrease within buffer zone
   - Medium decrease up to defined threshold
   - Asymptotic decrease for severe deviations

2. **Driver Satisfaction**:
   - Work time satisfaction based on ideal and maximum hours
   - Work time consistency uses percentage deviation approach
   - Route consistency uses Jaccard similarity