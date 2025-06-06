"""
This file contains all the constants used across the delivery routing project.
"""

# Area settings
AREA_SIZE = 10000  # Size of the area in meters (10km x 10km)
DEPOT_X = 5000.0   # X-coordinate of the depot (center of the area)
DEPOT_Y = 5000.0   # Y-coordinate of the depot (center of the area)

# Time settings
DRIVER_START_TIME = 480.0   # 8:00 AM in minutes from midnight
DRIVER_FINISH_TIME = 990.0  # 4:30 PM in minutes from midnight
BUFFER_MINUTES = 45.0       # Buffer time for satisfaction calculation
SERVICE_TIME_MINUTES = 5.0  # Time spent at each customer
LUNCH_BREAK_START = 720.0   # 12:00 PM in minutes from midnight
LUNCH_BREAK_END = 750.0     # 12:30 PM in minutes from midnight
SPEED_METERS_PER_MINUTE = 500.0  # Vehicle speed (500 m/min = 30 km/h)

# Satisfaction parameters
SATISFACTION_STEEP_THRESHOLD = 15.0  # Time threshold for steeper satisfaction decrease

# Time window settings
TIME_WINDOW_START = 480.0   # 8:00 AM
TIME_WINDOW_END = 990.0     # 4:30 PM
TIME_WINDOW_INTERVAL = 30   # 30-minute intervals for time windows

# Driver satisfaction settings
IDEAL_WORKING_HOURS = 8.5  # 8.5 hours (510 minutes) is the ideal working time including lunch
MAX_WORKING_HOURS = 9.5    # 9.5 hours (570 minutes) is the maximum before severe dissatisfaction
WORK_TIME_WEIGHT = 0.5     # Weight for work time consistency in driver satisfaction
ROUTE_CONSISTENCY_WEIGHT = 0.5  # Weight for route consistency in driver satisfaction

# Satisfaction weights - Modified to use alpha parameter
ALPHA = 0.0  # Weight for customer satisfaction (between 0 and 1)
# Driver weight is implicitly (1-ALPHA) and doesn't need to be stored separately

# Edge consistency bonus
EDGE_CONSISTENCY_BONUS = 0.05  # Edge consistency bonus, 0 minimal

# Extended algorithm parameters
MUST_SERVE_PRIORITY = 2.0   # Priority boost for must-serve customers in insertion cost
EXTENDED_HOURS_MINUTES = 120  # Minutes to extend working hours for must-serve customers

# Algorithm settings
MAX_ITERATIONS_2OPT = 200   # Maximum iterations for 2-opt improvement

# Tabu search parameters
MAX_TABU_ITERATIONS = 200   # Maximum iterations for tabu search
TABU_DIVERSIFICATION_THRESHOLD = 20  # Iterations without improvement before diversification
TABU_ASPIRATION_COEF = 0.95  # Coefficient for aspiration criterion (allow tabu move if cost < best_cost*this)
TABU_INITIAL_TENURE_FACTOR = 1.0  # Factor to multiply with sqrt(n) for tabu tenure

# Visualization settings
VISUALIZATION_DPI = 300     # DPI for saved visualization images
VISUALIZATION_SIZE = (20, 10)  # Size of visualization figures