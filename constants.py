#area
AREA_SIZE = 10000  # size of the area in meters (10km x 10km)
DEPOT_X = 5000.0   # X-coordinate,depot
DEPOT_Y = 5000.0   # Y-coordinate,depot

#time settings
DRIVER_START_TIME = 480.0   #8:00 AM
DRIVER_FINISH_TIME = 990.0  #4:30 PM
BUFFER_MINUTES = 45.0       #buffer time for satisfaction
SERVICE_TIME_MINUTES = 5.0  #time spent at each customer
LUNCH_BREAK_START = 720.0   #12:00 PM
LUNCH_BREAK_END = 750.0     #12:30 PM
SPEED_METERS_PER_MINUTE = 500.0  #speed (500 m/min = 30 km/h)
SATISFACTION_STEEP_THRESHOLD = 15.0  #time threshold

#time window settings
TIME_WINDOW_START = 480.0   #8:00 AM
TIME_WINDOW_END = 990.0     #4:30 PM
TIME_WINDOW_INTERVAL = 30   #30-minute intervals

#driver satisfaction settings, alpha
IDEAL_WORKING_HOURS = 8.5  #8.5 hours: ideal working time including lunch
MAX_WORKING_HOURS = 9.5    #9.5 hours: maximum before severe dissatisfaction
WORK_TIME_WEIGHT = 0.5     #weight for work time consistency
ROUTE_CONSISTENCY_WEIGHT = 0.5  #weight for route consistency
ALPHA = 0.1  #weight for customer satisfaction (between 0 and 1)

EDGE_CONSISTENCY_BONUS = 0.05  #edge consistency bonus, 0 minimal

MUST_SERVE_PRIORITY = 2.0   #priority boost for must-serve customers
EXTENDED_HOURS_MINUTES = 120  #minutes to extend working hours

#algorithm settings
MAX_ITERATIONS_2OPT = 200   #maximum iterations for 2-opt
MAX_TABU_ITERATIONS = 200   #maximum iterations for tabu search
TABU_DIVERSIFICATION_THRESHOLD = 20  #iterations without improvement before diversification
TABU_ASPIRATION_COEF = 0.95  #coefficient for aspiration criterion
TABU_INITIAL_TENURE_FACTOR = 1.0  #factor to multiply with sqrt(n)

# vsualization settings
VISUALIZATION_DPI = 300     #DPI for saved visualization images
VISUALIZATION_SIZE = (20, 10)  #size of visualization figures