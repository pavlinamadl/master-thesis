import numpy as np
import matplotlib.pyplot as plt

# Define the constants inline
BUFFER_MINUTES = 45
SATISFACTION_STEEP_THRESHOLD = 15

from collections import namedtuple
TimeWindow = namedtuple('TimeWindow', ['start', 'end'])

# Revised function
def calculate_customer_satisfaction(
        arrival_time: float,
        time_window: TimeWindow,
        buffer_minutes: float = BUFFER_MINUTES
) -> float:
    if time_window.start <= arrival_time <= time_window.end:
        return 1.0
    deviation = (time_window.start - arrival_time) if arrival_time < time_window.start else (arrival_time - time_window.end)
    if deviation <= buffer_minutes:
        return 1.0 - (deviation / buffer_minutes) * 0.5
    tau = SATISFACTION_STEEP_THRESHOLD
    if deviation <= buffer_minutes + tau:
        buffer_satisfaction    = 0.5
        threshold_satisfaction = 0.25
        dev_beyond_buffer      = deviation - buffer_minutes
        return buffer_satisfaction - (dev_beyond_buffer / tau) * (
               buffer_satisfaction - threshold_satisfaction)
    base_satisfaction = 0.25
    extra_deviation   = deviation - (buffer_minutes + SATISFACTION_STEEP_THRESHOLD)
    return base_satisfaction / (1.0 + 0.1 * extra_deviation)

# Test examples
tw = TimeWindow(start=600, end=630)
test_deviations = [0, BUFFER_MINUTES/2, BUFFER_MINUTES, BUFFER_MINUTES + SATISFACTION_STEEP_THRESHOLD/2, BUFFER_MINUTES + SATISFACTION_STEEP_THRESHOLD, BUFFER_MINUTES + SATISFACTION_STEEP_THRESHOLD + 30]
for d in test_deviations:
    arr = tw.end + d
    print(f"δ = {d:.1f} → S = {calculate_customer_satisfaction(arr, tw):.3f}")

