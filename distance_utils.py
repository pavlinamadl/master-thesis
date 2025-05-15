import math
from customer_data import Customer

def distance(customer1: Customer, customer2: Customer) -> float:
    """Calculate Manhattan distance between two customers"""
    return abs(customer1.x - customer2.x) + abs(customer1.y - customer2.y)
