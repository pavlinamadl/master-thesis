from additional_customers import Customer

def distance(customer1: Customer, customer2: Customer) -> float: #manhattan distance
    return abs(customer1.x - customer2.x) + abs(customer1.y - customer2.y)
