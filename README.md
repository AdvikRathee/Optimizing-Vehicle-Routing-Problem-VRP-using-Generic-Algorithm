# Optimizing-Vehicle-Routing-Problem-VRP-using-Generic-Algorithm
import numpy as np
import matplotlib.pyplot as plt
import random

# --- Problem Setup ---
num_customers = 10
num_vehicles = 3
vehicle_capacity = 50
customer_demands = np.random.randint(1, 20, size=num_customers)
customer_locations = np.random.rand(num_customers, 2) * 100  # Random customer locations
depot_location = np.array([50, 50])  # Depot location at the center

# --- GA Parameters ---
population_size = 50
generations = 100
mutation_rate = 0.1
elite_size = 5

# --- Helper Functions ---
def calc_distance(loc1, loc2): return np.linalg.norm(loc1 - loc2)

def route_distance(route):
    dist = calc_distance(depot_location, customer_locations[route[0]])  # Depot to first customer
    for i in range(len(route) - 1):
        dist += calc_distance(customer_locations[route[i]], customer_locations[route[i + 1]])
    dist += calc_distance(customer_locations[route[-1]], depot_location)  # Last customer to depot
    return dist

def fitness(route): return 1 / (route_distance(route) + 1e-5)  # Avoid division by zero

# --- Create Random Population ---
def create_population():
    return [np.random.permutation(num_customers).tolist() for _ in range(population_size)]

# --- Selection (Roulette Wheel) ---
def select(pop_ranked):
    selection = random.choices(pop_ranked, weights=[f[1] for f in pop_ranked], k=population_size)
    return [crossover(selection[i][0], selection[len(selection)-i-1][0]) for i in range(population_size)]

# --- Crossover (Ordered Crossover) ---
def crossover(parent1, parent2):
    start, end = sorted([random.randint(0, len(parent1)), random.randint(0, len(parent1))])
    child_p1 = parent1[start:end]
    child_p2 = [item for item in parent2 if item not in child_p1]
    return child_p1 + child_p2

# --- Mutation (Swap Mutation) ---
def mutate(ind):
    if random.random() < mutation_rate:
        idx1, idx2 = random.sample(range(len(ind)), 2)
        ind[idx1], ind[idx2] = ind[idx2], ind[idx1]
    return ind

# --- GA Execution ---
def genetic_algorithm():
    population = create_population()
    progress = []
    
    for gen in range(generations):
        ranked_pop = sorted([(ind, fitness(ind)) for ind in population], key=lambda x: x[1], reverse=True)
        progress.append(1 / ranked_pop[0][1])
        
        # Print intermediate progress
        if gen % 10 == 0 or gen == generations - 1:
            best_route = ranked_pop[0][0]
            best_distance = 1 / ranked_pop[0][1]
            print(f"Generation {gen} | Best Distance: {best_distance:.2f}")
        
        population = select(ranked_pop[:elite_size])  # Selection
        population = [mutate(ind) for ind in population]  # Mutation
    
    return ranked_pop[0][0], progress

# --- Run GA and Plot Convergence ---
best_route, progress = genetic_algorithm()

plt.plot(progress)
plt.ylabel('Distance')
plt.xlabel('Generation')
plt.title('GA Convergence')
plt.show()

# --- Output Best Route and Distance ---
print(f"\nBest Route: {best_route}")
best_route_dist = route_distance(best_route)
print(f"Total Distance: {best_route_dist:.2f}")

# --- Plot Best Route ---
def plot_route(route):
    plt.figure(figsize=(10,6))
    plt.scatter(customer_locations[:, 0], customer_locations[:, 1], c='red', label='Customers')
    plt.scatter(depot_location[0], depot_location[1], c='blue', label='Depot')
    
    # Plot route lines
    ordered_locations = [depot_location] + [customer_locations[i] for i in route] + [depot_location]
    ordered_locations = np.array(ordered_locations)
    plt.plot(ordered_locations[:, 0], ordered_locations[:, 1], 'o-', label='Best Route')
    
    plt.legend()
    plt.title("Optimal Delivery Route")
    plt.show()

plot_route(best_route)
