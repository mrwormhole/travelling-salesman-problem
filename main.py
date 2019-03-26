import itertools
import random
import operator
import pandas as pd
import numpy as np
from math import inf as oo


def print_cities_and_distances(cities, distances):
    """
    This is for printing our distances matrix
    """
    df = pd.DataFrame(data=distances, columns=cities, index=cities)
    print(df)


def create_random_cities_and_distances():
    """
    This is for generating our distances matrix as 2D array
    """
    cities_count = random.randint(40, 40)
    cities = [c for c in range(cities_count)]
    distances = [[] for _ in range(cities_count)]

    for i in range(cities_count):
        for j in range(cities_count):
            if i == j:
                distance = 0
            elif i > j:
                distance = distances[j][i]
            else:
                distance = random.randint(1, 100)
            distances[i].append(distance)
    print_cities_and_distances(cities, distances)
    return distances


def get_cities(cities_count):
    """
    This is for generating cities in 1D array
    """
    return [c for c in range(cities_count)]


def get_specific_permutations(start_point, cities):
    """
    This gets rid of opposite mirrored permutations from list and ensures starting point is the first index
    Input: 1,[1,2,3] Output: [(1,2,3),(1,3,2)]
    """
    all_permutations = list(itertools.permutations(cities))  # this is all permutations n!
    specific_permutations = list()  # this is where we have start point as first index
                                    # and mirrored routes are removed (n-1)!/2
    for perm in all_permutations:
        if perm[::-1] in all_permutations:
            all_permutations.remove(perm[::-1])

    for perm in all_permutations:
        if perm[0] == start_point:
            specific_permutations.append(perm)

    return specific_permutations


def do_exhaustive_search(starting_point, distances_matrix):
    cities = get_cities(len(distances_matrix))  # we resemble [A,B,C,D,E] as [0,1,2,3,4]
    start_point = starting_point  # 0 means A, 1 means B, 2 means C etc.
    if start_point >= len(cities) or start_point < 0:
        start_point = 0
    specific_routes = get_specific_permutations(start_point, cities)
    results = list()  # for all of the sum values for each route

    for route in specific_routes:
        sum = 0
        for i in range(len(route)-1):
            sum += distances_matrix[route[i]][route[i+1]]
        sum += distances_matrix[route[len(route)-1]][route[0]]
        results.append(sum)

    index = results.index(min(results))
    print("Your route is: " + str(specific_routes[index]))
    print("Freshly calculated shortest route takes: " + str(min(results)))
    return min(results), specific_routes[index]


def do_greedy_search(starting_point, distances_matrix):
    cities = get_cities(len(distances_matrix))
    start_point = starting_point  # 0 means A, 1 means B, 2 means C
    if start_point >= len(cities) or start_point < 0:
        start_point = 0
    current_point = start_point
    shortest_distance = oo
    visited_route = list()
    visited_route.append(current_point)
    available_path = cities  # [0,1,2,3] for 4 cities
    sum = 0

    for city in range(len(cities)):
        for i in range(len(cities)):
            if available_path[i] == current_point:
                continue
            elif shortest_distance > distances_matrix[current_point][available_path[i]]:
                picked_point = available_path[i]
            shortest_distance = min(shortest_distance, distances_matrix[current_point][available_path[i]])
        if len(available_path) <= 1:
            sum += distances_matrix[current_point][start_point]
            print("Route: " + str(visited_route))
            print("Distance: " + str(sum))
        else:
            sum += distances_matrix[current_point][picked_point]
            visited_route.append(picked_point)
            available_path.remove(current_point)
            current_point = picked_point
            shortest_distance = oo
    return sum, visited_route


def create_initial_population(starting_point, cities, population_size):
    """
    This creates random routes as an array and stores into population.
    There can be duplicates. First index is always starting point
    """
    start_point = starting_point  # 0 means A, 1 means B, 2 means C etc.
    if start_point >= len(cities) or start_point < 0:
        start_point = 0
    population = []
    for i in range(population_size):
        temp = random.sample(cities, len(cities))
        if start_point in temp:
            temp.remove(start_point)
        temp[:0] = [start_point]
        population.append(temp)
    return population


def get_fitness_score(route, distances_matrix):
    """
    This gets the fitness score of the route
    Longer the route lower the score becomes
    """
    distance = 0
    for i in range(len(route)-1):
        distance += distances_matrix[route[i]][route[i+1]]
    distance += distances_matrix[route[len(route)-1]][route[0]]
    return 1 / float(distance)


def rank_routes(population, distances_matrix):
    """
    This function creates hash map as key,value pairs and
    Returns sorted according to fitness score like [(1, 0.0792), (3, 0.0760), (2, 0.0742), (4, 0.0739)]
    """
    fitness_results = {}
    for i in range(len(population)):
        fitness_results[i] = get_fitness_score(population[i], distances_matrix)
    return sorted(fitness_results.items(), key=operator.itemgetter(1), reverse=True)


def selection(population_ranked, elite_size):
    """
    This function guarantees first elites to be included in output array according to fitness points
    Then for the remaining part it picks random percentage and compares it with the cumulative percentage
    If that percentage is higher it adds it and goes for the next iteration until it feels population back to its size
    Lower the pick more healthy population will be generated but variation will get less which is something we dont like
    """
    selection_results = []
    df = pd.DataFrame(np.array(population_ranked), columns=["Index", "Fitness"])
    df["cum_sum"] = df.Fitness.cumsum()
    df["cum_perc"] = 100 * df.cum_sum/df.Fitness.sum()
    for i in range(elite_size):
        selection_results.append(population_ranked[i][0])
    for i in range(len(population_ranked) - elite_size):
        pick = 100 * random.random()
        for i in range(len(population_ranked)):
            if pick <= df.iat[i, 3]:
                selection_results.append(population_ranked[i][0])
                break
    return selection_results


def get_mating_pool(population, selection_results):
    """
    This function gets our population in array with respect to selection results
    """
    mating_pool = []
    for i in range(len(selection_results)):
        index = selection_results[i]
        mating_pool.append(population[index])
    return mating_pool


def breed(parent1, parent2):
    """
    This functions constructs a child from parent1 and parent2.ChildPart1 comes from parent1 and
    ChildPart2 comes from parent2.Idea is we always pick [0 to random value] interval from parent1 and
    Complete missing parts from parent1 with order
    """
    child = []
    childPart1 = []
    childPart2 = []

    geneA = random.randint(0,len(parent1)-1)
    geneB = random.randint(0,len(parent1)-1)
    startingPointOfGene = 0  # set to 0 because we wanna keep our starting point in child
    endingPointOfGene = (geneA+geneB) // 2 + 1
    if endingPointOfGene == len(parent1):
        endingPointOfGene -= 1

    for i in range(startingPointOfGene, endingPointOfGene):
        childPart1.append(parent1[i])

    childPart2 = [i for i in parent2 if i not in childPart1]

    child = childPart1 + childPart2
    return child


def breed_population(mating_pool, elite_size):
    """
    Elites directly become the children since they don't die for generations.Pool is obtained by random samples
    Of mating pool.In mating pool one couple can have more than 1 kids but their kids can be different than the siblings
    In odd number of populations a person can have a kid by itself and the kid will be exactly itself.Make sure we dont
    Have odd population numbers for more variety
    """
    children = []
    length = len(mating_pool) - elite_size
    pool = random.sample(mating_pool, len(mating_pool))

    for i in range(elite_size):
        children.append(mating_pool[i])

    for i in range(length):
        child = breed(pool[i], pool[len(mating_pool)-i-1])
        children.append(child)
    return children


def mutate(individual, mutation_rate):
    """
    Mutation is actually basic swap operation which depends on its rate for each iteration
    """
    for swapped in range(1, len(individual)):
        if random.random() < mutation_rate:
            swapWith = random.randint(1, len(individual)-1)
            city1 = individual[swapped]
            city2 = individual[swapWith]
            individual[swapped] = city2
            individual[swapWith] = city1
    return individual


def mutate_population(population, mutation_rate):
    """
    Tries to apply mutation to all population
    """
    population_mutated = []

    for i in range(len(population)):
        individual = mutate(population[i], mutation_rate)
        population_mutated.append(individual)
    return population_mutated


def generate_new_generation(current_generation, elite_size, mutation_rate, distances_matrix):
    """
    Generates a new population from the 1 generation older population
    """
    population_ranked = rank_routes(current_generation, distances_matrix)
    selection_results = selection(population_ranked, elite_size)
    mating_pool = get_mating_pool(current_generation, selection_results)
    children = breed_population(mating_pool, elite_size)
    next_generation = mutate_population(children, mutation_rate)

    return next_generation


def do_genetic_search(starting_point, population_size, elite_size, mutation_rate, generations, distances_matrix):
    population = create_initial_population(starting_point, get_cities(len(distances_matrix)) ,population_size)
    print("Initial distance: " + str(1/rank_routes(population, distances_matrix)[0][1]))

    for i in range(generations):
        population = generate_new_generation(population, elite_size, mutation_rate, distances_matrix)
        print("Distance: " + str(1 / rank_routes(population, distances_matrix)[0][1]))

    print("Final distance: " + str(1/rank_routes(population, distances_matrix)[0][1]))
    index = rank_routes(population, distances_matrix)[0][0]
    route = population[index]
    print("Your ideal route is: " + str(route))
    return 1/rank_routes(population, distances_matrix)[0][1], route


def main():
    # Generate and get city distances
    all_distances = create_random_cities_and_distances()
    # Apply exhaustive search
    #do_exhaustive_search(0, all_distances)
    # Apply genetic search
    #do_genetic_search(0, 100, 20, 0.01, 1, all_distances)
    # Apply greedy search
    # do_greedy_search(0, all_distances)


if __name__ == "__main__":
    main()
