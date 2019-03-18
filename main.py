from os import path
import itertools
import random
import operator
import sys
import math
import pandas as pd
import numpy as np


def get_distances(distances_file, cities_count):
    """
    This constructs our distance 2 dimensional matrix
    Input: 1 4 Output: [[1,4],[2,1]
           2 1
    """
    all_distances = list()
    for each_line in distances_file:
        current = list()
        for i in range(cities_count):
            current.append(int(each_line.split()[i]))
        all_distances.append(current)
    return all_distances


def get_specific_permutations(start_point, arr):
    """
    This gets rid of opposite mirrored permutations from list
    Input: [1,2,3],1 Output: [(1,2,3),(1,3,2)]
    """
    all_permutations = list(itertools.permutations(arr))  # this is all permutations with little twist n!/2
    specific_permutations = list()  # this is where we have startPoints as first index
    for perm in all_permutations:
        if perm[::-1] in all_permutations:
            all_permutations.remove(perm[::-1])

    for perm in all_permutations:
        if perm[0] == start_point:
            specific_permutations.append(perm)

    return specific_permutations


def do_exhaustive_search(starting_point, distances_matrix):
    start_point = starting_point  # 0 means A, 1 means B, 2 means C etc.
    matrix_length = len(distances_matrix)
    if start_point >= matrix_length or start_point <= 0:
        start_point = 0
    cities = list()  # we resemble [A,B,C,D,E] as [0,1,2,3,4]
    for i in range(matrix_length):
        cities.append(i)
    specific_routes = get_specific_permutations(start_point, cities)
    print("Routes: " + str(specific_routes))
    results = list()  # for all of the sum values for each route

    for route in specific_routes:
        sum = 0
        '''sum += distances_matrix[route[0]][route[1]]
        sum += distances_matrix[route[1]][route[2]]
        sum += distances_matrix[route[2]][route[0]]'''
        for i in range(len(route)-1):
            sum += distances_matrix[route[i]][route[i+1]]
        sum += distances_matrix[route[len(route)-1]][route[0]]
        results.append(sum)
    print("Results: " + str(results))
    index = results.index(min(results))
    print("Your path route is: " + str(specific_routes[index]))
    print("Freshly calculated shortest path takes: " + str(min(results)))


def create_initial_population(population_size, cities_count):
    population = []
    cities = []
    for i in range(cities_count):
        cities.append(i)
    for i in range(population_size):
        population.append(random.sample(cities, cities_count))
    return population


def get_fitness_score(route, distances_matrix):
    distance = 0
    for i in range(len(route)-1):
        distance += distances_matrix[route[i]][route[i+1]]
    distance += distances_matrix[route[len(route)-1]][route[0]]
    return 1 / float(distance)


def rank_routes(population, distances_matrix):
    fitness_results = {}
    for i in range(len(population)):
        fitness_results[i] = get_fitness_score(population[i], distances_matrix)
    return sorted(fitness_results.items(), key=operator.itemgetter(1), reverse=True)


def selection(population_ranked, elite_size):
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
    mating_pool = []
    for i in range(len(selection_results)):
        index = selection_results[i]
        mating_pool.append(population[index])
    return mating_pool


def breed(parent1, parent2):
    child = []
    childPart1 = []
    childPart2 = []

    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))
    startingPointOfGene = min(geneA, geneB)
    endingPointOfGene = max(geneA, geneB)

    for i in range(startingPointOfGene,endingPointOfGene):
        childPart1.append(parent1[i])

    childPart2 = [item for item in parent2 if item not in childPart1]

    child = childPart1 + childPart2
    return child


def breed_population(mating_pool, elite_size):
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
    for swapped in range(len(individual)):
        if random.random() < mutation_rate:
            swapWith = int(random.random() * len(individual))
            city1 = individual[swapped]
            city2 = individual[swapWith]
            individual[swapped] = city2
            individual[swapWith] = city1
    return individual


def mutate_population(population, mutation_rate):
    population_mutated = []

    for i in range(len(population)):
        individual = mutate(population[i], mutation_rate)
        population_mutated.append(individual)
    return population_mutated


def generate_new_generation(current_generation, elite_size, mutation_rate, distances_matrix):
    popRanked = rank_routes(current_generation, distances_matrix)
    selection_results = selection(popRanked, elite_size)
    mating_pool = get_mating_pool(current_generation, selection_results)
    children = breed_population(mating_pool, elite_size)
    next_generation = mutate_population(children, mutation_rate)
    return next_generation


def do_genetic_search(population_size, elite_size, mutation_rate, generations, distances_matrix):
    population = create_initial_population(population_size, len(distances_matrix[0]))
    print("Initial distance: " + str(1/rank_routes(population, distances_matrix)[0][1]))

    for i in range(generations):
        population = generate_new_generation(population, elite_size, mutation_rate, distances_matrix)

    print("Final distance: " + str(1/rank_routes(population, distances_matrix)[0][1]))
    index = rank_routes(population, distances_matrix)[0][0]
    route = population[index]
    return route


def main():
    # Get city distances
    directory_data_filename = "data/10cities"
    variable_cities_count = 10
    distances_file = open(path.join(path.dirname(__file__), directory_data_filename))
    all_distances = get_distances(distances_file, variable_cities_count)
    # Apply exhaustive search
    # do_exhaustive_search(0, all_distances)
    # Apply genetic search
    do_genetic_search(100, 20, 0.01, 500, all_distances)


if __name__ == "__main__":
    main()