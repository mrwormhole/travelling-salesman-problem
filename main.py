from os import path
import itertools
import sys
import math


def get_distances(distances_file, matrix_length):
    """
    This constructs our distance 2 dimensional matrix
    Input: 1 4 Output: [[1,4],[2,1]
           2 1
    """
    all_distances = list()
    for each_line in distances_file:
        current = list()
        for i in range(matrix_length):
            current.append(int(each_line.split()[i]))
        all_distances.append(current)
    return all_distances


def get_specific_permutations(startPoint, arr):
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
        if perm[0] == startPoint:
            specific_permutations.append(perm)

    return specific_permutations


def do_exhaustive_search(starting_point, distances_matrix):
    start_point = starting_point  # 0 means A, 1 means B, 2 means C etc.
    matrix_length = len(distances_matrix)
    if starting_point >= matrix_length or start_point <= 0:
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
    print("Freshly calculated path takes: " + str(min(results)))


def main():
    # Get city distances
    data_filename = "data/5cities"
    cities_count = 5
    distances_file = open(path.join(path.dirname(__file__), data_filename))
    all_distances = get_distances(distances_file, cities_count)
    # Apply exhaustive search
    do_exhaustive_search(0, all_distances)













if __name__ == "__main__":
    main()