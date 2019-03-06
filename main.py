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


def get_specific_permutations(array):
    """
    This gets rid of opposite mirrored permutations from list
    Input: [1,2,3] Output: [(1,2,3),(1,3,2),(2,1,3)]
    """
    all_permutations = list(itertools.permutations(array))
    for perm in all_permutations:
        if perm[::-1] in all_permutations:
            all_permutations.remove(perm[::-1])
    return all_permutations


def do_exhaustive_search(starting_point, distances_matrix):
    # [TODO] WORK ON THIS LATER ON
    start_point = starting_point  # 0 means A
    matrix_length = len(distances_matrix)
    if starting_point >= matrix_length or start_point <= 0:
        start_point = 0

    results = list()  # for all of the sum values


    #sum = 0
    #for i in range(matrix_length):
        #for j in range(matrix_length):
            #distances_matrix[]



def main():
    # Get city distances
    data_filename = "data/5cities"
    distances_file = open(path.join(path.dirname(__file__), data_filename))
    all_distances = get_distances(distances_file,5)
    print(all_distances)
    print(get_specific_permutations([1,2,3]))













if __name__ == "__main__":
    main()