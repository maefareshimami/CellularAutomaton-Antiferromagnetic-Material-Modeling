import numpy as np
import random as rd

import constants as cst


def initializationPositive()->np.array:     # Choose this function or another Initialize function in the function "averageMagnetization()"
    """Initialize a matrix of spin 1"""
    return np.ones(cst.NB_SPINS, dtype = int)

def initializationNegative()->np.array:
    """Initialize a matrix of spin -1"""
    return np.ones(cst.NB_SPINS, dtype = int)

def initializationRandom()->np.array:
    """Initialize a matrix with random spins"""
    initialization_matrix = np.ones(cst.NB_SPINS, dtype = int)
    for i in range(0, cst.NB_SPINS):
        if rd.random() < 0.5:
            initialization_matrix[i] = - initialization_matrix[i]
    return initialization_matrix

def neighbors(i:int)->list:
    """Create de list of i's neighbors: left, right, bottom, top"""
    list_neighbors = []
    index_row = i // cst.HEIGHT
    index_column = i % cst.HEIGHT
    if index_column == 0:
        list_neighbors.append(i + cst.HEIGHT - 1)
    else:
        list_neighbors.append(i - 1)
    if index_column == cst.HEIGHT - 1:
        list_neighbors.append(i - (cst.HEIGHT - 1))
    else:
        list_neighbors.append(i + 1)
    if index_row == cst.HEIGHT - 1:
        list_neighbors.append(i % cst.HEIGHT)
    else:
        list_neighbors.append(i + cst.HEIGHT)
    if index_row == 0:
        list_neighbors.append(cst.HEIGHT * (cst.HEIGHT - 1) + i)
    else:
        list_neighbors.append(i - cst.HEIGHT)
    return list_neighbors

def energy(array_spins:np.array)->float:
    """Compute the energy with the list of i's neighbors"""
    energy_value = 0.0
    for i in range(0, cst.NB_SPINS):
        list_neighbors_i = neighbors(i)
        for j in list_neighbors_i:
            energy_value += array_spins[i] * array_spins[j]
    return - energy_value / 2.0 * cst.J


def testBoltzmann(delta_e:float)->bool:
    """Compute a random value depending on the Boltzmann distribution"""
    probaility = np.exp(- delta_e / cst.KB_TEMPERATURE)
    if delta_e <= 0 or rd.random() <= probaility:
        return True
    else:
        return False

def computeDeltaE(array_spins:np.array, i:int)->float:
    """Compute the energy difference between neighbors"""
    delta_e = 0.0
    for j in neighbors(i):
        delta_e += cst.J * 2 * array_spins[i] * array_spins[j]
    return delta_e

def monteCarlo(array_spins:np.array)->None:
    """Modify a matrix with the random delta_e and a Monte Carlo method"""
    for _ in range(0, cst.NB_SPINS_TESTED):
        i = rd.randrange(cst.NB_SPINS)
        if testBoltzmann(computeDeltaE(array_spins, i)):
            array_spins[i] = - array_spins[i]
    return None

def averageMagnetization()->(np.array, np.array, float):
    """Main function to compute the new matrix with new spins"""
    array_spins = initializationRandom()
    array_spins_initialization = array_spins.copy()     # Deep copy of the matrix to keep the initialization
    monteCarlo(array_spins)     # Modify in place the list list_spin
    sum_spins = 0
    for spin in array_spins:
        sum_spins += spin
    return array_spins_initialization, array_spins, sum_spins / cst.NB_SPINS_TESTED

def fold(array_spins:np.array)->list:
    """Create a matrix with a HEIGHT * HEIGHT size from a matrix with NB_SPINS size"""
    array_spins_folded = np.ones((cst.HEIGHT, cst.HEIGHT), dtype = int)
    for i in range(0, cst.HEIGHT):
        for j in range(0, cst.HEIGHT):
            array_spins_folded[i, j] = array_spins[i + cst.HEIGHT + j]
    return array_spins_folded 


if __name__ == "__main__":
    initialization_matrix, array_spins, average_magnetization = averageMagnetization()     # You can choose the initialization function at the begining
    with open("antiferromagnetic_matrix.txt", "w") as f:
        f.write(f"Antiferromagnetic Matrix Before:\n{fold(initialization_matrix)}\n\n")
        f.write(f"Antiferromagnetic Matrix After:\n{fold(array_spins)}\n\n")
        f.write(f"Average Magnetization: {round(average_magnetization, 4)} for {round(cst.TEMPERATURE, 2)} K")