import numpy as np
import random as rd
import matplotlib.pyplot as plt

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
    """Create de list of i's neighbors: left, right, bottom, top, forward, bacward"""
    i_prime = i % cst.NB_SPINS_SLICE
    i_slice = i // cst.NB_SPINS_SLICE
    list_neighbors = []
    index_row = i_prime // cst.HEIGHT
    index_column = i_prime % cst.HEIGHT
    if index_column == 0:     # Left neighbor
        list_neighbors.append(i + cst.HEIGHT - 1)
    else:
        list_neighbors.append(i - 1)
    if index_column == cst.HEIGHT - 1:     # Right neighbor
        list_neighbors.append(i - (cst.HEIGHT - 1))
    else:
        list_neighbors.append(i + 1)
    if index_row == cst.HEIGHT - 1:     # Bottom neighbor
        list_neighbors.append(i % cst.HEIGHT)
    else:
        list_neighbors.append(i + cst.HEIGHT)
    if index_row == 0:     # Top neighbor
        list_neighbors.append(cst.HEIGHT * (cst.HEIGHT - 1) + i)
    else:
        list_neighbors.append(i - cst.HEIGHT)
    if i_slice == 0:     # Forward neighbor
        list_neighbors.append(i + (cst.HEIGHT - 1) * cst.NB_SPINS_SLICE)
    else:
        list_neighbors.append(i - cst.NB_SPINS_SLICE)
    if i_slice == cst.HEIGHT - 1:     # Backward neighbor
        list_neighbors.append(i - (cst.HEIGHT - 1) * cst.NB_SPINS_SLICE)
    else:
        list_neighbors.append(i + cst.NB_SPINS_SLICE)
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
    array_spins_folded = np.ones((cst. HEIGHT, cst.HEIGHT, cst.HEIGHT), dtype = int)
    for k in range(0, cst.HEIGHT):
        for i in range(0, cst.HEIGHT):
            for j in range(0, cst.HEIGHT):
                array_spins_folded[k, i, j] = array_spins[k * (i * cst.HEIGHT + j)]
    return array_spins_folded

def createMatrixDisplay(matrix_folded:np.array)->(list, list, list, list, list, list):
    """Create lists to show the 3D matrix with matplotlib"""
    x_up = []
    y_up = []
    z_up = []
    x_down = []
    y_down = []
    z_down = []
    for k in range(0, cst.HEIGHT):
        for i in range(0, cst.HEIGHT):
            for j in range(0, cst.HEIGHT):
                if matrix_folded[k, i, j] == 1:
                    x_up.append(i)
                    y_up.append(j)
                    z_up.append(k)
                else:
                    x_down.append(i)
                    y_down.append(j)
                    z_down.append(k)
    return x_up, y_up, z_up, x_down, y_down, z_down


if __name__ == "__main__":
    array_spins_initialization, array_spins, average_magnetization = averageMagnetization()     # You can choose the initialization function at the begining
    array_spins_initialization_folded = fold(array_spins_initialization)
    array_spins_folded = fold(array_spins)

    print(f"Average Magnetization: {round(average_magnetization, 4)} for {round(cst.TEMPERATURE, 2)} K")
    
    x_up_init, y_up_init, z_up_init, x_down_init, y_down_init, z_down_init = createMatrixDisplay(array_spins_initialization_folded)
    x_up, y_up, z_up, x_down, y_down, z_down = createMatrixDisplay(array_spins_folded)
    
    fig = plt.figure("Cellular Automaton - Antiferromagnetic Material")
    ax_1 = fig.add_subplot(1, 2, 1, projection = "3d")

    ax_1.grid(color = "black", linestyle = "-", linewidth = 0.1)     # Show Initialization Matrix
    ax_1.axis("equal")
    ax_1.scatter(x_up_init, y_up_init, z_up_init, color = "red", marker = ".", linewidths = 0.001)
    ax_1.scatter(x_down_init, y_down_init, z_down_init, color = "blue", marker = ".", linewidths = 0.001)
    ax_1.set_title("Initialization")
    ax_1.set_xlabel("Cell n°")
    ax_1.set_ylabel("Cell n°")
    ax_1.set_zlabel("Cell n°")
    
    ax_2 = fig.add_subplot(1, 2, 2, projection = "3d")     # Show Modeled Matrix
    ax_2.grid(color = "black", linestyle = "-", linewidth = 0.1)
    ax_2.axis("equal")
    ax_2.scatter(x_up, y_up, z_up, color = "red", marker = ".", linewidths = 0.001)
    ax_2.scatter(x_down, y_down, z_down, color = "blue", marker = ".", linewidths = 0.001)
    ax_2.set_title("Modeling")
    ax_2.set_xlabel("Cell n°")
    ax_2.set_ylabel("Cell n°")
    ax_2.set_zlabel("Cell n°")

    plt.tight_layout()
    plt.show()