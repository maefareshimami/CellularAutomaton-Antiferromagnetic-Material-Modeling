import numpy as np


HEIGHT = 20
assert(HEIGHT % 2 == 0)
NB_SPINS_SLICE = HEIGHT * HEIGHT
NB_SPINS = HEIGHT * HEIGHT * HEIGHT
NB_SPINS_TESTED = 80000

J = 1.0    # Exchange integral: arbitrary value

KB = 1.0     # Boltzmann constant (J/K): arbitrary value
TEMPERATURE = 373.0     # Kelvin (K)
KB_TEMPERATURE = KB * TEMPERATURE     # Boltzmann constant multiplied by the temperature (Kelvin)