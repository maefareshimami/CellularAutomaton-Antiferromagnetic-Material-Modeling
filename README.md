# Cellular Automaton - Antiferromagnetic Material Modeling
> If you use my code or a part of it for your projects, thank you for quoting me!

## 2D Modeling
I represent an antiferromagnetic material by a 2D matrix with spin up (value +1) and spin down (value -1) of electrons.
You can choose a matrix, for instance a random matrix, which is created and another is computed using the Boltzmann distribution.
Then, I compute the average magnetization I define as the average value of all spins.
Finally, I show the two matrices in a txt file properly and on a graphic.

## 3D Modeling
Same idea but I show the antiferromagnetic material in 3D with `matplotlib`. The spins up are red and the spins down are blue.

## Protocol
Put all files in a same folder and write `python main.py` in your terminal.
