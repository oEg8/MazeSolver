# Maze Solver with NEAT and Visualization

This repository contains Python scripts for solving mazes using the NeuroEvolution of Augmented Topologies (NEAT) algorithm and visualizing the process.

## Contents

- [Description](#description)
- [Requirements](#requirements)
- [Usage](#usage)
- [Files](#files)
- [Contributors](#contributors)
- [License](#license)
- [References](#references)

## Description

The project includes three main Python scripts:

1. **neat_solver.py**: Implements a maze-solving algorithm using NEAT. Given a maze, the solver uses NEAT to evolve a solution, finding the optimal path from the start to the goal.
   
2. **maze_maker.py**: Generates randomized mazes with specified characteristics, such as size, obstacle density, and minimum route length. The mazes are represented as numpy arrays, with different values denoting empty cells, walls, start, and end points.

3. **visualizer.py**: Provides a visualization of a maze-solving process. It uses pygame to create a graphical interface where the NEAT solver navigates through the maze, displaying the path and progress in real-time.

4. **neat-configuration.txt**: Configuration file where all parameters are defined. 

## Requirements

- Python 3.x
- numpy
- neat-python
- pygame

You can install the dependencies using pip:

```
pip install numpy, neat-python, pygame
```

## Usage

1. Clone the repository:

```bash
git clone https://github.com/oEg8/NEAT-maze-solver.git
```

2. Navigate to the project directory:

```bash
cd yourdirectory
```

3. Run the main script:

```bash
python NEATclass.py
```

## Files

- `grids`: Folder where grids are saved.
- `.gitignore`: Files that are not pushed.
- `best_neat_solver.plk`: Pre-trained model.
- `maze_maker.py`: Maze generation script.
- `neat-configuration.txt`: Model configuration file.
- `NEATclass.py`: NEAT algorithm implementation for solving mazes.
- `requirements.txt`: Necessary libraries to run all.
- `Visualizer.py`: Visualization script for maze solving.

## Contributors

- https://github.com/oEg8

## License

This project is licensed under the MIT License

## References

- NEAT-Python: [GitHub](https://github.com/CodeReclaimers/neat-python)
- Pygame Documentation: [Pygame](https://www.pygame.org/docs/)
