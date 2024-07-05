# Maze Navigation

This project implements a maze navigation system using two diffrent algorithms: 
- Deep Q-learning
- NeuroEvolution of Augmented Topologies 

## Contents

- [Description](#description)
- [Requirements](#requirements)
- [Usage](#usage)
- [Files](#files)
- [Contributors](#contributors)
- [License](#license)
- [References](#references)

## Description

This project implements a maze navigation system using two diffrent algorithms: Deep Q-learning (reinforcment learning) and NeuroEvolution of Augmented Topologies (genetic algorithm). It generates random mazes, trains a model to navigate through them, and visualizes the learning process.
It focusses on being able to handle multiple action types. The reason for this ..

## Requirements

- numpy
- pygame
- keras
- tensorflow
- neat-python

## Usage

## Installation

To install the project locally, follow these steps:

1. **Clone the Repository**:
    ```sh
    git clone https://github.com/your-username/your-repository.git
    ```

2. **Navigate to the Project Directory**:
    ```sh
    cd your-repository
    ```

3. **Install Dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

## Files

- `main.py`: Main script to execute the maze navigation system with model of choice.

- `grids`: Folder where grids are saved.
- `.gitignore`: Files git should ignore.
- `best_dql_solver.h5`: Pre-trained dql model.
- `best_neat_solver.plk`: Pre-trained neat model.
- `DQLsolver.py`: Contains the DQL algorithm.
- `DQLVisualiser.py`: Visualises the learning process of the DQL algorithm.
- `Experience_Replay.py`: Implements experience replay for training the Q-learning model.
- `MazeEnv.py`: Defines the maze environment class.
- `MazeMaker.py`: Generates random mazes for navigation.
- `neat-configuration.txt`: Model configuration file.
- `NEATclass.py`: NEAT algorithm implementation for solving mazes.
- `NEATVisualiser.py`: Visualization script for NEAT maze solving.
- `README.md`: This file, providing an overview of the project.
- `requirements.txt`: Dependencies to run all.

## Contributors

- https://github.com/oEg8

## License

This project is licensed under the MIT License

## Refrences

- Pygame Documentation: [Pygame](https://www.pygame.org/docs/)
- Tensorflow Documentation: [Tensorflow](https://www.tensorflow.org/api_docs)
- NEAT-Python: [GitHub](https://github.com/CodeReclaimers/neat-python)