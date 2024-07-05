import os
import neat
from MazeEnv import MazeEnv
import numpy as np
from NEATVisualiser import NEATVisualiser
import pickle

ROW = 0
COL = 1

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3


def save_model(network, filename: str):
    """Saves NEAT model with pickle"""
    with open(filename, 'wb') as f:
        pickle.dump(network, f)
        f.close()

def load_model(filename: str):
    """Loads NEAT model with pickle"""
    return pickle.load(open(filename, 'rb')) 


class neatSolver:
    """
    This class uses a NeuroEvolution of Augmented Topologies (NEAT) algorithm to solve a maze.


    Attributes
    __________
    move                : list[int, int]
                        Move the player by one step in a given direction.
    random_move         : tuple[list[int, int], int]
                        Move the player randomly among possible actions.
    novelty_score       : float
                        Calculate the novelty score based on unique positions.
    possible_actions    :list[int, int]
                        Return possible moves given a grid and a position.
    distance_to_goal    : int
                        Calculate Manhattan distance from current position to goal.
    reset_position      : list[int, int]
                        Reset the player position to the starting position.
    calc_fitness        : int
                        Calculate the fitness for a given genome.
    eval_genomes        : None
                        Evaluate the genomes and assign fitness.
    run                 : tuple[int, int, int]
                        Run the NEAT algorithm.
    """
    def __init__(self, env: MazeEnv, visualise: bool = False) -> None:
        """
        Initializes the neatSolver object.

        Parameters:
            env (MazeEnv): The maze environment.
            visualise (bool): Wether the learning process should be visualised.
        """
        self.env = env
        self.visualise = visualise

        self.max_steps = 100
        self.step_cost = 1
        self.goal_reward = 100
        self.illegal_pen = 3
        self.novel_pen = 1
        self.exploration_factor = 0.1

        self.winner_directions = []
        np.random.seed(1)

        if self.visualise:
            self.visualiser = NEATVisualiser(fps=2)
    

    def novelty_score(self, path: list[tuple[int, int]]) -> float:
        """
        Calculate the novelty score based on unique positions.

        Parameters:
            path (list): List of positions visited.

        Returns:
            float: Novelty score.
        """
        novel_score = 0
        unique_positions = []
        for item in path:
            if item not in unique_positions:
                unique_positions.append(item)

        # counts the amount of times the algorithms has been on a posistion and
        # penalizes with the novelty penalty * that amount
        for i in range(len(unique_positions)):
            pen_multiplier = path.count(unique_positions[i])
            if pen_multiplier > 1:
                novel_score += pen_multiplier-1 * self.novel_pen

        return novel_score
    

    def select_action(self, output: list, action_type: int) -> int:
        """
        Selects the action to be taken from the output of the model.

        Parameters:
            output (list): The output from the models network.
            action_type (int): The action type to be taken.
        
        Returns:
            int: The action.
        """
        if np.random.rand() < self.exploration_factor:
            return np.random.choice(self.env.get_action_space(action_type))
        else:
            if action_type == 0:
                return np.argmax(output[:4])  # Player actions
            else:
                return np.argmax(output[4:]) + 4  # Goal actions


    def calc_fitness(self, genome, config: str) -> int:
        """
        Calculate the fitness for a given genome.

        Parameters:
            genome: Genome to calculate fitness for.
            config (str): NEAT configuration file path.

        Returns:
            int: Fitness value.
        """
        fitness = 0
        self.env.reset()
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        actions = []

        for _ in range(self.max_steps):
            # define the state which will be used as input for the algorithm
            state = self.env.calculate_state()

            # generate output from input (state)
            output = net.activate(state)

            action_type = self.env.get_action_type()
            action = self.select_action(output, action_type)
            
            possible_actions = self.env.possible_actions(action_type)

            # checks if the action is possible and take it if so, otherwise
            # step is illegal so genome will be penalized

            if action in possible_actions:
                fitness -= self.step_cost
                self.env.move(action, action_type)
                actions.append(action)
            else:
                fitness -= self.illegal_pen
                # to introduce randomness the genome will take a random action a percentage of times
                # (illegal_mutation_rate) the genome wants to take a illegal action
                if np.random.rand() < self.exploration_factor:
                    self.env.move(rand_action := np.random.choice(possible_actions), action_type)
                    actions.append(rand_action)


            # checks if the goal is reached and breaks out of the loop if so
            if self.env.test_for_completion():
                fitness += self.goal_reward
                # saves the winning directions for viualisation
                self.winner_directions = actions
                break

        fitness -= self.env.distance_to_goal() ##

        return fitness


    def eval_genomes(self, genomes, config: str) -> None:
        """
        Evaluate the genomes and assign fitness.

        Parameters:
            genomes: List of genomes to evaluate.
            config (str): NEAT configuration file path.
        """
        for genome_id, genome in genomes:  # genome_id is used by the neat-python library  
            fitness = self.calc_fitness(genome, config)
            genome.fitness = fitness


    def run(self, n_generations: int) -> tuple[int, int, int]:
        """
        Run the NEAT algorithm.

        Parameters:
            num_generations (int): Maximum nomber of generations allowed.
        
        Returns:
            tuple[int, int, int]:
                - max_gen_fitness: Maximum generation fitness.
                - num_generations: Number of generations completed.
                - winner_directions: Directions of best genome
        """
        config_file = 'neat-configuration.txt'

        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_file)

        # Create the population
        p = neat.Population(config)

        # check if an earlier trained model exists and use that best genome as the initial training genome
        if os.path.exists('best_neat_solver.pkl'):
            # Load the previously trained model
            winner_net = load_model('best_neat_solver.pkl')

            # Create a genome from the loaded model's structure
            winner_genome = neat.DefaultGenome(1)  # Set the ID to 1
            winner_genome.configure_new(config.genome_config)
            winner_genome.nodes = winner_net.nodes
            winner_genome.connections = winner_net.connections
            winner_genome.fitness = winner_net.fitness

            p.population[0] = winner_genome


        p.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)

        p.run(self.eval_genomes, n_generations)

        # Save the best genome
        best_genome = stats.best_genome()
        save_model(best_genome, 'best_neat_solver.pkl')

        max_gen_fitness = best_genome.fitness
        num_generations = len(stats.get_fitness_mean())

        print('\nBest genome:\n{!s}'.format(best_genome))

        if self.visualise:
            grid = self.env.get_grid()
            start = self.env.get_start()
            goal = self.env.get_goal()
            self.visualiser.draw_maze(grid, [start[ROW], start[COL]], goal, self.winner_directions)

        return max_gen_fitness, num_generations, self.winner_directions


if __name__ == '__main__':
    env = MazeEnv(5, 5, model='NEAT')
    neatSolver(env=env, visualise=False).run(100)


