from MazeEnv import MazeEnv
from DQLsolver import QLearn
from NEATclass import neatSolver

"""
NOTE: 
When using neat algorithm change number of inputs accordingly to the size of the grid > (length*width)+3 
in the neat-configuration.txt file.

"DQL" for deep q-learning
"NEAT" for NeuroEvolution of Augmented Topologies
"""
def main(env, visualise, model):
    if model == "DQL":
        return print(QLearn(env, visualise).run(env=env, n_episodes=100))
    elif model == "NEAT":
        return neatSolver(env, visualise).run(n_generations=100)

# change "DQL" to "NEAT" to use other model
env = MazeEnv(5, 5, model="DQL")
print(main(env, False, "DQL"))