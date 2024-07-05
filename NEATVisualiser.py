import pygame
import sys
import numpy as np

ROW = 0
COL = 1


class NEATVisualiser:
    """
    This class uses the pygame library to create a visualization given a maze,
    start coordinate, end coordinate, and a route.

    The maze being a numpy array (0: empty cell, 1: wall, 2: start, 3: end)
    The route being a list of directions (0: up, 1: down, 2: left, 3: right)


    Attributes
    __________
    draw_maze           : None
                        Draw the maze and respond to following keyboard inputs:
                        - 'p':    Plays and pauses the visualisation
                        - 'Esc':  Stops and exits the visualisation
                        - 'up arrow':     Increases fps with 1
                        - 'down arrow':   Decreases fps with 1
                        - 'r':    Resets and replays the visualisation
    """
    def __init__(self, fps: int) -> None:
        """
        Initialize the Visualiser object.
        """
        pygame.init()

        self.fps = fps

        self.width = 800
        self.height = 800
        self.screen = pygame.display.set_mode((self.width, self.height))

        self.white = (255, 255, 255)
        self.black = (0, 0, 0)
        self.blue = (0, 0, 255)
        self.red = (255, 0, 0)
        self.green = (0, 255, 0)
        self.grid_color = (200, 200, 200)

        self.pause_text = pygame.font.SysFont('Consolas', 52)
        self.pause_text = self.pause_text.render('Paused, press "p"',
                                                 True, self.black,
                                                 self.grid_color)
        self.solved_text = pygame.font.SysFont('Consolas', 52)
        self.solved_text = self.solved_text.render('Maze solved!',
                                                     True,
                                                     self.black,
                                                     self.grid_color)
        
        self.solved = False


    def draw_maze(self, grid: np.ndarray, start: tuple[int, int], goal: tuple[int, int], route: list[tuple[int, int]]) -> None:
        """
        Draw the maze and respond to following keyboard inputs:
        'p':    Plays and pauses the visualisation
        'Esc':  Stops and exits the visualisation
        'up arrow':     Increases fps with 1
        'down arrow':   Decreases fps with 1
        'r':    Resets and replays the visualisation

        Parameters:
            grid (np.ndarray): The maze grid.
            pos (Tuple[int, int]): The starting position.
            goal (Tuple[int, int]): The goal position.
            route (List[Tuple[int, int]]): List of directions for the route.
        """
        cell_size = min(self.width // np.shape(grid)[COL],
                        self.height // np.shape(grid)[ROW])
        pygame.display.set_caption('Maze')
        clock = pygame.time.Clock()

        pos = [start[0], start[1]]
        route_index = 0

        run = True
        RUNNING, PAUSE = 0, 1
        state = PAUSE

        while run:
            # Gets the keyboard inputs
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        sys.exit()
                    elif event.key == pygame.K_UP:
                        self.fps += 1
                    elif event.key == pygame.K_DOWN and self.fps > 0:
                        self.fps -= 1
                    elif event.key == pygame.K_p:
                        if state == RUNNING:
                            state = PAUSE
                        else:
                            state = RUNNING
                    elif event.key == pygame.K_r:
                        pos = [start[ROW], start[COL]]
                        route_index = 0
                        self.solved = False
                        state = PAUSE

            # When the player reaches the goal, a text will be displayed and the
            # visualisation will be closed after the predetermined delay.
            if pos[ROW] == goal[ROW] and pos[COL] == goal[COL] and not self.solved:
                self.solved = True
                self.screen.blit(self.solved_text,
                                 (self.width // 2 - self.solved_text.get_width() // 2, self.height // 2 - self.solved_text.get_height() // 2))
                pygame.display.flip()
            if self.solved:
                continue


            # Moves the player to the next position.
            if state == RUNNING:
                if route_index < len(route):
                    next_move = route[route_index]
                    route_index += 1
                    if next_move == 0:
                        pos[ROW] -= 1
                    elif next_move == 1:
                        pos[ROW] += 1
                    elif next_move == 2:
                        pos[COL] -= 1
                    elif next_move == 3:
                        pos[COL] += 1

            # Draws the screen and the grid with the corresponding color code.
            self.screen.fill(self.white)
            for row in range(len(grid)):
                for col in range(len(grid[row])):
                    if row == pos[ROW] and col == pos[COL]:
                        color = self.blue
                    elif grid[row][col] == 3:
                        color = self.green
                    elif grid[row][col] == 2:
                        color = self.red
                    elif grid[row][col] == 1:
                        color = self.black
                    else:
                        color = self.white

                    pygame.draw.rect(self.screen,
                                     color,
                                     (col * cell_size, row * cell_size, cell_size, cell_size))

            # Draws the gridlines.
            for x in range(0, self.width, cell_size):
                pygame.draw.line(self.screen,
                                 self.grid_color,
                                 (x, 0),
                                 (x, self.height),
                                 1)
            for y in range(0, self.height, cell_size):
                pygame.draw.line(self.screen,
                                 self.grid_color,
                                 (0, y),
                                 (self.width, y),
                                 1)

            pygame.draw.rect(self.screen,
                             self.black,
                             (0, 0, self.width,
                              self.height), width=3)

            # Draws the pause textbox.
            if state == PAUSE:
                self.screen.blit(self.pause_text,
                                 (self.width // 2 - self.pause_text.get_width() // 2, self.height // 2 - self.pause_text.get_height() // 2))

            pygame.display.flip()
            clock.tick(self.fps)

if __name__ == '__main__':
    grid = np.array([[0, 1, 2, 0, 1, 0, 1, 1],
                     [0, 0, 0, 0, 1, 0, 0, 0],
                     [0, 1, 0, 0, 1, 0, 1, 0],
                     [0, 0, 1, 0, 1, 0, 1, 0],
                     [1, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 1, 1, 1, 0],
                     [1, 0, 0, 0, 1, 0, 1, 0],
                     [1, 1, 0, 1, 0, 3, 0, 0]])
    start = (0, 2)
    goal = (7, 5)
    route = [1, 1, 3, 1, 1, 3, 3, 3, 3, 1, 1, 1, 2, 2]
    v = NEATVisualiser(fps=2)
    v.draw_maze(grid, start, goal, route)