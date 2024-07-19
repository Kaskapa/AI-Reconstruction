import pygame
import neat
import os
import random
from rubiks_cube import Cube
import math
import pickle

WIN_WIDTH = 1000
WIN_HEIGHT = 800

pygame.font.init()
STAT_FONT = pygame.font.SysFont("C:/Users/krist/Documents/Code/PythonWorkspace/flappy Bird/Roboto-Black.ttf", 50)

colors = [(255, 255, 255), (0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 165, 0), (255, 255, 0), (255, 255, 255), (255, 192, 203)]
MOVES = ["L", "L'", "L2", "R", "R'", "R2", "U", "U'", "U2", "D", "D'", "D2", "F", "F'", "F2", "B", "B'", "B2"]
ALTERNATIVEMOVES = [["L", "R"], ["U", "D"], ["F", "B"]]

def draw_cube(which, cube, display):
    row = math.floor(which/10)
    which = which % 10
    for i in range(len(cube.cube[0])):
        for j in range(len(cube.cube[0][i])):
            pygame.draw.rect(display, colors[cube.cube[0][i][j]], ((which*100)+j*5 + 40, (row*50)+i*5 + 10, 3, 3))

    for i in range(len(cube.cube[1])):
        for j in range(len(cube.cube[1][i])):
            pygame.draw.rect(display, colors[cube.cube[1][i][j]], ((which*100)+j*5 + 20, (row*50)+i*5 + 30, 3, 3))

    for i in range(len(cube.cube[2])):
        for j in range(len(cube.cube[2][i])):
            pygame.draw.rect(display, colors[cube.cube[2][i][j]], ((which*100)+j*5 + 40, (row*50)+i*5 + 30, 3, 3))

    for i in range(len(cube.cube[3])):
        for j in range(len(cube.cube[3][i])):
            pygame.draw.rect(display, colors[cube.cube[3][i][j]], ((which*100)+j*5 + 60, (row*50)+i*5 + 30, 3, 3))

    for i in range(len(cube.cube[4])):
        for j in range(len(cube.cube[4][i])):
            pygame.draw.rect(display, colors[cube.cube[4][i][j]], ((which*100)+j*5 + 80, (row*50)+i*5 + 30, 3, 3))

    for i in range(len(cube.cube[5])):
        for j in range(len(cube.cube[5][i])):
            pygame.draw.rect(display, colors[cube.cube[5][i][j]], ((which*100)+j*5 + 40, (row*50)+i*5 + 50, 3, 3))


def draw_window(display, cubes):
    for i in range(len(cubes)):
        if(cubes[i] != None):
            draw_cube(i, cubes[i], display)

    pygame.display.update()
    pygame.display.flip()

increaseIndex = False

index = 0

def main(genomes, config):
    global index
    global increaseIndex
    if increaseIndex:
        index += 1

    increaseIndex = False
    nets = []
    ge = []
    cubes = []

    iteration = 0

    scramble = ""

    while iteration < 3:
        scramble += " " + random.choice(MOVES)
        iteration += 1

    print(scramble)

    for _, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)

        cube = Cube(2)

        cube.do_algorithm(scramble)

        cube.previousMoves = []

        cubes.append(cube)
        g.fitness = 0
        ge.append(g)

    display = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    clock = pygame.time.Clock()

    run = True
    while run:
        clock.tick(100)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()

        for x, cube in enumerate(cubes):
            if(cube != None):
                state = cube.cube
                output = nets[x].activate((state[0][0][0], state[0][0][1], state[0][0][2], state[0][1][0], state[0][1][1], state[0][1][2], state[0][2][0], state[0][2][1], state[0][2][2],
                                        state[1][0][0], state[1][0][1], state[1][0][2], state[1][1][0], state[1][1][1], state[1][1][2], state[1][2][0], state[1][2][1], state[1][2][2],
                                        state[2][0][0], state[2][0][1], state[2][0][2], state[2][1][0], state[2][1][1], state[2][1][2], state[2][2][0], state[2][2][1], state[2][2][2],
                                        state[3][0][0], state[3][0][1], state[3][0][2], state[3][1][0], state[3][1][1], state[3][1][2], state[3][2][0], state[3][2][1], state[3][2][2],
                                        state[4][0][0], state[4][0][1], state[4][0][2], state[4][1][0], state[4][1][1], state[4][1][2], state[4][2][0], state[4][2][1], state[4][2][2],
                                        state[5][0][0], state[5][0][1], state[5][0][2], state[5][1][0], state[5][1][1], state[5][1][2], state[5][2][0], state[5][2][1], state[5][2][2], len(cube.get_white_inserted_pairs()), cube.is_white_cross_solved()))
                move = output.index(max(output))
                cube.do_moves_numerical(move)

        for x, cube in enumerate(cubes):
            if(cube != None):
                whitePairsPaired = []
                for i in range(24):
                    if (cube.is_pair_paired(i) and 0 in cube.get_pair(i)[0] and 0 not in cube.get_pair(i)[1]):
                        whitePairsPaired.append(i)

                if len(cube.get_white_inserted_pairs()) == 4 and cube.is_white_cross_solved():
                    ge[x].fitness += 100
                    print("Solved")
                    # cubes.pop(x)
                    # nets.pop(x)
                    # ge.pop(x)
                    cubes[x] = None
                    nets[x] = None
                    ge[x] = None
                    increaseIndex = True
                    print(index)
                    break

                # elif (cube.previousMoves[-1] == cube.previousMoves[-2] and cube.previousMoves[-2] == cube.previousMoves[-3]):
                #     ge[x].fitness -= 10
                #     cubes.pop(x)
                #     nets.pop(x)
                #     ge.pop(x)

                elif (len(cube.previousMoves) > 2 and MOVES[cube.previousMoves[-1]][0] == MOVES[cube.previousMoves[-2]][0]):
                    ge[x].fitness -= 2
                    # cubes.pop(x)
                    # nets.pop(x)
                    # ge.pop(x)
                    cubes[x] = None
                    nets[x] = None
                    ge[x] = None
                for alter in ALTERNATIVEMOVES:
                    if (len(cube.previousMoves) > 3 and ((MOVES[cube.previousMoves[-1]][0] == alter[0] and MOVES[cube.previousMoves[-2]][0] == alter[1] and MOVES[cube.previousMoves[-3]][0] == alter[0]) or (MOVES[cube.previousMoves[-1]][0] == alter[1] and MOVES[cube.previousMoves[-2]][0] == alter[0] and MOVES[cube.previousMoves[-3]][0] == alter[1]))):
                        ge[x].fitness -= 3
                        # cubes.pop(x)
                        # nets.pop(x)
                        # ge.pop(x)
                        cubes[x] = None
                        nets[x] = None
                        ge[x] = None

        if cubes.count(cubes[0]) == len(cubes):
            break

        draw_window(display, cubes)

def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    p = neat.Population(config)

    #OPTIONAL
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(main, 2000)

    # Save the trained model
    model_path = os.path.join(local_dir, "trained_model.pkl")
    population_path = os.path.join(local_dir, "population.pkl")
    with open(model_path, 'wb') as model_file:
        pickle.dump(winner, model_file)

    with open(population_path, 'wb') as population_file:
        pickle.dump(p, population_file)

def runPopulation(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    with open('NEATv2/population.pkl', 'rb') as f:
        p = pickle.load(f)

    #OPTIONAL
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(main, 1000)

    # Save the trained model
    model_path = os.path.join(local_dir, "trained_model.pkl")
    population_path = os.path.join(local_dir, "population.pkl")
    with open(model_path, 'wb') as model_file:
        pickle.dump(winner, model_file)

    with open(population_path, 'wb') as population_file:
        pickle.dump(p, population_file)

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward.txt")
    runPopulation(config_path)
    # run(config_path)
    print(index)