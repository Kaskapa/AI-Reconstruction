import pygame
import neat
import time
import os
import random
from rubiks_cube import Cube
from givemecrossUsable import CrossSolver
import math

WIN_WIDTH = 1200
WIN_HEIGHT = 1000

pygame.font.init()
STAT_FONT = pygame.font.SysFont("C:/Users/krist/Documents/Code/PythonWorkspace/flappy Bird/Roboto-Black.ttf", 50)

colors = [(255, 255, 255), (0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 165, 0), (255, 255, 0), (255, 255, 255)]

def draw_cube(which, cube, display):
    row = math.floor(which/5)
    which = which % 5
    for i in range(len(cube.cube[0])):
        for j in range(len(cube.cube[0][i])):
            pygame.draw.rect(display, colors[cube.cube[0][i][j]], ((which*200)+j*15 + 60, (row*150)+i*15 + 10, 10, 10))

    for i in range(len(cube.cube[1])):
        for j in range(len(cube.cube[1][i])):
            pygame.draw.rect(display, colors[cube.cube[1][i][j]], ((which*200)+j*15 + 10, (row*150)+i*15 + 60, 10, 10))

    for i in range(len(cube.cube[2])):
        for j in range(len(cube.cube[2][i])):
            pygame.draw.rect(display, colors[cube.cube[2][i][j]], ((which*200)+j*15 + 60, (row*150)+i*15 + 60, 10, 10))

    for i in range(len(cube.cube[3])):
        for j in range(len(cube.cube[3][i])):
            pygame.draw.rect(display, colors[cube.cube[3][i][j]], ((which*200)+j*15 + 110, (row*150)+i*15 + 60, 10, 10))

    for i in range(len(cube.cube[4])):
        for j in range(len(cube.cube[4][i])):
            pygame.draw.rect(display, colors[cube.cube[4][i][j]], ((which*200)+j*15 + 160, (row*150)+i*15 + 60, 10, 10))

    for i in range(len(cube.cube[5])):
        for j in range(len(cube.cube[5][i])):
            pygame.draw.rect(display, colors[cube.cube[5][i][j]], ((which*200)+j*15 + 60, (row*150)+i*15 + 110, 10, 10))


def draw_window(display, cubes):
    for cube in range(len(cubes)):
        draw_cube(cube, cubes[cube], display)

    pygame.display.update()
    pygame.display.flip()



def main(genomes, config):
    nets = []
    ge = []
    cubes = []

    for _, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)

        cube = Cube(0)

        scramble = "F2 B L2 U' F2 D' B L' U' F2 R' B2 D F' L2 B2 D' R2 U2 L U B2 D' L'"
        cube.do_algorithm(scramble)

        crossSolver = CrossSolver(scramble)
        crossSolver.solve()

        crossSolution = crossSolver.allSol[0]
        cube.do_algorithm(crossSolution)

        cubes.append(cube)
        g.fitness = 0
        ge.append(g)

    display = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    clock = pygame.time.Clock()

    number_of_moves = 0

    run = True
    while run:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()

        for x, cube in enumerate(cubes):
            ge[x].fitness += 0.1
            state = cube.cube
            output = nets[x].activate((state[0][0][0], state[0][0][1], state[0][0][2], state[0][1][0], state[0][1][1], state[0][1][2], state[0][2][0], state[0][2][1], state[0][2][2],
                                       state[1][0][0], state[1][0][1], state[1][0][2], state[1][1][0], state[1][1][1], state[1][1][2], state[1][2][0], state[1][2][1], state[1][2][2],
                                       state[2][0][0], state[2][0][1], state[2][0][2], state[2][1][0], state[2][1][1], state[2][1][2], state[2][2][0], state[2][2][1], state[2][2][2],
                                       state[3][0][0], state[3][0][1], state[3][0][2], state[3][1][0], state[3][1][1], state[3][1][2], state[3][2][0], state[3][2][1], state[3][2][2],
                                       state[4][0][0], state[4][0][1], state[4][0][2], state[4][1][0], state[4][1][1], state[4][1][2], state[4][2][0], state[4][2][1], state[4][2][2],
                                       state[5][0][0], state[5][0][1], state[5][0][2], state[5][1][0], state[5][1][1], state[5][1][2], state[5][2][0], state[5][2][1], state[5][2][2], len(cube.get_white_inserted_pairs()), cube.is_white_cross_solved()))
            move = output.index(max(output))
            print(output)
            print(move)
            cube.do_moves_numerical(move)

        for x, cube in enumerate(cubes):
            if len(cube.get_white_inserted_pairs()) > 0 and cube.is_white_cross_solved():
                ge[x].fitness -= 1
                print("Solved")
                cubes.pop(x)
                nets.pop(x)
                ge.pop(x)

        if len(cubes) == 0:
            break

        number_of_moves += 1

        draw_window(display, cubes)

def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    p = neat.Population(config)

    #OPTIONAL
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(main, 50)

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward.txt")
    run(config_path)