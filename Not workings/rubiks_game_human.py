import pygame
import random
from enum import Enum
from collections import namedtuple
from rubiks_cube import Cube

pygame.init()

class Move(Enum):
    LEFT = 1
    LEFTPRIME = 2
    LEFTTWO = 3
    RIGHT = 4
    RIGHTPRIME = 5
    RIGHTTWO = 6
    UP = 7
    UPPRIME = 8
    UPTWO = 9
    DOWN = 10
    DOWNPRIME = 11
    DOWNTWO = 12
    FRONT = 13
    FRONPRIME = 14
    FRONTTWO = 15
    BACK = 16
    BACKPRIME = 17
    BACKTWO = 18
    X = 19
    XPRIME = 20
    Y = 21
    YPRIME = 22
    Z = 23
    ZPRIME = 24
    NONE = 25

Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
ORANGE = (255, 165, 0)
YELLOW = (255, 255, 0)
BLACK = (0, 0, 0)

SPEED = 20

class RubiksGame:
    def __init__(self, w=700, h=580):
        self.w = w
        self.h = h
        self.direction = Move.NONE
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('RUBIKS CUBE')
        self.clock = pygame.time.Clock()
        self.colors = [
            (255, 255, 255),
            (255, 165, 0),
            (0, 255, 0),
            (255, 0, 0),
            (0, 0, 255),
            (255, 255, 0)
        ]

        self.cube = Cube(0)

    def play_step(self):
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_l:
                    self.direction = Move.LEFT
                elif event.key == pygame.K_k:
                    self.direction = Move.LEFTPRIME
                elif event.key == pygame.K_j:
                    self.direction = Move.LEFTTWO
                elif event.key == pygame.K_r:
                    self.direction = Move.RIGHT
                elif event.key == pygame.K_e:
                    self.direction = Move.RIGHTPRIME
                elif event.key == pygame.K_w:
                    self.direction = Move.RIGHTTWO
                elif event.key == pygame.K_u:
                    self.direction = Move.UP
                elif event.key == pygame.K_i:
                    self.direction = Move.UPPRIME
                elif event.key == pygame.K_o:
                    self.direction = Move.UPTWO
                elif event.key == pygame.K_d:
                    self.direction = Move.DOWN
                elif event.key == pygame.K_s:
                    self.direction = Move.DOWNPRIME
                elif event.key == pygame.K_a:
                    self.direction = Move.DOWNTWO
                elif event.key == pygame.K_f:
                    self.direction = Move.FRONT
                elif event.key == pygame.K_g:
                    self.direction = Move.FRONTTWO
                elif event.key == pygame.K_h:
                    self.direction = Move.FRONPRIME
                elif event.key == pygame.K_b:
                    self.direction = Move.BACK
                elif event.key == pygame.K_n:
                    self.direction = Move.BACKTWO
                elif event.key == pygame.K_m:
                    self.direction = Move.BACKPRIME
                elif event.key == pygame.K_x:
                    self.direction = Move.X
                elif event.key == pygame.K_c:
                    self.direction = Move.XPRIME
                elif event.key == pygame.K_y:
                    self.direction = Move.Y
                elif event.key == pygame.K_t:
                    self.direction = Move.YPRIME
                elif event.key == pygame.K_z:
                    self.direction = Move.Z
                elif event.key == pygame.K_v:
                    self.direction = Move.ZPRIME

        # 2. move
        if(self.direction != Move.NONE):
            self._move(self.direction)

        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. return game over and score
        return False, 0

    def _move(self, move):
        if move == Move.LEFT:
            self.cube.left_turn()
            self.direction = Move.NONE
        elif move == Move.LEFTPRIME:
            self.cube.left_turn();
            self.cube.left_turn();
            self.cube.left_turn();
            self.direction = Move.NONE

        elif move == Move.LEFTTWO:
            self.cube.left_turn();
            self.cube.left_turn();
            self.direction = Move.NONE

        elif move == Move.RIGHT:
            self.cube.right_turn()
            self.direction = Move.NONE

        elif move == Move.RIGHTPRIME:
            self.cube.right_turn();
            self.cube.right_turn();
            self.cube.right_turn();
            self.direction = Move.NONE

        elif move == Move.RIGHTTWO:
            self.cube.right_turn();
            self.cube.right_turn();
            self.direction = Move.NONE

        elif move == Move.UP:
            self.cube.up_turn()
            self.direction = Move.NONE

        elif move == Move.UPPRIME:
            self.cube.up_turn();
            self.cube.up_turn();
            self.cube.up_turn();
            self.direction = Move.NONE

        elif move == Move.UPTWO:
            self.cube.up_turn();
            self.cube.up_turn();
            self.direction = Move.NONE

        elif move == Move.DOWN:
            self.cube.down_turn()
            self.direction = Move.NONE

        elif move == Move.DOWNPRIME:
            self.cube.down_turn();
            self.cube.down_turn();
            self.cube.down_turn();
            self.direction = Move.NONE

        elif move == Move.DOWNTWO:
            self.cube.down_turn();
            self.cube.down_turn();
            self.direction = Move.NONE

        elif move == Move.FRONT:
            self.cube.front_turn()
            self.direction = Move.NONE

        elif move == Move.FRONTTWO:
            self.cube.front_turn();
            self.cube.front_turn();
            self.direction = Move.NONE

        elif move == Move.FRONPRIME:
            self.cube.front_turn();
            self.cube.front_turn();
            self.cube.front_turn();
            self.direction = Move.NONE

        elif move == Move.BACK:
            self.cube.back_turn()
            self.direction = Move.NONE

        elif move == Move.BACKTWO:
            self.cube.back_turn();
            self.cube.back_turn();
            self.direction = Move.NONE

        elif move == Move.BACKPRIME:
            self.cube.back_turn();
            self.cube.back_turn();
            self.cube.back_turn();
            self.direction = Move.NONE

        elif move == Move.X:
            self.cube.x_rotation()
            self.direction = Move.NONE

        elif move == Move.XPRIME:
            self.cube.x_rotation();
            self.cube.x_rotation();
            self.cube.x_rotation();
            self.direction = Move.NONE

        elif move == Move.Y:
            self.cube.y_rotation()
            self.direction = Move.NONE

        elif move == Move.YPRIME:
            self.cube.y_rotation();
            self.cube.y_rotation();
            self.cube.y_rotation();
            self.direction = Move.NONE

        elif move == Move.Z:
            self.cube.z_rotation()
            self.direction = Move.NONE

        elif move == Move.ZPRIME:
            self.cube.z_rotation();
            self.cube.z_rotation();
            self.cube.z_rotation();
            self.direction = Move.NONE

        elif move == Move.NONE:
            pass

    def _update_ui(self):
        self.display.fill(BLACK)
        self._drawCube()
        pygame.display.flip()

    def _drawCube(self):
        for i in range(len(self.cube.cube[0])):
            for j in range(len(self.cube.cube[0][i])):
                pygame.draw.rect(self.display, self.colors[self.cube.cube[0][i][j]], (j*50 + 185, i*50 + 25, 40, 40))

        for i in range(len(self.cube.cube[1])):
            for j in range(len(self.cube.cube[1][i])):
                pygame.draw.rect(self.display, self.colors[self.cube.cube[1][i][j]], (j*50 + 25, i*50 + 185, 40, 40))

        for i in range(len(self.cube.cube[2])):
            for j in range(len(self.cube.cube[2][i])):
                pygame.draw.rect(self.display, self.colors[self.cube.cube[2][i][j]], (j*50 + 185, i*50 + 185, 40, 40))

        for i in range(len(self.cube.cube[3])):
            for j in range(len(self.cube.cube[3][i])):
                pygame.draw.rect(self.display, self.colors[self.cube.cube[3][i][j]], (j*50 + 345, i*50 + 185, 40, 40))

        for i in range(len(self.cube.cube[4])):
            for j in range(len(self.cube.cube[4][i])):
                pygame.draw.rect(self.display, self.colors[self.cube.cube[4][i][j]], (j*50 + 505, i*50 + 185, 40, 40))

        for i in range(len(self.cube.cube[5])):
            for j in range(len(self.cube.cube[5][i])):
                pygame.draw.rect(self.display, self.colors[self.cube.cube[5][i][j]], (j*50 + 185, i*50 + 345, 40, 40))


if __name__ == '__main__':
    game = RubiksGame()

    # game loop
    while True:
        game_over, score = game.play_step()

        if game_over == True:
            break

    print('Final Score', score)


    pygame.quit()