import pygame
import random
from enum import Enum
from collections import namedtuple
from rubiks_cube import Cube
import numpy as np

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
    X2 = 25
    Y2 = 26
    Z2 = 27
    MIDDLE = 28
    MIDDLEPRIME = 29
    MIDDLETWO = 30
    EQUATORIAL = 31
    EQUATORIALPRIME = 32
    EQUATORIALTWO = 33
    STANDING = 34
    STANDINGPRIME = 35
    STANDINGTWO = 36
    LEFTWIDE = 37
    LEFTWIDEPRIME = 38
    LEFTWIDETWO = 39
    RIGHTWIDE = 40
    RIGHTWIDEPRIME = 41
    RIGHTWIDETWO = 42
    UPWIDE = 43
    UPWIDEPRIME = 44
    UPWIDETWO = 45
    DOWNWIDE = 46
    DOWNWIDEPRIME = 47
    DOWNWIDETWO = 48
    FRONTWIDE = 49
    FRONTWIDEPRIME = 50
    FRONTWIDETWO = 51
    BACKWIDE = 52
    BACKWIDEPRIME = 53
    BACKWIDETWO = 54
    NONE = 55

# rgb colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
ORANGE = (255, 165, 0)
YELLOW = (255, 255, 0)
BLACK = (0, 0, 0)

SPEED = 1000

class RubiksGameAI:
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
        self.move_count = 0
        self.cube = Cube(0)
        self.previous_moves = []

    def reset(self):
        self.cube = Cube(0)
        self.direction = Move.NONE
        self.move_count = 0
        self.previous_moves = []

    def play_step(self, action):
        self.previous_moves.append(action)
        self.move_count += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 2. move
        self._move(action)
        reward = 0
        pairsInserted = self.cube.get_white_inserted_pairs()
        done = False
        moveOnToTheNextState = False

        if len(self.previous_moves) > 1 and self.previous_moves[-1] == self.previous_moves[-2]:
            reward -= 5
        if len(pairsInserted) > 0 and self.cube.is_white_cross_solved():
            for i in range(len(pairsInserted)):
                reward += 10
        if (len(pairsInserted) > 0 and self.cube.is_white_cross_solved()) or self.move_count > 100:
            done = True
            if len(pairsInserted) == 0 or not self.cube.is_white_cross_solved():
                reward -= 2
            else:
                moveOnToTheNextState = True
                plusReward = 100-self.move_count

                reward += plusReward

            self.move_count = 0

            pairs = []
            for i in range(24):
                if(self.cube.is_pair_paired(i)):
                    pairs.append(self.cube.get_pair(i))

            return reward, done, moveOnToTheNextState, len(pairs), len(pairsInserted)

        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. return game over and score
        return reward, done, moveOnToTheNextState, 0, 0

    def _move(self, action):
        if np.array_equal(action, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]):
            self.direction = Move.LEFT
        elif np.array_equal(action, [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]):
            self.direction = Move.LEFTPRIME
        elif np.array_equal(action, [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]):
            self.direction = Move.LEFTTWO
        elif np.array_equal(action, [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]):
            self.direction = Move.RIGHT
        elif np.array_equal(action, [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]):
            self.direction = Move.RIGHTPRIME
        elif np.array_equal(action, [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]):
            self.direction = Move.RIGHTTWO
        elif np.array_equal(action, [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]):
            self.direction = Move.UP
        elif np.array_equal(action, [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]):
            self.direction = Move.UPPRIME
        elif np.array_equal(action, [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]):
            self.direction = Move.UPTWO
        elif np.array_equal(action, [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]):
            self.direction = Move.DOWN
        elif np.array_equal(action, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]):
            self.direction = Move.DOWNPRIME
        elif np.array_equal(action, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]):
            self.direction = Move.DOWNTWO
        elif np.array_equal(action, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]):
            self.direction = Move.FRONT
        elif np.array_equal(action, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]):
            self.direction = Move.FRONTTWO
        elif np.array_equal(action, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]):
            self.direction = Move.FRONPRIME
        elif np.array_equal(action, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]):
            self.direction = Move.BACK
        elif np.array_equal(action, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]):
            self.direction = Move.BACKTWO
        elif np.array_equal(action, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]):
            self.direction = Move.BACKPRIME
        elif np.array_equal(action, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]):
            self.direction = Move.X
        elif np.array_equal(action, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]):
            self.direction = Move.XPRIME
        elif np.array_equal(action, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]):
            self.direction = Move.Y
        elif np.array_equal(action, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]):
            self.direction = Move.YPRIME
        elif np.array_equal(action, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]):
            self.direction = Move.Z
        elif np.array_equal(action, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]):
            self.direction = Move.ZPRIME
        elif np.array_equal(action, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]):
            self.direction = Move.X2
        elif np.array_equal(action, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]):
            self.direction = Move.Y2
        elif np.array_equal(action, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]):
            self.direction = Move.Z2
        elif np.array_equal(action, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]):
            self.direction = Move.MIDDLE
        elif np.array_equal(action, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]):
            self.direction = Move.MIDDLEPRIME
        elif np.array_equal(action, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]):
            self.direction = Move.MIDDLETWO
        elif np.array_equal(action, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]):
            self.direction = Move.EQUATORIAL
        elif np.array_equal(action, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]):
            self.direction = Move.EQUATORIALPRIME
        elif np.array_equal(action, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]):
            self.direction = Move.EQUATORIALTWO
        elif np.array_equal(action, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]):
            self.direction = Move.STANDING
        elif np.array_equal(action, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]):
            self.direction = Move.STANDINGPRIME
        elif np.array_equal(action, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]):
            self.direction = Move.STANDINGTWO
        elif np.array_equal(action, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]):
            self.direction = Move.LEFTWIDE
        elif np.array_equal(action, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]):
            self.direction = Move.LEFTWIDEPRIME
        elif np.array_equal(action, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]):
            self.direction = Move.LEFTWIDETWO
        elif np.array_equal(action, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]):
            self.direction = Move.RIGHTWIDE
        elif np.array_equal(action, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]):
            self.direction = Move.RIGHTWIDEPRIME
        elif np.array_equal(action, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]):
            self.direction = Move.RIGHTWIDETWO
        elif np.array_equal(action, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]):
            self.direction = Move.UPWIDE
        elif np.array_equal(action, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]):
            self.direction = Move.UPWIDEPRIME
        elif np.array_equal(action, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]):
            self.direction = Move.UPWIDETWO
        elif np.array_equal(action, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]):
            self.direction = Move.DOWNWIDE
        elif np.array_equal(action, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]):
            self.direction = Move.DOWNWIDEPRIME
        elif np.array_equal(action, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]):
            self.direction = Move.DOWNWIDETWO
        elif np.array_equal(action, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]):
            self.direction = Move.FRONTWIDE
        elif np.array_equal(action, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]):
            self.direction = Move.FRONTWIDEPRIME
        elif np.array_equal(action, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]):
            self.direction = Move.FRONTWIDETWO
        elif np.array_equal(action, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]):
            self.direction = Move.BACKWIDE
        elif np.array_equal(action, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]):
            self.direction = Move.BACKWIDEPRIME
        elif np.array_equal(action, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]):
            self.direction = Move.BACKWIDETWO
        elif np.array_equal(action, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]):
            self.direction = Move.NONE

        if self.direction == Move.LEFT:
            self.cube.left_turn()
            self.direction = Move.NONE
        elif self.direction == Move.LEFTPRIME:
            self.cube.left_turn();
            self.cube.left_turn();
            self.cube.left_turn();
            self.direction = Move.NONE

        elif self.direction == Move.LEFTTWO:
            self.cube.left_turn();
            self.cube.left_turn();
            self.direction = Move.NONE

        elif self.direction == Move.RIGHT:
            self.cube.right_turn()
            self.direction = Move.NONE

        elif self.direction == Move.RIGHTPRIME:
            self.cube.right_turn();
            self.cube.right_turn();
            self.cube.right_turn();
            self.direction = Move.NONE

        elif self.direction == Move.RIGHTTWO:
            self.cube.right_turn();
            self.cube.right_turn();
            self.direction = Move.NONE

        elif self.direction == Move.UP:
            self.cube.up_turn()
            self.direction = Move.NONE

        elif self.direction == Move.UPPRIME:
            self.cube.up_turn();
            self.cube.up_turn();
            self.cube.up_turn();
            self.direction = Move.NONE

        elif self.direction == Move.UPTWO:
            self.cube.up_turn();
            self.cube.up_turn();
            self.direction = Move.NONE

        elif self.direction == Move.DOWN:
            self.cube.down_turn()
            self.direction = Move.NONE

        elif self.direction == Move.DOWNPRIME:
            self.cube.down_turn();
            self.cube.down_turn();
            self.cube.down_turn();
            self.direction = Move.NONE

        elif self.direction == Move.DOWNTWO:
            self.cube.down_turn();
            self.cube.down_turn();
            self.direction = Move.NONE

        elif self.direction == Move.FRONT:
            self.cube.front_turn()
            self.direction = Move.NONE

        elif self.direction == Move.FRONTTWO:
            self.cube.front_turn();
            self.cube.front_turn();
            self.direction = Move.NONE

        elif self.direction == Move.FRONPRIME:
            self.cube.front_turn();
            self.cube.front_turn();
            self.cube.front_turn();
            self.direction = Move.NONE

        elif self.direction == Move.BACK:
            self.cube.back_turn()
            self.direction = Move.NONE

        elif self.direction == Move.BACKTWO:
            self.cube.back_turn();
            self.cube.back_turn();
            self.direction = Move.NONE

        elif self.direction == Move.BACKPRIME:
            self.cube.back_turn();
            self.cube.back_turn();
            self.cube.back_turn();
            self.direction = Move.NONE

        elif self.direction == Move.X:
            self.cube.x_rotation()
            self.direction = Move.NONE

        elif self.direction == Move.XPRIME:
            self.cube.x_rotation();
            self.cube.x_rotation();
            self.cube.x_rotation();
            self.direction = Move.NONE

        elif self.direction == Move.Y:
            self.cube.y_rotation()
            self.direction = Move.NONE

        elif self.direction == Move.YPRIME:
            self.cube.y_rotation();
            self.cube.y_rotation();
            self.cube.y_rotation();
            self.direction = Move.NONE

        elif self.direction == Move.Z:
            self.cube.z_rotation()
            self.direction = Move.NONE

        elif self.direction == Move.ZPRIME:
            self.cube.z_rotation();
            self.cube.z_rotation();
            self.cube.z_rotation();
            self.direction = Move.NONE

        elif self.direction == Move.X2:
            self.cube.x_rotation();
            self.cube.x_rotation();
            self.direction = Move.NONE

        elif self.direction == Move.Y2:
            self.cube.y_rotation();
            self.cube.y_rotation();
            self.direction = Move.NONE

        elif self.direction == Move.Z2:
            self.cube.z_rotation();
            self.cube.z_rotation();
            self.direction = Move.NONE

        elif self.direction == Move.MIDDLE:
            self.cube.middle_turn()
            self.direction = Move.NONE

        elif self.direction == Move.MIDDLEPRIME:
            self.cube.middle_turn();
            self.cube.middle_turn();
            self.cube.middle_turn();
            self.direction = Move.NONE

        elif self.direction == Move.MIDDLETWO:
            self.cube.middle_turn();
            self.cube.middle_turn();
            self.direction = Move.NONE

        elif self.direction == Move.EQUATORIAL:
            self.cube.equatorial_turn()
            self.direction = Move.NONE

        elif self.direction == Move.EQUATORIALPRIME:
            self.cube.equatorial_turn();
            self.cube.equatorial_turn();
            self.cube.equatorial_turn();
            self.direction = Move.NONE

        elif self.direction == Move.EQUATORIALTWO:
            self.cube.equatorial_turn();
            self.cube.equatorial_turn();
            self.direction = Move.NONE

        elif self.direction == Move.STANDING:
            self.cube.standing_turn()
            self.direction = Move.NONE

        elif self.direction == Move.STANDINGPRIME:
            self.cube.standing_turn();
            self.cube.standing_turn();
            self.cube.standing_turn();
            self.direction = Move.NONE

        elif self.direction == Move.STANDINGTWO:
            self.cube.standing_turn();
            self.cube.standing_turn();
            self.direction = Move.NONE

        elif self.direction == Move.LEFTWIDE:
            self.cube.l_small_turn()
            self.direction = Move.NONE

        elif self.direction == Move.LEFTWIDEPRIME:
            self.cube.l_small_turn();
            self.cube.l_small_turn();
            self.cube.l_small_turn();
            self.direction = Move.NONE

        elif self.direction == Move.LEFTWIDETWO:
            self.cube.l_small_turn();
            self.cube.l_small_turn();
            self.direction = Move.NONE

        elif self.direction == Move.RIGHTWIDE:
            self.cube.r_small_turn()
            self.direction = Move.NONE

        elif self.direction == Move.RIGHTWIDEPRIME:
            self.cube.r_small_turn();
            self.cube.r_small_turn();
            self.cube.r_small_turn();
            self.direction = Move.NONE

        elif self.direction == Move.RIGHTWIDETWO:
            self.cube.r_small_turn();
            self.cube.r_small_turn();
            self.direction = Move.NONE

        elif self.direction == Move.UPWIDE:
            self.cube.u_small_turn()
            self.direction = Move.NONE

        elif self.direction == Move.UPWIDEPRIME:
            self.cube.u_small_turn();
            self.cube.u_small_turn();
            self.cube.u_small_turn();
            self.direction = Move.NONE

        elif self.direction == Move.UPWIDETWO:
            self.cube.u_small_turn();
            self.cube.u_small_turn();
            self.direction = Move.NONE

        elif self.direction == Move.DOWNWIDE:
            self.cube.d_small_turn()
            self.direction = Move.NONE

        elif self.direction == Move.DOWNWIDEPRIME:
            self.cube.d_small_turn();
            self.cube.d_small_turn();
            self.cube.d_small_turn();
            self.direction = Move.NONE

        elif self.direction == Move.DOWNWIDETWO:
            self.cube.d_small_turn();
            self.cube.d_small_turn();
            self.direction = Move.NONE

        elif self.direction == Move.FRONTWIDE:
            self.cube.f_small_turn()
            self.direction = Move.NONE

        elif self.direction == Move.FRONTWIDEPRIME:
            self.cube.f_small_turn();
            self.cube.f_small_turn();
            self.cube.f_small_turn();
            self.direction = Move.NONE

        elif self.direction == Move.FRONTWIDETWO:
            self.cube.f_small_turn();
            self.cube.f_small_turn();
            self.direction = Move.NONE

        elif self.direction == Move.BACKWIDE:
            self.cube.b_small_turn()
            self.direction = Move.NONE

        elif self.direction == Move.BACKWIDEPRIME:
            self.cube.b_small_turn();
            self.cube.b_small_turn();
            self.cube.b_small_turn();
            self.direction = Move.NONE

        elif self.direction == Move.BACKWIDETWO:
            self.cube.b_small_turn();
            self.cube.b_small_turn();
            self.direction = Move.NONE

        elif self.direction == Move.NONE:
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
