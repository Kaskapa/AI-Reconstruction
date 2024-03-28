import pygame
from cube import Cube
import time

class Display:
    def __init__(self, cube):
        self.cube = cube
        self.colors = [
            (255, 255, 255),
            (255, 165, 0),
            (0, 255, 0),
            (255, 0, 0),
            (0, 0, 255),
            (255, 255, 0)
        ]
        pygame.init()
        self.running = True
        self.screen = pygame.display.set_mode([675, 700])
        self.screen.fill((0, 0, 0))


    def drawCube(self):
        for i in range(3):
            for j in range(3):
                pygame.draw.rect(self.screen, self.colors[self.cube.cube[0][i][j]], (j*50 + 185, i*50 + 25, 40, 40))

        for i in range(3):
            for j in range(3):
                pygame.draw.rect(self.screen, self.colors[self.cube.cube[1][i][j]], (j*50 + 25, i*50 + 185, 40, 40))

        for i in range(3):
            for j in range(3):
                pygame.draw.rect(self.screen, self.colors[self.cube.cube[2][i][j]], (j*50 + 185, i*50 + 185, 40, 40))

        for i in range(3):
            for j in range(3):
                pygame.draw.rect(self.screen, self.colors[self.cube.cube[3][i][j]], (j*50 + 345, i*50 + 185, 40, 40))

        for i in range(3):
            for j in range(3):
                pygame.draw.rect(self.screen, self.colors[self.cube.cube[4][i][j]], (j*50 + 505, i*50 + 185, 40, 40))

        for i in range(3):
            for j in range(3):
                pygame.draw.rect(self.screen, self.colors[self.cube.cube[5][i][j]], (j*50 + 185, i*50 + 345, 40, 40))


cube = Cube()
display = Display(cube)

while display.running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            display.running = False
    # Other code here
    display.drawCube()
    pygame.display.flip()


pygame.quit()
