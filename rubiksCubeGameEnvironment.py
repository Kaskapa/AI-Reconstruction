import pygame
from cube import Cube

class RubiksCubeGameEnvironment:
    def __init__(self, screen):
        self.screen = screen
        self.cube = Cube()
        self.colors = {
            0: (255, 255, 255),
            5: (255, 255, 0),
            3: (255, 0, 0),
            1: (255, 165, 0),
            2: (0, 128, 0),
            4: (0, 0, 255)
        }

    def draw(self):
        self.screen.fill((0, 0, 0))
        self.drawCube()
        pygame.display.flip()

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

    def update(self):
        self.screen.fill((0, 0, 0))
        self.drawCube()

    def handleEvent(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_u:
                    if event.mod & pygame.KMOD_SHIFT:
                        self.cube.up_turn();
                        self.cube.up_turn();
                        self.cube.up_turn();
                    else:
                        self.cube.up_turn()
                if event.key == pygame.K_d:
                    if event.mod & pygame.KMOD_SHIFT:
                        self.cube.down_turn();
                        self.cube.down_turn();
                        self.cube.down_turn();
                    else:
                        self.cube.down_turn()
                if event.key == pygame.K_r:
                    if event.mod & pygame.KMOD_SHIFT:
                        self.cube.right_turn();
                        self.cube.right_turn();
                        self.cube.right_turn();
                    else:
                        self.cube.right_turn()
                if event.key == pygame.K_l:
                    if event.mod & pygame.KMOD_SHIFT:
                        self.cube.left_turn();
                        self.cube.left_turn();
                        self.cube.left_turn();
                    else:
                        self.cube.left_turn()
                if event.key == pygame.K_f:
                    if event.mod & pygame.KMOD_SHIFT:
                        self.cube.front_turn();
                        self.cube.front_turn();
                        self.cube.front_turn();
                    else:
                        self.cube.front_turn()
                if event.key == pygame.K_b:
                    if event.mod & pygame.KMOD_SHIFT:
                        self.cube.back_turn();
                        self.cube.back_turn();
                        self.cube.back_turn();
                    else:
                        self.cube.back_turn()
            self.update()

    def run(self):
        self.running = True
        while self.running:
            self.handleEvent()
            self.draw()
            pygame.display.flip()
        pygame.quit()

def main():
    clock = pygame.time.Clock()
    pygame.init()
    screen = pygame.display.set_mode([675, 700])
    game = RubiksCubeGameEnvironment(screen)
    game.run()

if __name__ == '__main__':
    main()


