from rubiks_cube import Cube
import threading
import math
import copy
import os

class CrossSolver:
    def __init__(self, scramble):
        self.scramble = scramble
        self.cube = Cube(1)
        self.cubeCOPY = Cube(0)
        self.scramblearr = scramble.split(" ")
        self.allSol = []

        for i in self.scramblearr:
            self.cube.do_moves(action=i)
            self.cubeCOPY.do_moves(action=i)
        self.cube.cateogrize()

    def findSol(self, min, max, cube, cubeCOPY, a):
        cube.cateogrize()
        with open(f"CATEGORY2/category_{cube.cateogry}.txt", "r") as f:
            for i in range(min):
                next(f)
            for line in range(max):
                line = f.readline()
                cube = copy.deepcopy(cubeCOPY)

                lineArr = line.split(" ")

                for i in lineArr:
                    cube.do_moves(i)

                if(cube.is_white_cross_solved()):
                    solution = a+line
                    self.allSol.append(solution.replace("\n", ""))
                    break

    def solve(self):
        if(os.path.isfile(f"CATEGORY2/category_{self.cube.cateogry}.txt")):
            with open(f"CATEGORY2/category_{self.cube.cateogry}.txt", "r") as f:
                linesLen = len(f.readlines())

                t1 = threading.Thread(target=self.findSol, args=(0, math.floor(linesLen/8), self.cube, self.cubeCOPY, ""))
                t2 = threading.Thread(target=self.findSol, args=(math.floor(linesLen/8), math.floor(linesLen/8) * 2, self.cube, self.cubeCOPY, "" ))
                t3 = threading.Thread(target=self.findSol, args=(math.floor(linesLen/8)*2, math.floor(linesLen/8) * 3, self.cube, self.cubeCOPY, ""))
                t4 = threading.Thread(target=self.findSol, args=(math.floor(linesLen/8)*3, math.floor(linesLen/8) * 4, self.cube, self.cubeCOPY, ""))
                t5 = threading.Thread(target=self.findSol, args=(math.floor(linesLen/8) * 4, math.floor(linesLen/8) * 5, self.cube, self.cubeCOPY, ""))
                t6 = threading.Thread(target=self.findSol, args=(math.floor(linesLen/8) * 5, math.floor(linesLen/8) * 6, self.cube, self.cubeCOPY, ""))
                t7 = threading.Thread(target=self.findSol, args=(math.floor(linesLen/8) * 6, math.floor(linesLen/8) * 7, self.cube, self.cubeCOPY, ""))
                t8 = threading.Thread(target=self.findSol, args=(math.floor(linesLen/8) * 7, linesLen, self.cube, self.cubeCOPY, ""))

                t1.start()
                t2.start()
                t3.start()
                t4.start()
                t5.start()
                t6.start()
                t7.start()
                t8.start()

                t1.join()
                t2.join()
                t3.join()
                t4.join()
                t5.join()
                t6.join()
                t7.join()
                t8.join()

        if(len(self.allSol) == 0):
            arr = ["L", "L'", "L2", "R", "R'", "R2", "U", "U'", "U2", "D", "D'", "D2", "F", "F'", "F2", "B", "B'", "B2"]
            orgScramble = self.scramble
            for a in arr:
                self.scramble = orgScramble
                self.scramble = self.scramble + " " + a
                self.cube = Cube(1)
                self.cubeCOPY = Cube(0)

                self.scramblearr = self.scramble.split(" ")

                for i in self.scramblearr:
                    self.cube.do_moves(action=i)
                    self.cubeCOPY.do_moves(action=i)

                self.cube.cateogrize()

                if(os.path.isfile(f"CATEGORY2/category_{self.cube.cateogry}.txt")):
                    with open(f"CATEGORY2/category_{self.cube.cateogry}.txt", "r") as f:
                        linesLen = len(f.readlines())

                        t1 = threading.Thread(target=self.findSol, args=(0, math.floor(linesLen/8), self.cube, self.cubeCOPY, a + " "))
                        t2 = threading.Thread(target=self.findSol, args=(math.floor(linesLen/8), math.floor(linesLen/8) * 2, self.cube, self.cubeCOPY, a + " " ))
                        t3 = threading.Thread(target=self.findSol, args=(math.floor(linesLen/8)*2, math.floor(linesLen/8) * 3, self.cube, self.cubeCOPY, a + " "))
                        t4 = threading.Thread(target=self.findSol, args=(math.floor(linesLen/8)*3, math.floor(linesLen/8) * 4, self.cube, self.cubeCOPY, a + " "))
                        t5 = threading.Thread(target=self.findSol, args=(math.floor(linesLen/8) * 4, math.floor(linesLen/8) * 5, self.cube, self.cubeCOPY, a + " "))
                        t6 = threading.Thread(target=self.findSol, args=(math.floor(linesLen/8) * 5, math.floor(linesLen/8) * 6, self.cube, self.cubeCOPY, a + " "))
                        t7 = threading.Thread(target=self.findSol, args=(math.floor(linesLen/8) * 6, math.floor(linesLen/8) * 7, self.cube, self.cubeCOPY, a + " "))
                        t8 = threading.Thread(target=self.findSol, args=(math.floor(linesLen/8) * 7, linesLen, self.cube, self.cubeCOPY, a + " "))

                        t1.start()
                        t2.start()
                        t3.start()
                        t4.start()
                        t5.start()
                        t6.start()
                        t7.start()
                        t8.start()

                        t1.join()
                        t2.join()
                        t3.join()
                        t4.join()
                        t5.join()
                        t6.join()
                        t7.join()
                        t8.join()
