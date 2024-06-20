from rubiks_cube import Cube
import threading
import math
import copy
import os

scramble = input("Enter a scramble: ")

cube = Cube(1)
cubeCOPY = Cube(0)

scramblearr = scramble.split(" ")

for i in scramblearr:
    cube.do_moves(action=i)
    cubeCOPY.do_moves(action=i)

cube.cateogrize()

allSol = []

def findSol(min, max, cube, cubeCOPY, a):
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
                print(a + line)
                allSol.append(a + line)

if(os.path.isfile(f"CATEGORY2/category_{cube.cateogry}.txt")):
    with open(f"CATEGORY2/category_{cube.cateogry}.txt", "r") as f:
        linesLen = len(f.readlines())

        t1 = threading.Thread(target=findSol, args=(0, math.floor(linesLen/8), cube, cubeCOPY, ""))
        t2 = threading.Thread(target=findSol, args=(math.floor(linesLen/8), math.floor(linesLen/8) * 2, cube, cubeCOPY, "" ))
        t3 = threading.Thread(target=findSol, args=(math.floor(linesLen/8)*2, math.floor(linesLen/8) * 3, cube, cubeCOPY, ""))
        t4 = threading.Thread(target=findSol, args=(math.floor(linesLen/8)*3, math.floor(linesLen/8) * 4, cube, cubeCOPY, ""))
        t5 = threading.Thread(target=findSol, args=(math.floor(linesLen/8) * 4, math.floor(linesLen/8) * 5, cube, cubeCOPY, ""))
        t6 = threading.Thread(target=findSol, args=(math.floor(linesLen/8) * 5, math.floor(linesLen/8) * 6, cube, cubeCOPY, ""))
        t7 = threading.Thread(target=findSol, args=(math.floor(linesLen/8) * 6, math.floor(linesLen/8) * 7, cube, cubeCOPY, ""))
        t8 = threading.Thread(target=findSol, args=(math.floor(linesLen/8) * 7, linesLen, cube, cubeCOPY, ""))

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

if(len(allSol) == 0):
    arr = ["L", "L'", "L2", "R", "R'", "R2", "U", "U'", "U2", "D", "D'", "D2", "F", "F'", "F2", "B", "B'", "B2"]
    orgScramble = scramble
    for a in arr:
        scramble = orgScramble
        scramble = scramble + " " + a
        cube = Cube(1)
        cubeCOPY = Cube(0)

        scramblearr = scramble.split(" ")

        for i in scramblearr:
            cube.do_moves(action=i)
            cubeCOPY.do_moves(action=i)

        cube.cateogrize()

        if(os.path.isfile(f"CATEGORY2/category_{cube.cateogry}.txt")):
            with open(f"CATEGORY2/category_{cube.cateogry}.txt", "r") as f:
                linesLen = len(f.readlines())

                t1 = threading.Thread(target=findSol, args=(0, math.floor(linesLen/8), cube, cubeCOPY, a + " "))
                t2 = threading.Thread(target=findSol, args=(math.floor(linesLen/8), math.floor(linesLen/8) * 2, cube, cubeCOPY, a + " " ))
                t3 = threading.Thread(target=findSol, args=(math.floor(linesLen/8)*2, math.floor(linesLen/8) * 3, cube, cubeCOPY, a + " "))
                t4 = threading.Thread(target=findSol, args=(math.floor(linesLen/8)*3, math.floor(linesLen/8) * 4, cube, cubeCOPY, a + " "))
                t5 = threading.Thread(target=findSol, args=(math.floor(linesLen/8) * 4, math.floor(linesLen/8) * 5, cube, cubeCOPY, a + " "))
                t6 = threading.Thread(target=findSol, args=(math.floor(linesLen/8) * 5, math.floor(linesLen/8) * 6, cube, cubeCOPY, a + " "))
                t7 = threading.Thread(target=findSol, args=(math.floor(linesLen/8) * 6, math.floor(linesLen/8) * 7, cube, cubeCOPY, a + " "))
                t8 = threading.Thread(target=findSol, args=(math.floor(linesLen/8) * 7, linesLen, cube, cubeCOPY, a + " "))

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