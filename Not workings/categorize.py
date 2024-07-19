from rubiks_cube import Cube
from datetime import datetime
import copy
import time
import threading


def fillOld(which):
    with open(f"SOLUTIONS/every_possible_solution{which}.txt", 'r') as f:
        i = 1
        for line in f:
            line = line.replace("\n", "")
            lineArr = line.split(" ")
            lineArr.reverse()
            cube = Cube(1)

            for move in lineArr:
                cube = do_alternative_moves(cube, move)

            saveCube = [0,0,0,0]
            for face in range(len(cube.cube)):
                for row in range(len(cube.cube[face])):
                    for i in range(len(cube.cube[face][row])):
                        if 0 == cube.cube[face][row][i]:
                            if face == 0:
                                if row == 0:
                                    saveCube[cube.cube[4][0][1] - 1] = 11
                                if row == 1 and i == 0:
                                    saveCube[cube.cube[1][0][1] - 1] = 10
                                if row == 1 and i == 2:
                                    saveCube[cube.cube[3][0][1] - 1] = 12
                                if row == 2:
                                    saveCube[cube.cube[2][0][1] - 1] = 13
                            if face == 1:
                                if row == 0:
                                    saveCube[cube.cube[0][1][0] - 1] = 21
                                if row == 1 and i == 0:
                                    saveCube[cube.cube[4][1][2] - 1] = 310
                                if row == 1 and i == 2:
                                    saveCube[cube.cube[2][1][0] - 1] = 312
                                if row == 2:
                                    saveCube[cube.cube[5][1][0] - 1] = 41
                            if face == 2:
                                if row == 0:
                                    saveCube[cube.cube[0][2][1] - 1] = 22
                                if row == 1 and i == 0:
                                    saveCube[cube.cube[1][1][2] - 1] = 320
                                if row == 1 and i == 2:
                                    saveCube[cube.cube[3][1][0] - 1] = 322
                                if row == 2:
                                    saveCube[cube.cube[5][0][1] - 1] = 42
                            if face == 3:
                                if row == 0:
                                    saveCube[cube.cube[0][1][2] - 1] = 23
                                if row == 1 and i == 0:
                                    saveCube[cube.cube[2][1][2] - 1] = 330
                                if row == 1 and i == 2:
                                    saveCube[cube.cube[4][1][0] - 1] = 332
                                if row == 2:
                                    saveCube[cube.cube[5][1][2] - 1] = 43
                            if face == 4:
                                if row == 0:
                                    saveCube[cube.cube[0][0][1] - 1] = 24
                                if row == 1 and i == 0:
                                    saveCube[cube.cube[3][1][2] - 1] = 340
                                if row == 1 and i == 2:
                                    saveCube[cube.cube[1][1][0] - 1] = 342
                                if row == 2:
                                    saveCube[cube.cube[5][2][1] - 1] = 44
                            if face == 5:
                                if row == 0:
                                    saveCube[cube.cube[2][2][1] - 1] = 51
                                if row == 1 and i == 0:
                                    saveCube[cube.cube[1][2][1] - 1] = 50
                                if row == 1 and i == 2:
                                    saveCube[cube.cube[3][2][1] - 1] = 52
                                if row == 2:
                                    saveCube[cube.cube[4][2][1] - 1] = 53
            
            with open(f"CATEGORY2/category_{saveCube}.txt", "a") as valid_file:
                valid_file.write(line+"\n")



def do_moves(cube, action):
    if(action == "L"):
        cube.left_turn()
    elif(action == "L'"):
        cube.left_turn()
        cube.left_turn()
        cube.left_turn()
    elif(action == "L2"):
        cube.left_turn()
        cube.left_turn()
    elif(action == "R"):
       cube.right_turn()
    elif(action == "R'"):
        cube.right_turn()
        cube.right_turn()
        cube.right_turn()
    elif(action == "R2"):
        cube.right_turn()
        cube.right_turn()
    elif(action == "U"):
       cube.up_turn()
    elif(action == "U'"):
        cube.up_turn()
        cube.up_turn()
        cube.up_turn()
    elif(action == "U2"):
        cube.up_turn()
        cube.up_turn()
    elif(action == "D"):
       cube.down_turn()
    elif(action == "D'"):
        cube.down_turn()
        cube.down_turn()
        cube.down_turn()
    elif(action == "D2"):
        cube.down_turn()
        cube.down_turn()
    elif(action == "F"):
       cube.front_turn()
    elif(action == "F'"):
        cube.front_turn()
        cube.front_turn()
        cube.front_turn()
    elif(action == "F2"):
        cube.front_turn()
        cube.front_turn()
    elif(action == "B"):
       cube.back_turn()
    elif(action == "B'"):
        cube.back_turn()
        cube.back_turn()
        cube.back_turn()
    elif(action == "B2"):
        cube.back_turn()
        cube.back_turn()
    return cube
def do_alternative_moves(cube, action):
    if(action == "L'"):
        cube.left_turn()
    elif(action == "L"):
        cube.left_turn()
        cube.left_turn()
        cube.left_turn()
    elif(action == "L2"):
        cube.left_turn()
        cube.left_turn()
    elif(action == "R'"):
       cube.right_turn()
    elif(action == "R"):
        cube.right_turn()
        cube.right_turn()
        cube.right_turn()
    elif(action == "R2"):
        cube.right_turn()
        cube.right_turn()
    elif(action == "U'"):
       cube.up_turn()
    elif(action == "U"):
        cube.up_turn()
        cube.up_turn()
        cube.up_turn()
    elif(action == "U2"):
        cube.up_turn()
        cube.up_turn()
    elif(action == "D'"):
       cube.down_turn()
    elif(action == "D"):
        cube.down_turn()
        cube.down_turn()
        cube.down_turn()
    elif(action == "D2"):
        cube.down_turn()
        cube.down_turn()
    elif(action == "F'"):
       cube.front_turn()
    elif(action == "F"):
        cube.front_turn()
        cube.front_turn()
        cube.front_turn()
    elif(action == "F2"):
        cube.front_turn()
        cube.front_turn()
    elif(action == "B'"):
       cube.back_turn()
    elif(action == "B"):
        cube.back_turn()
        cube.back_turn()
        cube.back_turn()
    elif(action == "B2"):
        cube.back_turn()
        cube.back_turn()
    return cube


now = datetime.now()
t1 = threading.Thread(target=fillOld, args=[1])
t2 = threading.Thread(target=fillOld, args=[2])
t3 = threading.Thread(target=fillOld, args=[3])
t4 = threading.Thread(target=fillOld, args=[4])
t5 = threading.Thread(target=fillOld, args=[5])
t6 = threading.Thread(target=fillOld, args=[6])
t7 = threading.Thread(target=fillOld, args=[7])
t8 = threading.Thread(target=fillOld, args=[8])
t9 = threading.Thread(target=fillOld, args=[9])
t10 = threading.Thread(target=fillOld, args=[10])
t11 = threading.Thread(target=fillOld, args=[11])
t12 = threading.Thread(target=fillOld, args=[12])
t13 = threading.Thread(target=fillOld, args=[13])
t14 = threading.Thread(target=fillOld, args=[14])
t15 = threading.Thread(target=fillOld, args=[15])
t16 = threading.Thread(target=fillOld, args=[16])
t17 = threading.Thread(target=fillOld, args=[17])
t18 = threading.Thread(target=fillOld, args=[18])


t1.start()
t2.start()
t3.start()
t4.start()
t5.start()
t6.start()
t7.start()
t8.start()
t9.start()
t10.start()
t11.start()
t12.start()
t13.start()
t14.start()
t15.start()
t16.start()
t17.start()
t18.start()

t1.join()
t2.join()
t3.join()
t4.join()
t5.join()
t6.join()
t7.join()
t8.join()
t9.join()
t10.join()
t11.join()
t12.join()
t13.join()
t14.join()
t15.join()
t16.join()
t17.join()
t18.join()

end = datetime.now()
howLOng = end-now
print("Done!")
print(howLOng)