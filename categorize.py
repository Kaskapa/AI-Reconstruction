from rubiks_cube import Cube
from datetime import datetime
import copy

def fillOld(which):
    with open(f"SOLUTIONS/every_possible_solution{which}.txt", 'r') as f:
        i = 1
        for line in f:
            line = line.replace("\n", "")
            lineArr = line.split(" ")
            lineArr.reverse();
            cube = Cube(1)

            for move in lineArr:
                cube = do_alternative_moves(cube, move)

            if(i == 48790):
                saveCube = [];
                sum = 0
                print(lineArr)
                j = 0
                for face in cube.cube:
                    for row in face:
                        if 0 in row:
                            if j >= 0 and j <= 2:
                                saveCube.append(1)
                            elif j == 3 or j == 6 or j == 9 or j == 12:
                                saveCube.append(2)
                            elif j == 4 or j == 7 or j == 10 or j == 13:
                                saveCube.append(3)
                            elif j == 5 or j == 8 or j == 11 or j == 14:
                                saveCube.append(4)
                            elif j >= 15 and j <= 17:
                                saveCube.append(4)
                        j+=1
                saveCube.sort()
                print(saveCube)

            saveCube2 = []
            if i > 48790:
                j = 0
                for face in cube.cube:
                    for row in face:
                        if 0 in row:
                            if j >= 0 and j <= 2:
                                saveCube2.append(1)
                            elif j == 3 or j == 6 or j == 9 or j == 12:
                                saveCube2.append(2)
                            elif j == 4 or j == 7 or j == 10 or j == 13:
                                saveCube2.append(3)
                            elif j == 5 or j == 8 or j == 11 or j == 14:
                                saveCube2.append(4)
                            elif j >= 15 and j <= 17:
                                saveCube2.append(4)
                        j+=1
                saveCube2.sort()

            if  i > 48790 and saveCube2 == saveCube:
                sum+=1

            i += 1
    print(sum)



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
fillOld(1);
end = datetime.now()
howLOng = end-now
print("Done!")
print(howLOng)