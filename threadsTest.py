import copy
import threading
from datetime import datetime
from rubiks_cube import Cube
import csv
import time

def preprocess_data(file_path):
    with open(file_path, 'r') as file:
        csvreader = csv.reader(file)
        stateStr = []
        for row in csvreader:
            row_str = row[0]
            # Convert the scramble to a list of integers
            stateStr.append(row_str.split(";")[1])

        state = []

        for row in stateStr:
            cube = []
            faces = row.split("|")
            for face in faces:
                rows = face.split(" ")
                facesInt = []
                for row in rows:
                    elements = [*row]
                    desired_array = [int(numeric_string)-1 for numeric_string in elements]
                    facesInt.append(desired_array)
                cube.append(facesInt)
            state.append(cube)

    return state

states = preprocess_data("new_data/18to22/File(30).csv")

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
def path(paths, inWhich):
    cube = Cube()
    cube.cube = copy.deepcopy(states[0])
    r = 5
    arr = ["L", "L'", "L2", "R", "R'", "R2", "U", "U'", "U2", "D", "D'", "D2", "F", "F'", "F2", "B", "B'", "B2"]
    ALTERNATING = [["L", "R"], ["U", "D"], ["F", "B"]]
    combList = []

    while paths:
        path = paths.pop()

        for i in arr:
            shouldContinue = False
            for j in range(len(ALTERNATING)):
                if(len(path) > 0 and (path[len(path)-1][0] == i[0])):
                    shouldContinue = True
                    break
                elif(len(path) > 1 and (path[len(path)-1][0] == ALTERNATING[j][0][0] or path[len(path)-1][0] == ALTERNATING[j][1][0])
                   and (i[0] == ALTERNATING[j][0][0] or i[0] == ALTERNATING[j][1][0]) and
                   (path[len(path)-2][0] == ALTERNATING[j][0][0] or path[len(path)-2][0] == ALTERNATING[j][1][0])):
                    shouldContinue = True
                    break

            if(shouldContinue):
                continue

            if(path not in combList):
                combList.append(path)
                print(path)
                delimiter = " "
                result_string = delimiter.join(path)
                with open(f"SOLUTIONS/every_possible_solution{inWhich}.txt", "a") as valid_file:
                    valid_file.write(result_string+"\n")

            if(len(path) != r):
                newPath = copy.deepcopy(path)
                newPath.append(i)

                if(newPath in combList):
                    newPath = []
                    continue

                paths.append(newPath)
                newPath = []



if __name__ =="__main__":
    now = datetime.now()
    t1 = threading.Thread(target=path, args=([["L"]],1))
    t2 = threading.Thread(target=path, args=([["L'"]],2))
    t3 = threading.Thread(target=path, args=([["L2"]],3))
    t4 = threading.Thread(target=path, args=([["R"]],4))
    t5 = threading.Thread(target=path, args=([["R'"]],5))
    t6 = threading.Thread(target=path, args=([["R2"]],6))
    t7 = threading.Thread(target=path, args=([["U"]],7))
    t8 = threading.Thread(target=path, args=([["U'"]],8))
    t9 = threading.Thread(target=path, args=([["U2"]],9))
    t10 = threading.Thread(target=path, args=([["D"]],10))
    t11 = threading.Thread(target=path, args=([["D'"]],11))
    t12 = threading.Thread(target=path, args=([["D2"]],12))
    t13 = threading.Thread(target=path, args=([["F"]],13))
    t14 = threading.Thread(target=path, args=([["F'"]],14))
    t15 = threading.Thread(target=path, args=([["F2"]],15))
    t16 = threading.Thread(target=path, args=([["B"]],16))
    t17 = threading.Thread(target=path, args=([["B'"]],17))
    t18 = threading.Thread(target=path, args=([["B2"]],18))


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

    howLOng = end-now;
    print("Done!")
    print(howLOng)
