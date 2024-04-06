from rubiks_cube import Cube
import copy
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

states = preprocess_data("new_data/18to22/file(30).csv")

MOVES = ["L", "L'", "L2", "R", "R'", "R2", "U", "U'", "U2", "D", "D'", "D2", "F", "F'", "F2", "B", "B'", "B2"];
ALTERNATING = [["L", "R"], ["U", "D"], ["F", "B"]]
cube = Cube()
MAX_MOVES = 8

cube.cube = copy.deepcopy(states[0])

def performe_move(action):
    if(action == 0):
        cube.left_turn()
    elif(action == 1):
        cube.left_turn()
        cube.left_turn()
        cube.left_turn()
    elif(action == 2):
        cube.left_turn()
        cube.left_turn()
    elif(action == 3):
        cube.right_turn()
    elif(action == 4):
        cube.right_turn()
        cube.right_turn()
        cube.right_turn()
    elif(action == 5):
        cube.right_turn()
        cube.right_turn()
    elif(action == 6):
        cube.up_turn()
    elif(action == 7):
        cube.up_turn()
        cube.up_turn()
        cube.up_turn()
    elif(action == 8):
        cube.up_turn()
        cube.up_turn()
    elif(action == 9):
        cube.down_turn()
    elif(action == 10):
        cube.down_turn()
        cube.down_turn()
        cube.down_turn()
    elif(action == 11):
        cube.down_turn()
        cube.down_turn()
    elif(action == 12):
        cube.front_turn()
    elif(action == 13):
        cube.front_turn()
        cube.front_turn()
        cube.front_turn()
    elif(action == 14):
        cube.front_turn()
        cube.front_turn()
    elif(action == 15):
        cube.back_turn()
    elif(action == 16):
        cube.back_turn()
        cube.back_turn()
        cube.back_turn()
    elif(action == 17):
        cube.back_turn()
        cube.back_turn()

def backtrack(MOVES, res, tempList):
    res.append(tempList)
    print(tempList)
    if(len(tempList) != MAX_MOVES):
        for i in range(len(MOVES)):
            shouldContinue = False
            for j in range(len(ALTERNATING)):
                if(len(tempList) > 0 and (tempList[len(tempList)-1][0] == MOVES[i][0])):
                    shouldContinue = True
                    break
                elif(len(tempList) > 1 and (tempList[len(tempList)-1][0] == ALTERNATING[j][0][0] or tempList[len(tempList)-1][0] == ALTERNATING[j][1][0])
                   and (MOVES[i][0] == ALTERNATING[j][0][0] or MOVES[i][0] == ALTERNATING[j][1][0]) and
                   (tempList[len(tempList)-2][0] == ALTERNATING[j][0][0] or tempList[len(tempList)-2][0] == ALTERNATING[j][1][0])):
                    shouldContinue = True
                    break

            if(shouldContinue):
                continue

            tempList.append(MOVES[i])
            backtrack(MOVES, res, tempList)
            tempList.pop()

def permute():
    res = [];
    backtrack(MOVES, res, [])
    return res

end = permute();
