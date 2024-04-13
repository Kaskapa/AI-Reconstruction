import copy


def findAllCombinations(arr, r):
    ALTERNATING = [["L", "R"], ["U", "D"], ["F", "B"]]
    combList = []
    paths = []

    for i in arr:
        list = []
        list.append(i)
        paths.append(list)
        list = []

    while paths:
        path = paths.pop(0)

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

            if(len(path) != r):
                newPath = copy.deepcopy(path)
                newPath.append(i)

                if(newPath in combList):
                    newPath = []
                    continue

                paths.append(newPath)
                newPath = []


MOVES = ["L", "L'", "L2", "R", "R'", "R2", "U", "U'", "U2", "D", "D'", "D2", "F", "F'", "F2", "B", "B'", "B2"]

findAllCombinations(MOVES, 3)