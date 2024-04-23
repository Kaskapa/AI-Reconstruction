def fillOld(which):
    with open(f"SOLUTIONS/every_possible_solution{which}.txt", 'r') as f:
        combList = [line.strip().split() for line in f]

    pathsMaxLength = len(combList[-1])-1

    paths = []

    for element in combList:
        if len(element) == pathsMaxLength:
            paths.append(element)

    return combList, paths
