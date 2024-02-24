file = open("scramble_solution.txt", "r")

for i in range (1, 48341):
    line = file.readline()

    if(line.find("Scramble: ") != -1):
        scramble = line
        scramble = scramble.strip()
        print(scramble)
    else:
        continue

    line = file.readline()
    if(line.find("Solution: ") != -1):
        solution = line

        while line != "\n":
            line = file.readline()
            solution += line

        solution = solution.strip()
        print(solution)

    if(solution.find("//cross") != -1):
        with open("valid.txt", "a") as valid_file:
            valid_file.write(scramble + "\n")
            valid_file.write(solution + "\n\n")
