file = open('valid.txt', 'r')

for line in file:
    if (line.find("Scramble:") != -1):
        scramble = line.split(":")[1].strip()

        with open('scramble.txt', 'a') as f:
            f.write(scramble + "\n")

    if (line.find("//inspection") != -1):
        inspection = line.split("//")[0].strip()
        inspection = inspection.split(":")[1].strip()

        with open('scramble.txt', 'a') as f:
            f.write(inspection + "\n")

    if (line.find("//cross") != -1):
        cross = line.split("//")[0].strip()

        if (cross.find("Solution:") != -1):
            cross = cross.split(":")[1].strip()

        with open('scramble.txt', 'a') as f:
            f.write(cross + "\n")
            f.write("\n")

