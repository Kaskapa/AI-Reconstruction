file = open('c:/Users/krist/Documents/GitHub/AI-Reconstruction/some websraping stuff/valid.txt', 'r')

for line in file:
    if (line.find("Scramble:") != -1):
        scramble = line.split(":")[1].strip()

        with open('scramble.csv', 'a') as f:
            f.write(scramble + ";")

    if (line.find("//inspection") != -1):
        inspection = line.split("//")[0].strip()
        inspection = inspection.split(":")[1].strip()
        inspection = inspection.replace("(", "")
        inspection = inspection.replace(")", "")

        with open('scramble.csv', 'a') as f:
            f.write(inspection + ";")

    if (line.find("//cross") != -1):
        cross = line.split("//")[0].strip()
        cross = cross.replace("(", "")
        cross = cross.replace(")", "")

        if (cross.find("Solution:") != -1):
            cross = cross.split(":")[1].strip()

        with open('scramble.csv', 'a') as f:
            f.write(cross + ";")


    if (line.find("//1st pair") != -1):
        f2l = line.split("//")[0].strip()
        f2l = f2l.replace("(", "")
        f2l = f2l.replace(")", "")

        if (f2l.find("Solution:") != -1):
            f2l = f2l.split(":")[1].strip()

        with open('scramble.csv', 'a') as f:
            f.write(f2l)
            f.write("\n")