import copy

class Cube:
    def __init__(self):
        self.cube = [
            [[0, 0, 0],
             [0, 0, 0],
             [0, 0, 0]],

            [[1, 1, 1],
             [1, 1, 1],
             [1, 1, 1]],

            [[2, 2, 2],
             [2, 2, 2],
             [2, 2, 2]],

            [[3, 3, 3],
             [3, 3, 3],
             [3, 3, 3]],

            [[4, 4, 4],
             [4, 4, 4],
             [4, 4, 4]],

            [[5, 5, 5],
             [5, 5, 5],
             [5, 5, 5]]
        ]

    def reset(self):
        self.cube = [
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            [[2, 2, 2], [2, 2, 2], [2, 2, 2]],
            [[3, 3, 3], [3, 3, 3], [3, 3, 3]],
            [[4, 4, 4], [4, 4, 4], [4, 4, 4]],
            [[5, 5, 5], [5, 5, 5], [5, 5, 5]]
        ]

    def is_up_cross_solved(self, color):
        return (self.cube[0][0][1] == color and
                self.cube[0][1][0] == color and
                self.cube[0][1][1] == color and
                self.cube[0][1][2] == color and
                self.cube[0][2][1] == color and
                self.cube[1][0][1] == self.cube[1][1][1] and
                self.cube[2][0][1] == self.cube[2][1][1] and
                self.cube[3][0][1] == self.cube[3][1][1] and
                self.cube[4][0][1] == self.cube[4][1][1])

    def is_left_cross_solved(self, color):
        return (self.cube[1][0][1] == color and
                self.cube[1][1][0] == color and
                self.cube[1][1][1] == color and
                self.cube[1][1][2] == color and
                self.cube[1][2][1] == color and
                self.cube[0][1][0] == self.cube[0][1][1] and
                self.cube[2][1][0] == self.cube[2][1][1] and
                self.cube[4][1][2] == self.cube[4][1][1] and
                self.cube[5][1][0] == self.cube[5][1][1])

    def is_front_cross_solved(self, color):
        return (self.cube[2][0][1] == color and
                self.cube[2][1][0] == color and
                self.cube[2][1][1] == color and
                self.cube[2][1][2] == color and
                self.cube[2][2][1] == color and
                self.cube[0][2][1] == self.cube[0][1][1] and
                self.cube[1][1][2] == self.cube[1][1][1] and
                self.cube[3][1][0] == self.cube[3][1][1] and
                self.cube[5][0][1] == self.cube[5][1][1])

    def is_right_cross_solved(self, color):
        return (self.cube[3][0][1] == color and
                self.cube[3][1][0] == color and
                self.cube[3][1][1] == color and
                self.cube[3][1][2] == color and
                self.cube[3][2][1] == color and
                self.cube[0][1][2] == self.cube[0][1][1] and
                self.cube[2][1][2] == self.cube[2][1][1] and
                self.cube[4][1][0] == self.cube[4][1][1] and
                self.cube[5][1][2] == self.cube[5][1][1])

    def is_back_cross_solved(self, color):
        return (self.cube[4][0][1] == color and
                self.cube[4][1][0] == color and
                self.cube[4][1][1] == color and
                self.cube[4][1][2] == color and
                self.cube[4][2][1] == color and
                self.cube[0][0][1] == self.cube[0][1][1] and
                self.cube[1][1][0] == self.cube[1][1][1] and
                self.cube[3][1][2] == self.cube[3][1][1] and
                self.cube[5][2][1] == self.cube[5][1][1])

    def is_down_cross_solved(self, color):
        return (self.cube[5][0][1] == color and
                self.cube[5][1][0] == color and
                self.cube[5][1][1] == color and
                self.cube[5][1][2] == color and
                self.cube[5][2][1] == color and
                self.cube[1][2][1] == self.cube[1][1][1] and
                self.cube[2][2][1] == self.cube[2][1][1] and
                self.cube[3][2][1] == self.cube[3][1][1] and
                self.cube[4][2][1] == self.cube[4][1][1])

    def is_white_cross_solved(self):
        for i in range(len(self.cube)):
            if self.cube[i][1][1] == 0:
                return self.switch(i)

    def is_orange_cross_solved(self):
        for i in range(len(self.cube)):
            if self.cube[i][1][1] == 1:
                return self.switch(i)

    def is_green_cross_solved(self):
        for i in range(len(self.cube)):
            if self.cube[i][1][1] == 2:
                return self.switch(i)

    def is_red_cross_solved(self):
        for i in range(len(self.cube)):
            if self.cube[i][1][1] == 3:
                return self.switch(i)

    def is_blue_cross_solved(self):
        for i in range(len(self.cube)):
            if self.cube[i][1][1] == 4:
                return self.switch(i)

    def is_yellow_cross_solved(self):
        for i in range(len(self.cube)):
            if self.cube[i][1][1] == 5:
                return self.switch(i)

    def switch(self, i):
        if i == 0:
            return self.is_up_cross_solved(i)
        elif i == 1:
            return self.is_left_cross_solved(i)
        elif i == 2:
            return self.is_front_cross_solved(i)
        elif i == 3:
            return self.is_right_cross_solved(i)
        elif i == 4:
            return self.is_back_cross_solved(i)
        elif i == 5:
            return self.is_down_cross_solved(i)

    def y_rotation(self):
        temp = [[0]*3 for _ in range(3)]
        for i in range(3):
            for j in range(3):
                temp[i][j] = copy.deepcopy(self.cube[1][i][j])
        for i in range(3):
            for j in range(3):
                self.cube[1][i][j] = copy.deepcopy(self.cube[2][i][j])
                self.cube[2][i][j] = copy.deepcopy(self.cube[3][i][j])
                self.cube[3][i][j] = copy.deepcopy(self.cube[4][i][j])
                self.cube[4][i][j] = copy.deepcopy(temp[i][j])

        temp = [[0]*3 for _ in range(3)]
        for i in range(3):
            for j in range(3):
                temp[i][j] = copy.deepcopy(self.cube[0][i][j])
        for i in range(3):
            for j in range(3):
                self.cube[0][i][j] = copy.deepcopy(temp[2-j][i])

        temp = [[0]*3 for _ in range(3)]
        for i in range(3):
            for j in range(3):
                temp[i][j] = copy.deepcopy(self.cube[5][i][j])
        for i in range(3):
            for j in range(3):
                self.cube[5][i][j] = copy.deepcopy(temp[j][2-i])

    def x_rotation(self):
        temp = [[0]*3 for _ in range(3)]
        for i in range(3):
            for j in range(3):
                temp[i][j] = copy.deepcopy(self.cube[0][i][j])
        for i in range(3):
            for j in range(3):
                self.cube[0][i][j] = copy.deepcopy(self.cube[2][i][j])
                self.cube[2][i][j] = copy.deepcopy(self.cube[5][i][j])
                self.cube[5][i][j] = copy.deepcopy(self.cube[4][2-i][2-j])
                self.cube[4][i][j] = copy.deepcopy(temp[2-i][2-j])

        temp = [[0]*3 for _ in range(3)]
        for i in range(3):
            for j in range(3):
                temp[i][j] = copy.deepcopy(self.cube[1][i][j])
        for i in range(3):
            for j in range(3):
                self.cube[1][i][j] = copy.deepcopy(temp[j][2-i])

        temp = [[0]*3 for _ in range(3)]
        for i in range(3):
            for j in range(3):
                temp[i][j] = copy.deepcopy(self.cube[3][i][j])
        for i in range(3):
            for j in range(3):
                self.cube[3][i][j] = copy.deepcopy(temp[2-j][i])


    def z_rotation(self):
        temp = [[0]*3 for _ in range(3)]
        for i in range(3):
            for j in range(3):
                temp[i][j] = copy.deepcopy(self.cube[0][i][j])

        temp0 = copy.deepcopy(self.cube[1])
        temp1 = copy.deepcopy(self.cube[5])
        temp2 = copy.deepcopy(self.cube[3])
        temp3 = copy.deepcopy(temp)

        for i in range(3):
            for j in range(3):

                self.cube[0][i][j] = temp0[2-j][i]
                self.cube[1][i][j] = temp1[2-j][i]
                self.cube[5][i][j] = temp2[2-j][i]
                self.cube[3][i][j] = temp3[2-j][i]

        temp = [[0]*3 for _ in range(3)]
        for i in range(3):
            for j in range(3):
                temp[i][j] = copy.deepcopy(self.cube[2][i][j])
        for i in range(3):
            for j in range(3):
                self.cube[2][i][j] = copy.deepcopy(temp[2-j][i])

        temp = [[0]*3 for _ in range(3)]
        for i in range(3):
            for j in range(3):
                temp[i][j] = copy.deepcopy(self.cube[4][i][j])
        for i in range(3):
            for j in range(3):
                self.cube[4][i][j] = copy.deepcopy(temp[j][2-i])

    def right_turn(self):
        temp = [[0]*3 for _ in range(4)]
        for i in range(3):
            temp[0][i] = copy.deepcopy(self.cube[5][i][2])
            temp[1][i] = copy.deepcopy(self.cube[2][i][2])
            temp[2][i] = copy.deepcopy(self.cube[0][2 - i][2])
            temp[3][i] = copy.deepcopy(self.cube[4][2 - i][0])
        for i in range(3):
            self.cube[2][i][2] = copy.deepcopy(temp[0][i])
            self.cube[0][i][2] = copy.deepcopy(temp[1][i])
            self.cube[4][i][0] = copy.deepcopy(temp[2][i])
            self.cube[5][i][2] = copy.deepcopy(temp[3][i])

        temp = [[0]*3 for _ in range(3)]
        for i in range(3):
            for j in range(3):
                temp[i][j] = copy.deepcopy(self.cube[3][i][j])
        for i in range(3):
            for j in range(3):
                self.cube[3][i][j] = copy.deepcopy(temp[2 - j][i])

    def left_turn(self):
        temp = [[0]*3 for _ in range(4)]
        for i in range(3):
            temp[0][i] = copy.deepcopy(self.cube[5][2-i][0])
            temp[1][i] = copy.deepcopy(self.cube[2][i][0])
            temp[2][i] = copy.deepcopy(self.cube[0][i][0])
            temp[3][i] = copy.deepcopy(self.cube[4][2- i][2])
        for i in range(3):
            self.cube[2][i][0] = copy.deepcopy(temp[2][i])
            self.cube[0][i][0] = copy.deepcopy(temp[3][i])
            self.cube[4][i][2] = copy.deepcopy(temp[0][i])
            self.cube[5][i][0] = copy.deepcopy(temp[1][i])

        temp = [[0]*3 for _ in range(3)]
        for i in range(3):
            for j in range(3):
                temp[i][j] = copy.deepcopy(self.cube[1][i][j])
        for i in range(3):
            for j in range(3):
                self.cube[1][i][j] = copy.deepcopy(temp[2-j][i])
    def up_turn(self):
        temp = [[0]*3 for _ in range(4)]
        for i in range(3):
            temp[0][i] = copy.deepcopy(self.cube[1][0][i])
            temp[1][i] = copy.deepcopy(self.cube[2][0][i])
            temp[2][i] = copy.deepcopy(self.cube[3][0][i])
            temp[3][i] = copy.deepcopy(self.cube[4][0][i])
        for i in range(3):
            self.cube[1][0][i] = copy.deepcopy(temp[1][i])
            self.cube[2][0][i] = copy.deepcopy(temp[2][i])
            self.cube[3][0][i] = copy.deepcopy(temp[3][i])
            self.cube[4][0][i] = copy.deepcopy(temp[0][i])

        temp = [[0]*3 for _ in range(3)]
        for i in range(3):
            for j in range(3):
                temp[i][j] = copy.deepcopy(self.cube[0][i][j])
        for i in range(3):
            for j in range(3):
                self.cube[0][i][j] = copy.deepcopy(temp[2-j][i])

    def down_turn(self):
        temp = [[0]*3 for _ in range(4)]
        for i in range(3):
            temp[0][i] = copy.deepcopy(self.cube[1][2][i])
            temp[1][i] = copy.deepcopy(self.cube[2][2][i])
            temp[2][i] = copy.deepcopy(self.cube[3][2][i])
            temp[3][i] = copy.deepcopy(self.cube[4][2][i])
        for i in range(3):
            self.cube[1][2][i] = copy.deepcopy(temp[3][i])
            self.cube[2][2][i] = copy.deepcopy(temp[0][i])
            self.cube[3][2][i] = copy.deepcopy(temp[1][i])
            self.cube[4][2][i] = copy.deepcopy(temp[2][i])

        temp = [[0]*3 for _ in range(3)]
        for i in range(3):
            for j in range(3):
                temp[i][j] = copy.deepcopy(self.cube[5][i][j])
        for i in range(3):
            for j in range(3):
                self.cube[5][i][j] = copy.deepcopy(temp[2-j][i])

    def front_turn(self):
        temp = [[0]*3 for _ in range(4)]
        for i in range(3):
            temp[0][i] = copy.deepcopy(self.cube[1][i][2])
            temp[1][i] = copy.deepcopy(self.cube[0][2][i])
            temp[2][i] = copy.deepcopy(self.cube[3][i][0])
            temp[3][i] = copy.deepcopy(self.cube[5][0][i])
        for i in range(3):
            self.cube[1][i][2] = copy.deepcopy(temp[3][i])
            self.cube[0][2][i] = copy.deepcopy(temp[0][2-i])
            self.cube[3][i][0] = copy.deepcopy(temp[1][i])
            self.cube[5][0][i] = copy.deepcopy(temp[2][2-i])

        temp = [[0]*3 for _ in range(3)]
        for i in range(3):
            for j in range(3):
                temp[i][j] = copy.deepcopy(self.cube[2][i][j])
        for i in range(3):
            for j in range(3):
                self.cube[2][i][j] = copy.deepcopy(temp[2-j][i])

    def back_turn(self):
        temp = [[0]*3 for _ in range(4)]
        for i in range(3):
            temp[0][i] = copy.deepcopy(self.cube[1][i][0])
            temp[1][i] = copy.deepcopy(self.cube[0][0][i])
            temp[2][i] = copy.deepcopy(self.cube[3][i][2])
            temp[3][i] = copy.deepcopy(self.cube[5][2][i])
        for i in range(3):
            self.cube[1][i][0] = copy.deepcopy(temp[1][2-i])
            self.cube[0][0][i] = copy.deepcopy(temp[2][i])
            self.cube[3][i][2] = copy.deepcopy(temp[3][2-i])
            self.cube[5][2][i] = copy.deepcopy(temp[0][i])

        temp = [[0]*3 for _ in range(3)]
        for i in range(3):
            for j in range(3):
                temp[i][j] = copy.deepcopy(self.cube[4][i][j])
        for i in range(3):
            for j in range(3):
                self.cube[4][i][j] = copy.deepcopy(temp[2-j][i])

