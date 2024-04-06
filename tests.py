prevmoves_old = ["F", "D2", "D2", "R'", "U", "F", "L'", "D", "F2", "U2", "B'", "L2", "D'", "B'", "R2", "D", "B", "U2", "R", "B'"]
prevmoves = ["D", "U", "D2"]
MOVES = ["U","U2", "U'", "D", "D2", "D'","R","R2","R'","L","L2","L'","F","F2","F'","B","B2","B'"]
ALTERNATINGMOVES = [["R","L"], ["U","D"], ["F","B"]]


def _calculate_reward():
        total_reward = 0
        if(len(prevmoves) > 2 and prevmoves[-1][0] == prevmoves[-2][0] == prevmoves[-3][0]):
            total_reward -= 10

        elif(len(prevmoves) > 1 and prevmoves[-1][0] == prevmoves[-2][0]):
            total_reward -=10

        for moves in ALTERNATINGMOVES:
            if(len(prevmoves) > 2
               and ((prevmoves[-1][0] == moves[0]
                     and prevmoves[-2][0] == moves[1]
                     and prevmoves[-3][0] == moves[0])
                    or (prevmoves[-1][0] == moves[1]
                        and prevmoves[-2][0] == moves[0]
                        and prevmoves[-3][0] == moves[1]))):
                total_reward -= 10

        return total_reward

print (_calculate_reward())

