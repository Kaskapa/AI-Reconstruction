from givemecrossUsable import CrossSolver
from rubiks_cube import Cube
import kociemba as kc

scramble = "F' U' R' B' L U' B L' D2 F' R' B' D F U' L' F D' L' U' B R2 U F2 B"
crossSolver = CrossSolver(scramble)
crossSolver.solve()
print(crossSolver.allSol)

solution = kc.solve("UULUUUDURBRFRRFUBLLFDLFFFLFRDRBDRRBBLLFDLFBLDUBBDBRDDU")

print(solution)
