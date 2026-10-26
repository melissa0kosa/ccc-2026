import sys, time
sys.path.insert(0, "levels/level4")
from c1 import *
from colony_one.rules import legal_actions_for_team
from colony_one import constants
env = ColonyOneEnv()
print(env.render())
print(constants.SHRINK_ROUNDS, constants.MAGENTA_SETUP if hasattr(constants,'MAGENTA_SETUP') else '')
from colony_one.pieces import PieceType
print([(p.name, p.value, p.encode_index(), p.sacrifice_rank) for p in PieceType])
b = env.board
t=time.perf_counter()
for _ in range(1000): a = legal_actions_for_team(b, CYAN)
print("legal gen us", (time.perf_counter()-t)*1000, len(a))
t=time.perf_counter()
for _ in range(1000):
    for x in a:
        r = b.apply_action(x); b.undo_action(r)
print("apply/undo per action us", (time.perf_counter()-t)*1e6/1000/len(a))
t=time.perf_counter()
for _ in range(1000): encoding.encode_board(b, CYAN, 2, 1)
print("encode us", (time.perf_counter()-t)*1000)
print(a[0], a[0].__dict__ if hasattr(a[0],'__dict__') else '')
