# Level 1: In a Pinch  (COLONY: ONE)
Task:            Play exactly TWO actions (as CYAN) on an 8x8 board so that CYAN's Kernel ends on a tile
                 that is safe from MAGENTA's Kernel and Ping. Clear 3 scenes.
Inputs:          No zip yet. Only level-1.pdf (story + rules). The scenes are probably interactive on the platform.
Output format:   UNKNOWN: likely moves played in the web UI, or a text answer per scene.

## Rules (from slide 5)
- 8x8. Teams CYAN (us) and MAGENTA. K = Kernel, P = Ping.
- A turn is exactly 2 actions by one team. CYAN starts. Same piece may act twice. Can't skip.
- Kernel: per action, moves 1 step to any of the 8 neighbours; an enemy there is captured.
  Losing the Kernel loses immediately, and capturing the enemy Kernel wins immediately.
- Ping move: up to 2 tiles orthogonally, empty tiles only, no jumping.
- Ping capture: the first piece on a diagonal within 2 tiles, if it is an enemy, is captured
  (the Ping moves onto it). No orthogonal capture, no diagonal move onto an empty tile.

## Open questions
- Does "safe" mean safe from Magenta's whole next turn (2 actions) or from a single action?
  The solver reports both; the strict (2-action) answer satisfies either reading.
- Board/answer format on the platform (coordinates? a1-style? row,col?).
- The action list says "moving or shooting", but the Ping capture is described as moving onto the target.
