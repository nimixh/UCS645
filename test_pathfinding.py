"""
Multi-agent warehouse pathfinding with prioritized BFS.
Each step processes robots in random order — earlier robots get priority.
"""
import numpy as np
from collections import deque
import time, random

GRID_SIZE = 20
NUM_ROBOTS = 30
SIM_STEPS = 500

def is_obstacle(x, y):
    return (x % 4 == 0) and (y > 2) and (y < GRID_SIZE - 3)

def bfs_path(start, goal, blocked_cells):
    """BFS shortest path avoiding blocked cells."""
    if start == goal:
        return [start]
    q = deque([start])
    parent = {start: None}
    while q:
        cur = q.popleft()
        x, y = cur
        for dx, dy in [(0,1),(0,-1),(-1,0),(1,0)]:
            nx, ny = x+dx, y+dy
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                if is_obstacle(nx, ny):
                    continue
                nxt = (nx, ny)
                if nxt == goal:
                    path = [goal, cur]
                    while parent[path[-1]] is not None:
                        path.append(parent[path[-1]])
                    path.reverse()
                    return path
                if nxt not in parent and nxt not in blocked_cells:
                    parent[nxt] = cur
                    q.append(nxt)
    return None

# Setup
random.seed(42)
robots = {}
occupancy = {}  # (x,y) -> robot_id
for i in range(NUM_ROBOTS):
    while True:
        x = random.randint(0, GRID_SIZE-1)
        y = random.randint(0, GRID_SIZE-1)
        if not is_obstacle(x, y) and (x, y) not in occupancy:
            break
    while True:
        gx = random.randint(0, GRID_SIZE-1)
        gy = random.randint(0, GRID_SIZE-1)
        if not is_obstacle(gx, gy) and (gx, gy) != (x, y):
            break
    robots[i] = {'x': x, 'y': y, 'gx': gx, 'gy': gy, 'path': None}
    occupancy[(x, y)] = i

deliveries = 0
print("Step | Active | Deliveries")
print("-" * 30)

t0 = time.time()
for step in range(SIM_STEPS):
    # Process in random order each step (breaks deadlocks)
    order = list(robots.keys())
    random.shuffle(order)

    # Track which cells were moved INTO this step (cannot be occupied by others)
    reserved = set()

    active = 0
    for rid in order:
        r = robots[rid]
        if r['gx'] == -1:
            continue
        active += 1

        if (r['x'], r['y']) == (r['gx'], r['gy']):
            occupancy.pop((r['x'], r['y']), None)
            r['gx'] = r['gy'] = -1
            r['path'] = None
            deliveries += 1
            continue

        # Build blocked set: static obstacles + cells occupied by
        # robots that moved THIS step (reserved) or robots NOT YET processed.
        # NOT including robots ALREADY processed that didn't move — they stay put.
        blocked = set(occupancy.keys())
        blocked.discard((r['x'], r['y']))  # Self is not blocked

        if r['path'] is None:
            path = bfs_path((r['x'], r['y']), (r['gx'], r['gy']), blocked)
            if path is None or len(path) < 2:
                continue
            r['path'] = path

        # Try next step
        nx, ny = r['path'][1]
        if (nx, ny) not in occupancy and (nx, ny) not in reserved:
            occupancy.pop((r['x'], r['y']), None)
            r['x'], r['y'] = nx, ny
            occupancy[(nx, ny)] = rid
            reserved.add((nx, ny))
            r['path'] = r['path'][1:]
            if len(r['path']) <= 1:
                r['path'] = None
        else:
            r['path'] = None  # Replan next step

    if step % 50 == 0:
        print(f"  {step:3d} |   {active:3d}   |     {deliveries}")
    if active == 0:
        print(f"All done at step {step}!")
        break

t_total = time.time() - t0
print(f"\nDone in {t_total*1000:.1f} ms")
print(f"Deliveries: {deliveries}/{NUM_ROBOTS}")
