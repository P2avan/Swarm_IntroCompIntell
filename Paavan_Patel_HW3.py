# Dolphins vs Fish (Predator–Prey) — minimal deps (match MJ.py)
# --------------------------------------------------------------
# This remake uses ONLY the libraries MJ.py uses:
#   - random
#   - numpy
#   - matplotlib.pyplot
#   - matplotlib.animation.FuncAnimation
# No dataclasses/typing/colors/etc.

import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ------------------ Parameters ------------------ #
GRID_SIZE = 80
INITIAL_FISH = 100
INITIAL_DOLPHINS = 15

FISH_REPRODUCTION_PROB = 0.0175   # chance per step to spawn into a neighbor
FISH_MOVE_PROB = 0.5            # chance a fish attempts a move

DOLPHIN_VISION_RADIUS = 5       # how far dolphins "see" fish (Chebyshev)
DOLPHIN_MOVE_BIAS = 0.85        # prob to bias step toward nearest seen fish
DOLPHIN_STEP_COST = 1           # energy spent each step
DOLPHIN_EAT_GAIN = 4           # energy from eating one fish
DOLPHIN_REPRODUCTION_THRESHOLD = 4
DOLPHIN_REPRODUCTION_COST = 12

STEPS = 2000
VISUALIZE_EVERY = 2
RANDOM_SEED = 0  # set None for full randomness

# Colors (RGB floats)
FISH_COLOR = (0.2, 0.6, 1.0)    # light blue
DOLPHIN_COLOR = (0.0, 0.0, 0.0) # black
EMPTY_COLOR = (1.0, 1.0, 1.0)   # white

# ------------------ Utilities ------------------ #

def torus(x, n):
    return x % n

# Movement steps (8-neighborhood)
MOVE_STEPS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1)
]

# ------------------ World init ------------------ #

def init_world(size):
    """Return (fish_grid, dolphins, dolphin_grid).
    fish_grid: bool array (size,size)
    dolphins: list of dicts {x,y,energy}
    dolphin_grid: bool occupancy grid
    """
    fish_grid = np.zeros((size, size), dtype=bool)
    dolphin_grid = np.zeros((size, size), dtype=bool)

    # place fish
    cells = [(i, j) for i in range(size) for j in range(size)]
    random.shuffle(cells)
    for (i, j) in cells[:INITIAL_FISH]:
        fish_grid[i, j] = True

    # place dolphins on empty cells
    dolphins = []
    placed = 0
    k = 0
    while placed < INITIAL_DOLPHINS and k < len(cells):
        i, j = cells[k]
        k += 1
        if not fish_grid[i, j] and not dolphin_grid[i, j]:
            dolphin_grid[i, j] = True
            dolphins.append({"x": i, "y": j, "energy": DOLPHIN_EAT_GAIN})
            placed += 1

    return fish_grid, dolphins, dolphin_grid

# ------------------ Fish dynamics ------------------ #

def fish_step(fish_grid, dolphin_grid):
    size = fish_grid.shape[0]
    positions = list(zip(*np.where(fish_grid)))
    random.shuffle(positions)

    new_fish = fish_grid.copy()

    for (x, y) in positions:
        if not fish_grid[x, y]:
            continue  # moved already
        # try move
        if random.random() < FISH_MOVE_PROB:
            dx, dy = random.choice(MOVE_STEPS)
            nx, ny = torus(x + dx, size), torus(y + dy, size)
            if (not fish_grid[nx, ny]) and (not dolphin_grid[nx, ny]) and (not new_fish[nx, ny]):
                new_fish[x, y] = False
                new_fish[nx, ny] = True
                x, y = nx, ny  # update local pos for reproduction
        # try reproduction into a random neighbor
        if random.random() < FISH_REPRODUCTION_PROB:
            dx, dy = random.choice(MOVE_STEPS)
            rx, ry = torus(x + dx, size), torus(y + dy, size)
            if (not new_fish[rx, ry]) and (not dolphin_grid[rx, ry]):
                new_fish[rx, ry] = True

    fish_grid[:, :] = new_fish

# ------------------ Dolphin dynamics ------------------ #

def nearest_fish_direction(fish_grid, x, y, R):
    """Return (dx, dy) unit step toward nearest fish within Chebyshev radius R, or None."""
    size = fish_grid.shape[0]
    best = None
    best_d = None
    for dx in range(-R, R + 1):
        for dy in range(-R, R + 1):
            if dx == 0 and dy == 0:
                continue
            nx, ny = torus(x + dx, size), torus(y + dy, size)
            if fish_grid[nx, ny]:
                d = max(abs(dx), abs(dy))
                if best_d is None or d < best_d:
                    best_d = d
                    sx = 0 if dx == 0 else (1 if dx > 0 else -1)
                    sy = 0 if dy == 0 else (1 if dy > 0 else -1)
                    best = (sx, sy)
    return best


def dolphins_step(fish_grid, dolphins, dolphin_grid):
    size = fish_grid.shape[0]
    order = list(range(len(dolphins)))
    random.shuffle(order)

    new_grid = np.zeros_like(dolphin_grid)
    new_dolphins = []

    for idx in order:
        d = dolphins[idx]
        x, y, e = d["x"], d["y"], d["energy"]

        # choose movement
        step = None
        if random.random() < DOLPHIN_MOVE_BIAS:
            step = nearest_fish_direction(fish_grid, x, y, DOLPHIN_VISION_RADIUS)
        if step is None:
            step = random.choice(MOVE_STEPS)
        nx, ny = torus(x + step[0], size), torus(y + step[1], size)

        # collision avoid: if already taken this tick, try one random fallback
        if new_grid[nx, ny]:
            fx, fy = nx, ny
            step = random.choice(MOVE_STEPS)
            nx, ny = torus(x + step[0], size), torus(y + step[1], size)
            if new_grid[nx, ny]:
                nx, ny = x, y

        x, y = nx, ny
        e -= DOLPHIN_STEP_COST

        # eat
        if fish_grid[x, y]:
            fish_grid[x, y] = False
            e += DOLPHIN_EAT_GAIN

        # death
        if e <= 0:
            continue

        # reproduction if energetic and a neighbor cell available (w.r.t. dolphins only)
        if e >= DOLPHIN_REPRODUCTION_THRESHOLD:
            steps = MOVE_STEPS[:]
            random.shuffle(steps)
            for dx, dy in steps:
                bx, by = torus(x + dx, size), torus(y + dy, size)
                if not new_grid[bx, by] and not dolphin_grid[bx, by]:
                    calf_energy = max(4, DOLPHIN_REPRODUCTION_COST // 2)
                    new_dolphins.append({"x": bx, "y": by, "energy": calf_energy})
                    new_grid[bx, by] = True
                    e -= DOLPHIN_REPRODUCTION_COST
                    break

        # occupy & keep
        new_grid[x, y] = True
        new_dolphins.append({"x": x, "y": y, "energy": e})

    dolphin_grid[:, :] = new_grid
    dolphins[:] = new_dolphins

# ------------------ Rendering ------------------ #

def render_rgb(fish_grid, dolphin_grid):
    h, w = fish_grid.shape
    img = np.ones((h, w, 3), dtype=float)
    fish_mask = fish_grid & (~dolphin_grid)
    img[fish_mask] = FISH_COLOR
    img[dolphin_grid] = DOLPHIN_COLOR
    return img

# ------------------ Main / Animation ------------------ #

def main():
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

    fish_grid, dolphins, dolphin_grid = init_world(GRID_SIZE)

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.set_axis_off()

    img = ax.imshow(render_rgb(fish_grid, dolphin_grid), interpolation="nearest", animated=True)

    ticks = {"t": 0}

    def step(_):
        for _ in range(VISUALIZE_EVERY):
            fish_step(fish_grid, dolphin_grid)
            dolphins_step(fish_grid, dolphins, dolphin_grid)
            ticks["t"] += 1
        img.set_data(render_rgb(fish_grid, dolphin_grid))
        ax.set_title(
            f"Dolphins vs Fish — step {ticks['t']} | fish: {int(fish_grid.sum())} | dolphins: {len(dolphins)}"
        )
        return (img,)

    anim = FuncAnimation(fig, step, frames=STEPS // VISUALIZE_EVERY, interval=30, blit=True)
    plt.show()


if __name__ == "__main__":
    main()
