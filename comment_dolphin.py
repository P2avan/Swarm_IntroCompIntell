# Dolphins vs Fish (Predator–Prey) — minimal deps (match MJ.py)
# --------------------------------------------------------------
# PSEUDOCODE COMMENTS ADDED LINE-BY-LINE
# This remake uses ONLY the libraries MJ.py uses:
#   - random
#   - numpy
#   - matplotlib.pyplot
#   - matplotlib.animation.FuncAnimation
# No dataclasses/typing/colors/etc.

# import random number utilities
import random
# import numpy for arrays and fast math
import numpy as np
# import plotting library for visualization
import matplotlib.pyplot as plt
# import animation helper to update frames
from matplotlib.animation import FuncAnimation

# ------------------ Parameters ------------------ #
# size of the square world (GRID_SIZE x GRID_SIZE)
GRID_SIZE = 80
# starting count of fish (prey)
INITIAL_FISH = 1200
# starting count of dolphins (predators)
INITIAL_DOLPHINS = 25

# probability that a fish reproduces into a neighboring cell each step
FISH_REPRODUCTION_PROB = 0.02
# probability that a fish attempts to move each step
FISH_MOVE_PROB = 0.9

# how far (in Chebyshev distance) a dolphin can detect fish
DOLPHIN_VISION_RADIUS = 5
# probability a dolphin moves toward the seen fish instead of random
DOLPHIN_MOVE_BIAS = 0.85
# energy cost dolphins pay each step
DOLPHIN_STEP_COST = 1
# energy dolphins gain when they eat a fish
DOLPHIN_EAT_GAIN = 14
# energy level at/above which dolphins attempt to reproduce
DOLPHIN_REPRODUCTION_THRESHOLD = 24
# energy taken from parent when reproducing (child gets small starting energy)
DOLPHIN_REPRODUCTION_COST = 12

# total internal steps to simulate
STEPS = 2000
# number of internal steps per rendered animation frame
VISUALIZE_EVERY = 5
# random seed for reproducibility (set to None for full randomness)
RANDOM_SEED = 0

# Colors (RGB floats) used for rendering the grid
FISH_COLOR = (0.2, 0.6, 1.0)    # light blue for fish
DOLPHIN_COLOR = (0.0, 0.0, 0.0) # black for dolphins
EMPTY_COLOR = (1.0, 1.0, 1.0)   # white for empty cells

# ------------------ Utilities ------------------ #
# wrap coordinates on a torus (world edges connect)
def torus(x, n):
    # return x modulo n to keep within [0, n)
    return x % n

# define 8-directional movement options (no staying in place)
MOVE_STEPS = [
    (-1, -1), (-1, 0), (-1, 1),  # up-left, up, up-right
    (0, -1),           (0, 1),   # left,       right
    (1, -1),  (1, 0),  (1, 1)    # down-left, down, down-right
]

# ------------------ World init ------------------ #
# create initial fish and dolphin populations and grids
def init_world(size):
    """Return (fish_grid, dolphins, dolphin_grid).
    fish_grid: bool array (size,size) holding fish occupancy
    dolphins: list of dicts {x,y,energy} for dolphin agents
    dolphin_grid: bool array (size,size) holding dolphin occupancy
    """
    # create an empty fish occupancy grid (False everywhere)
    fish_grid = np.zeros((size, size), dtype=bool)
    # create an empty dolphin occupancy grid (False everywhere)
    dolphin_grid = np.zeros((size, size), dtype=bool)

    # build a list of all coordinates in the grid
    cells = [(i, j) for i in range(size) for j in range(size)]
    # randomize the order to place entities uniformly at random
    random.shuffle(cells)
    # place INITIAL_FISH fish at the first chosen coordinates
    for (i, j) in cells[:INITIAL_FISH]:
        # set fish present at that cell
        fish_grid[i, j] = True

    # prepare empty list to hold dolphin records
    dolphins = []
    # counter for dolphins placed so far
    placed = 0
    # index pointer for scanning candidate cells
    k = 0
    # loop until enough dolphins placed or cell list exhausted
    while placed < INITIAL_DOLPHINS and k < len(cells):
        # read candidate position
        i, j = cells[k]
        # advance to next candidate for next iteration
        k += 1
        # only place dolphin if cell has no fish and no dolphin
        if not fish_grid[i, j] and not dolphin_grid[i, j]:
            # mark dolphin present on occupancy grid
            dolphin_grid[i, j] = True
            # add a dolphin agent dict with initial energy
            dolphins.append({"x": i, "y": j, "energy": DOLPHIN_EAT_GAIN})
            # increment placed count
            placed += 1

    # return initialized grids and dolphin list
    return fish_grid, dolphins, dolphin_grid

# ------------------ Fish dynamics ------------------ #
# advance fish population by one step
def fish_step(fish_grid, dolphin_grid):
    # get grid size from array shape
    size = fish_grid.shape[0]
    # collect coordinates of all fish currently present
    positions = list(zip(*np.where(fish_grid)))
    # randomize order to avoid directional bias
    random.shuffle(positions)

    # start with a copy to apply moves and births safely
    new_fish = fish_grid.copy()

    # iterate through each fish position
    for (x, y) in positions:
        # if fish no longer at original cell (already moved), skip
        if not fish_grid[x, y]:
            continue
        # attempt random movement with probability FISH_MOVE_PROB
        if random.random() < FISH_MOVE_PROB:
            # choose a random neighboring step
            dx, dy = random.choice(MOVE_STEPS)
            # compute wrapped destination
            nx, ny = torus(x + dx, size), torus(y + dy, size)
            # only move if target cell has no fish and no dolphin, and not already filled this tick
            if (not fish_grid[nx, ny]) and (not dolphin_grid[nx, ny]) and (not new_fish[nx, ny]):
                # clear origin in new grid
                new_fish[x, y] = False
                # set destination in new grid
                new_fish[nx, ny] = True
                # update local position variables for potential reproduction
                x, y = nx, ny
        # attempt reproduction into a random neighbor with probability FISH_REPRODUCTION_PROB
        if random.random() < FISH_REPRODUCTION_PROB:
            # pick a random neighboring cell
            dx, dy = random.choice(MOVE_STEPS)
            # compute wrapped reproduction target
            rx, ry = torus(x + dx, size), torus(y + dy, size)
            # place offspring only if cell empty of fish and dolphin
            if (not new_fish[rx, ry]) and (not dolphin_grid[rx, ry]):
                # set fish at offspring location
                new_fish[rx, ry] = True

    # write back updated fish occupancy to main grid
    fish_grid[:, :] = new_fish

# ------------------ Dolphin dynamics ------------------ #
# find a unit movement vector toward the nearest fish within vision radius
def nearest_fish_direction(fish_grid, x, y, R):
    """Return (dx, dy) unit step toward nearest fish within Chebyshev radius R, or None."""
    # get grid size
    size = fish_grid.shape[0]
    # best step found so far (None means none yet)
    best = None
    # best Chebyshev distance found so far
    best_d = None
    # scan square of side (2R+1) centered on (x, y)
    for dx in range(-R, R + 1):
        for dy in range(-R, R + 1):
            # skip the origin cell
            if dx == 0 and dy == 0:
                continue
            # compute wrapped neighbor
            nx, ny = torus(x + dx, size), torus(y + dy, size)
            # check if a fish is at that neighbor
            if fish_grid[nx, ny]:
                # compute Chebyshev distance to neighbor
                d = max(abs(dx), abs(dy))
                # if this is the first fish or closer than previous best
                if best_d is None or d < best_d:
                    # update best distance
                    best_d = d
                    # compute unit step component in x toward target
                    sx = 0 if dx == 0 else (1 if dx > 0 else -1)
                    # compute unit step component in y toward target
                    sy = 0 if dy == 0 else (1 if dy > 0 else -1)
                    # store best step
                    best = (sx, sy)
    # return chosen step or None if no fish seen
    return best

# advance dolphin population by one step
def dolphins_step(fish_grid, dolphins, dolphin_grid):
    # grid size
    size = fish_grid.shape[0]
    # randomize processing order of dolphins to reduce bias
    order = list(range(len(dolphins)))
    random.shuffle(order)

    # start a fresh occupancy grid for dolphins for this tick
    new_grid = np.zeros_like(dolphin_grid)
    # container for next-step dolphin list
    new_dolphins = []

    # process each dolphin by randomized index
    for idx in order:
        # get dolphin state dict
        d = dolphins[idx]
        # unpack position and energy
        x, y, e = d["x"], d["y"], d["energy"]

        # default step is None (to be chosen)
        step = None
        # with bias probability, try to move toward nearest seen fish
        if random.random() < DOLPHIN_MOVE_BIAS:
            step = nearest_fish_direction(fish_grid, x, y, DOLPHIN_VISION_RADIUS)
        # if no fish seen or bias not taken, move randomly
        if step is None:
            step = random.choice(MOVE_STEPS)
        # compute wrapped destination for selected step
        nx, ny = torus(x + step[0], size), torus(y + step[1], size)

        # avoid two dolphins occupying same cell in this tick
        if new_grid[nx, ny]:
            # pick a random fallback move once
            step = random.choice(MOVE_STEPS)
            nx, ny = torus(x + step[0], size), torus(y + step[1], size)
            # if still colliding, stay in place
            if new_grid[nx, ny]:
                nx, ny = x, y

        # update position to destination
        x, y = nx, ny
        # pay step energy cost
        e -= DOLPHIN_STEP_COST

        # if a fish is at the new location, eat it
        if fish_grid[x, y]:
            # remove fish from grid
            fish_grid[x, y] = False
            # gain energy from food
            e += DOLPHIN_EAT_GAIN

        # if energy depleted, dolphin dies (skip adding to new list)
        if e <= 0:
            # skip to next dolphin
            continue

        # if sufficiently energetic, attempt reproduction into a neighboring free cell
        if e >= DOLPHIN_REPRODUCTION_THRESHOLD:
            # create a shuffled copy of neighbor steps to try positions
            steps = MOVE_STEPS[:]
            random.shuffle(steps)
            # iterate over neighbor candidates
            for dx, dy in steps:
                # compute wrapped baby cell position
                bx, by = torus(x + dx, size), torus(y + dy, size)
                # ensure no dolphin occupies baby cell (both new grid and previous grid)
                if not new_grid[bx, by] and not dolphin_grid[bx, by]:
                    # set small starting energy for calf (half cost or minimum 4)
                    calf_energy = max(4, DOLPHIN_REPRODUCTION_COST // 2)
                    # add new calf dolphin to next list
                    new_dolphins.append({"x": bx, "y": by, "energy": calf_energy})
                    # mark occupancy for calf
                    new_grid[bx, by] = True
                    # deduct reproduction cost from parent
                    e -= DOLPHIN_REPRODUCTION_COST
                    # stop after placing one calf
                    break

        # mark parent's final position as occupied
        new_grid[x, y] = True
        # append updated parent dolphin to the next list
        new_dolphins.append({"x": x, "y": y, "energy": e})

    # replace dolphin occupancy grid with the new one
    dolphin_grid[:, :] = new_grid
    # replace dolphin list with next-step dolphins
    dolphins[:] = new_dolphins

# ------------------ Rendering ------------------ #
# build an RGB image from fish and dolphin occupancy grids
def render_rgb(fish_grid, dolphin_grid):
    # extract grid dimensions
    h, w = fish_grid.shape
    # start with white image for all cells
    img = np.ones((h, w, 3), dtype=float)
    # compute mask for fish-only cells (fish present, dolphin absent)
    fish_mask = fish_grid & (~dolphin_grid)
    # color fish cells blue
    img[fish_mask] = FISH_COLOR
    # color dolphin cells black (drawn last to appear on top)
    img[dolphin_grid] = DOLPHIN_COLOR
    # return final RGB image array
    return img

# ------------------ Main / Animation ------------------ #
# driver function to run the simulation and animate it
def main():
    # set fixed seed if requested for reproducible runs
    if RANDOM_SEED is not None:
        # seed Python's random module
        random.seed(RANDOM_SEED)
        # seed NumPy's RNG
        np.random.seed(RANDOM_SEED)

    # create initial world state (fish grid, dolphin agents, dolphin grid)
    fish_grid, dolphins, dolphin_grid = init_world(GRID_SIZE)

    # set up plotting figure and axis
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    # hide axis decorations for cleaner look
    ax.set_axis_off()

    # draw initial frame as an image
    img = ax.imshow(render_rgb(fish_grid, dolphin_grid), interpolation="nearest", animated=True)

    # maintain a simple tick counter in a dict (mutable closure)
    ticks = {"t": 0}

    # define per-frame update function for FuncAnimation
    def step(_):
        # run multiple internal simulation steps per visual frame
        for _ in range(VISUALIZE_EVERY):
            # move and reproduce fish
            fish_step(fish_grid, dolphin_grid)
            # move, feed, reproduce, and possibly die dolphins
            dolphins_step(fish_grid, dolphins, dolphin_grid)
            # increment tick counter
            ticks["t"] += 1
        # update the image pixel data after internal steps
        img.set_data(render_rgb(fish_grid, dolphin_grid))
        # update plot title with current stats
        ax.set_title(
            f"Dolphins vs Fish — step {ticks['t']} | fish: {int(fish_grid.sum())} | dolphins: {len(dolphins)}"
        )
        # return updated artist for blitting
        return (img,)

    # build an animation object to repeatedly call step()
    anim = FuncAnimation(fig, step, frames=STEPS // VISUALIZE_EVERY, interval=30, blit=True)
    # display the animation window
    plt.show()

# run main if this file is executed as a script
if __name__ == "__main__":
    main()
