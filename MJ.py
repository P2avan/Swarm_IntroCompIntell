import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import random


# ---- Parameters ----
GRID_SIZE = 80            # grid is GRID_SIZE x GRID_SIZE
NUM_ITEMS = 1200          # total items placed randomly
NUM_TYPES = 3             # kinds/colors of items
NUM_AGENTS = 200          # number of termites
STEPS = 20000             # animation steps (or runtime steps)
NEIGHBORHOOD_RADIUS = 1   # R (neighborhood side is (2R+1))
k1 = 0.1                  # pickup constant (tuneable)
k2 = 0.3                  # drop constant (tuneable)
VISUALIZE_INTERVAL = 50   # draw every N steps


# ---- Helpers ----
def toroidal_index(x, size):
    return x % size

def neighborhood_positions(x, y, R, size):
    for dx in range(-R, R + 1):
        for dy in range(-R, R + 1):
            yield toroidal_index(x + dx, size), toroidal_index(y + dy, size)

def local_similarity(grid, x, y, R):
    """
    Compute fraction f of items in neighborhood similar to the item at (x,y).
    If cell at (x,y) is empty, we compute similarity to a hypothetical item passed in by agent,
    so this function expects grid value to be the type being compared or -1 for empty.
    """
    size = grid.shape[0]
    center_val = grid[x, y]
    if center_val == -1:
        # If empty, caller should provide the type to compare; but we handle empty separately in code below.
        return 0.0

    same = 0
    total = 0
    for nx, ny in neighborhood_positions(x, y, NEIGHBORHOOD_RADIUS, size):
        val = grid[nx, ny]
        if val != -1:
            total += 1
            if val == center_val:
                same += 1
    return (same / total) if total > 0 else 0.0


# ---- Initialize grid (items) ----
grid = -1 * np.ones((GRID_SIZE, GRID_SIZE), dtype=int)  # -1 means empty


# randomly place NUM_ITEMS items with types 0..NUM_TYPES-1
all_cells = [(i, j) for i in range(GRID_SIZE) for j in range(GRID_SIZE)]
random.shuffle(all_cells)
for idx in range(min(NUM_ITEMS, GRID_SIZE * GRID_SIZE)):
    x, y = all_cells[idx]
    grid[x, y] = random.randrange(NUM_TYPES)


# ---- Initialize agents ----
# Each agent: (x,y, carrying) where carrying is -1 for empty or type int
agents = []
empty_cells = [(i, j) for i in range(GRID_SIZE) for j in range(GRID_SIZE)]
random.shuffle(empty_cells)
for a in range(NUM_AGENTS):
    x = random.randrange(GRID_SIZE)
    y = random.randrange(GRID_SIZE)
    agents.append([x, y, -1])


# ---- Visualization setup ----
cmap = plt.get_cmap('tab10')
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xticks([])
ax.set_yticks([])
im = ax.imshow(np.zeros((GRID_SIZE, GRID_SIZE, 3)), interpolation='nearest')

def grid_to_rgb(grid):
    # map -1 to white, types to colors
    rgb = np.ones((GRID_SIZE, GRID_SIZE, 3))  # white background
    for t in range(NUM_TYPES):
        mask = (grid == t)
        color = cmap(t)[:3]   # tuple RGB
        rgb[mask] = color
    return rgb


# initial image
im.set_data(grid_to_rgb(grid))


# ---- Termite rules ----
def compute_f_for_type(grid, x, y, item_type):
    # compute fraction of same items in neighborhood given a hypothetical item_type
    size = grid.shape[0]
    same = 0
    total = 0
    for nx, ny in neighborhood_positions(x, y, NEIGHBORHOOD_RADIUS, size):
        val = grid[nx, ny]
        if val != -1:
            total += 1
            if val == item_type:
                same += 1
    return (same / total) if total > 0 else 0.0

def try_pickup(agent):
    x, y, carrying = agent
    val = grid[x, y]
    if val == -1:
        return False
    # compute local f around the agent for the item at cell
    f = compute_f_for_type(grid, x, y, val)
    p_pick = (k1 / (k1 + f)) ** 2
    if random.random() < p_pick:
        agent[2] = val     # pick up item
        grid[x, y] = -1    # cell becomes empty
        return True
    return False

def try_drop(agent):
    x, y, carrying = agent
    if carrying == -1:
        return False
    # compute f if we would drop carrying here
    f = compute_f_for_type(grid, x, y, carrying)
    p_drop = (f / (k2 + f)) ** 2
    if random.random() < p_drop and grid[x, y] == -1:
        grid[x, y] = carrying
        agent[2] = -1
        return True
    return False

def step_simulation():
    size = GRID_SIZE
    for agent in agents:
        # 1. random move: 8-neighbour or 4-neighbour; let's do Moore (8)
        dx = random.choice([-1, 0, 1])
        dy = random.choice([-1, 0, 1])
        # avoid standing still sometimes (optional)
        if dx == 0 and dy == 0:
            if random.random() < 0.5:
                dx = random.choice([-1, 1])
        agent[0] = toroidal_index(agent[0] + dx, size)
        agent[1] = toroidal_index(agent[1] + dy, size)

        x, y, carrying = agent

        if agent[2] == -1:
            # not carrying: try pickup (only if cell has item)
            if grid[x, y] != -1:
                try_pickup(agent)
        else:
            # carrying: try drop (only if cell empty)
            if grid[x, y] == -1:
                try_drop(agent)


# ---- Animation update ----
step_count = 0
def update(frame):
    global step_count
    # run several internal steps between frames for speed
    for _ in range(VISUALIZE_INTERVAL):
        step_simulation()
        step_count += 1
    im.set_data(grid_to_rgb(grid))
    ax.set_title(f"Termite clustering — step {step_count}")
    return (im,)

ani = animation.FuncAnimation(fig, update, frames=STEPS//VISUALIZE_INTERVAL, interval=50, blit=True)
plt.show()
