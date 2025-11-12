# Dolphins vs Fish (Predator–Prey) on a Toroidal Grid
# ----------------------------------------------------
# A rewrite of the termite clustering sim into a predator–prey model.
# - The world is a 2D torus.
# - Fish move randomly and can reproduce.
# - Dolphins move; if they see a fish within vision, they bias movement toward it.
# - Dolphins lose energy each step; eating a fish restores energy; they reproduce if energetic.
# - Simple single-occupancy per cell for dolphins; fish are also single-occupancy here for clarity.
# - Visualization uses matplotlib animation.

import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ------------------ Parameters ------------------ #
GRID_SIZE = 80
INITIAL_FISH = 100
INITIAL_DOLPHINS = 14

FISH_REPRODUCTION_PROB = 0.02   # probability per step that a fish reproduces into a neighboring empty cell
FISH_MOVE_PROB = 0.9            # fish attempt to move this step

DOLPHIN_VISION_RADIUS = 5       # how far dolphins can "see" fish (Chebyshev distance)
DOLPHIN_MOVE_BIAS = 0.85        # probability to step toward the nearest seen fish (else random)
DOLPHIN_STEP_COST = 1           # energy cost per step
DOLPHIN_EAT_GAIN = 14           # energy gained by eating one fish
DOLPHIN_REPRODUCTION_THRESHOLD = 24
DOLPHIN_REPRODUCTION_COST = 12  # energy lost by parent when reproducing

STEPS = 2000
VISUALIZE_EVERY = 5
RANDOM_SEED = 0  # set to None for fully random

# ------------------ Utilities ------------------ #

def torus(x: int, n: int) -> int:
    return x % n

# Moore neighborhood displacements (8 neighbors + stay)
NEIGHBOR_STEPS: List[Tuple[int, int]] = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 0), (0, 1),
    (1, -1), (1, 0), (1, 1)
]

# 8-neighborhood (no standstill) for movement
MOVE_STEPS: List[Tuple[int, int]] = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1)
]

# ------------------ Entities ------------------ #

@dataclass
class Dolphin:
    x: int
    y: int
    energy: int

# For simplicity, fish are represented by a boolean occupancy grid (True = fish present)

# ------------------ Initialization ------------------ #

def init_world(size: int) -> Tuple[np.ndarray, List[Dolphin], np.ndarray]:
    """
    Returns (fish_grid, dolphins, dolphin_grid)
    fish_grid: bool array shape (size, size)
    dolphin_grid: bool array shape (size, size)
    """
    fish_grid = np.zeros((size, size), dtype=bool)
    dolphin_grid = np.zeros((size, size), dtype=bool)

    # place fish at random empty cells
    all_cells = [(i, j) for i in range(size) for j in range(size)]
    random.shuffle(all_cells)

    fish_cells = all_cells[:INITIAL_FISH]
    for (i, j) in fish_cells:
        fish_grid[i, j] = True

    # place dolphins at random empty cells (avoid fish cells for clarity at t=0)
    remaining = [c for c in all_cells if c not in fish_cells]
    random.shuffle(remaining)

    dolphins: List[Dolphin] = []
    for (i, j) in remaining[:INITIAL_DOLPHINS]:
        dolphin_grid[i, j] = True
        dolphins.append(Dolphin(i, j, energy=DOLPHIN_EAT_GAIN))

    return fish_grid, dolphins, dolphin_grid

# ------------------ Fish dynamics ------------------ #

def fish_step(fish_grid: np.ndarray, dolphin_grid: np.ndarray) -> None:
    size = fish_grid.shape[0]
    # Move fish (process in random order to reduce drift)
    fish_positions = list(zip(*np.where(fish_grid)))
    random.shuffle(fish_positions)

    new_fish_grid = fish_grid.copy()

    for (x, y) in fish_positions:
        if not fish_grid[x, y]:
            # might have moved earlier in this loop
            continue
        # attempt movement
        if random.random() < FISH_MOVE_PROB:
            dx, dy = random.choice(MOVE_STEPS)
            nx, ny = torus(x + dx, size), torus(y + dy, size)
            # move only if destination is empty of fish and dolphin
            if (not fish_grid[nx, ny]) and (not dolphin_grid[nx, ny]) and (not new_fish_grid[nx, ny]):
                new_fish_grid[x, y] = False
                new_fish_grid[nx, ny] = True
                x, y = nx, ny  # update local variables for reproduction
        # reproduction attempt into a random neighbor empty of fish & dolphin
        if random.random() < FISH_REPRODUCTION_PROB:
            dx, dy = random.choice(MOVE_STEPS)
            rx, ry = torus(x + dx, size), torus(y + dy, size)
            if (not new_fish_grid[rx, ry]) and (not dolphin_grid[rx, ry]):
                new_fish_grid[rx, ry] = True

    fish_grid[:, :] = new_fish_grid

# ------------------ Dolphin dynamics ------------------ #


def nearest_fish_direction(fish_grid: np.ndarray, x: int, y: int, R: int) -> Optional[Tuple[int, int]]:
    """Return a unit step (dx, dy) toward the nearest fish within Chebyshev radius R, or None if none seen."""
    size = fish_grid.shape[0]
    best: Optional[Tuple[int, int]] = None
    best_dist = None

    # Scan vision square
    for dx in range(-R, R + 1):
        for dy in range(-R, R + 1):
            if dx == 0 and dy == 0:
                continue
            nx, ny = torus(x + dx, size), torus(y + dy, size)
            if fish_grid[nx, ny]:
                # Chebyshev distance
                d = max(abs(dx), abs(dy))
                if best_dist is None or d < best_dist:
                    best_dist = d
                    step_x = 0 if dx == 0 else (1 if dx > 0 else -1)
                    step_y = 0 if dy == 0 else (1 if dy > 0 else -1)
                    best = (step_x, step_y)

    return best


def dolphins_step(fish_grid: np.ndarray, dolphins: List[Dolphin], dolphin_grid: np.ndarray) -> None:
    size = fish_grid.shape[0]

    # Process dolphins in random order each step
    order = list(range(len(dolphins)))
    random.shuffle(order)

    # We'll rebuild dolphin occupancy as we go to avoid conflicts
    new_dolphin_grid = np.zeros_like(dolphin_grid)

    new_dolphins: List[Dolphin] = []

    for idx in order:
        d = dolphins[idx]

        # choose movement
        step = None
        if random.random() < DOLPHIN_MOVE_BIAS:
            step = nearest_fish_direction(fish_grid, d.x, d.y, DOLPHIN_VISION_RADIUS)
        if step is None:
            step = random.choice(MOVE_STEPS)

        nx, ny = torus(d.x + step[0], size), torus(d.y + step[1], size)

        # If another dolphin already took that cell this tick, try a random fallback once
        if new_dolphin_grid[nx, ny]:
            fx, fy = nx, ny
            step = random.choice(MOVE_STEPS)
            nx, ny = torus(d.x + step[0], size), torus(d.y + step[1], size)
            # if still occupied, stay put
            if new_dolphin_grid[nx, ny]:
                nx, ny = d.x, d.y

        # Move there
        d.x, d.y = nx, ny

        # Step energy cost
        d.energy -= DOLPHIN_STEP_COST

        # Eat if fish present
        if fish_grid[nx, ny]:
            fish_grid[nx, ny] = False
            d.energy += DOLPHIN_EAT_GAIN

        # Death check
        if d.energy <= 0:
            continue  # dolphin dies; do not add to new list

        # Reproduction: split energy to a new dolphin in a neighbor if threshold reached
        if d.energy >= DOLPHIN_REPRODUCTION_THRESHOLD:
            # try to place baby in a random neighboring empty cell (w.r.t. dolphins only)
            random.shuffle(MOVE_STEPS)
            placed = False
            for dx, dy in MOVE_STEPS:
                bx, by = torus(d.x + dx, size), torus(d.y + dy, size)
                if not new_dolphin_grid[bx, by] and not dolphin_grid[bx, by]:
                    # place calf with half of reproduction cost as starting energy (simple choice)
                    calf_energy = max(4, DOLPHIN_REPRODUCTION_COST // 2)
                    new_dolphins.append(Dolphin(bx, by, energy=calf_energy))
                    new_dolphin_grid[bx, by] = True
                    d.energy -= DOLPHIN_REPRODUCTION_COST
                    placed = True
                    break
            # if cannot place, skip reproduction this tick

        # Occupy cell
        new_dolphin_grid[d.x, d.y] = True
        new_dolphins.append(d)

    # write back
    dolphin_grid[:, :] = new_dolphin_grid
    dolphins[:] = new_dolphins

# ------------------ Visualization ------------------ #

from matplotlib import colors

FISH_COLOR = (0.2, 0.6, 1.0)        # light blue
DOLPHIN_COLOR = (0.0, 0.0, 0.0)     # black
EMPTY_COLOR = (1.0, 1.0, 1.0)       # white


def render_rgb(fish_grid: np.ndarray, dolphin_grid: np.ndarray) -> np.ndarray:
    """Return an (H, W, 3) float RGB image."""
    h, w = fish_grid.shape
    img = np.ones((h, w, 3), dtype=float)

    # color fish
    fish_mask = fish_grid & (~dolphin_grid)
    img[fish_mask] = FISH_COLOR

    # color dolphins (over fish; fish are eaten when co-located this frame, but rendering dolphins last keeps them visible)
    dolphin_mask = dolphin_grid
    img[dolphin_mask] = DOLPHIN_COLOR

    return img

# ------------------ Main Loop / Animation ------------------ #

def main():
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

    fish_grid, dolphins, dolphin_grid = init_world(GRID_SIZE)

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.set_title("Dolphins (black) hunting Fish (blue)")
    ax.set_axis_off()

    img = ax.imshow(render_rgb(fish_grid, dolphin_grid), interpolation="nearest", animated=True)

    step_counter = {"t": 0}

    def advance(_):
        # advance multiple internal steps per frame
        for _ in range(VISUALIZE_EVERY):
            fish_step(fish_grid, dolphin_grid)
            dolphins_step(fish_grid, dolphins, dolphin_grid)
            step_counter["t"] += 1
        img.set_data(render_rgb(fish_grid, dolphin_grid))
        ax.set_title(f"Dolphins vs Fish — step {step_counter['t']} | fish: {int(fish_grid.sum())} | dolphins: {len(dolphins)}")
        return (img,)

    anim = FuncAnimation(fig, advance, frames=STEPS // VISUALIZE_EVERY, interval=30, blit=True)
    plt.show()


if __name__ == "__main__":
    main()
