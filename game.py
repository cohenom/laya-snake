"""Snake game state/logic. No AI here — pure grid mechanics."""
import random

GRID_W, GRID_H = 16, 16
DIRS = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}
OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}


class SnakeGame:
    def __init__(self):
        self.reset()

    def reset(self):
        cx, cy = GRID_W // 2, GRID_H // 2
        self.body = [(cx, cy), (cx - 1, cy), (cx - 2, cy)]
        self.direction = "right"
        self.score = 0
        self.moves = 0
        self.alive = True
        self.food = self._spawn_food()

    def _spawn_food(self):
        occupied = set(self.body)
        free = [
            (x, y)
            for x in range(GRID_W)
            for y in range(GRID_H)
            if (x, y) not in occupied
        ]
        return random.choice(free) if free else None

    def head(self):
        return self.body[0]

    def is_fatal(self, direction):
        """Would moving in `direction` immediately kill the snake?"""
        hx, hy = self.head()
        dx, dy = DIRS[direction]
        nx, ny = hx + dx, hy + dy
        if nx < 0 or nx >= GRID_W or ny < 0 or ny >= GRID_H:
            return True
        # tail cell is vacated this tick unless we just ate, so it's safe
        body_to_check = self.body[:-1] if len(self.body) > 1 else self.body
        if (nx, ny) in body_to_check:
            return True
        return False

    def danger_map(self):
        return {d: self.is_fatal(d) for d in DIRS}

    def step(self, direction):
        """Apply one tick. Returns True if still alive after the move."""
        if not self.alive:
            return False
        # ignore a 180-degree reversal request (classic snake rule), keep going straight
        if direction == OPPOSITE.get(self.direction):
            direction = self.direction
        fatal = self.is_fatal(direction)
        self.direction = direction
        self.moves += 1
        if fatal:
            self.alive = False
            return False

        dx, dy = DIRS[direction]
        hx, hy = self.head()
        new_head = (hx + dx, hy + dy)
        self.body.insert(0, new_head)
        if new_head == self.food:
            self.score += 1
            self.food = self._spawn_food()
            if self.food is None:
                self.alive = False  # board full, you win
        else:
            self.body.pop()
        return self.alive

    def to_state_dict(self):
        """Serialize state for Laya's `choice` question."""
        hx, hy = self.head()
        fx, fy = self.food if self.food else (hx, hy)
        danger = self.danger_map()
        return {
            "grid_width": GRID_W,
            "grid_height": GRID_H,
            "head_x": hx,
            "head_y": hy,
            "current_direction": self.direction,
            "food_x": fx,
            "food_y": fy,
            "food_is_left_of_head": fx < hx,
            "food_is_right_of_head": fx > hx,
            "food_is_above_head": fy < hy,
            "food_is_below_head": fy > hy,
            "body_segments": [list(seg) for seg in self.body],
            "danger_up": danger["up"],
            "danger_down": danger["down"],
            "danger_left": danger["left"],
            "danger_right": danger["right"],
            "score": self.score,
            "moves": self.moves,
        }
