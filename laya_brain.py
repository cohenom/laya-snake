"""Wraps the real Laya agent's `choice` primitive as a Snake move-picker.

Laya (convaiinnovations/laya) is a non-autoregressive BERT-style classifier
trained to answer typed questions over a JSON state in a single forward pass.
It was never trained on Snake, or on any spatial/game task — we're repurposing
its general `choice` primitive ("given this state, pick the best of N labeled
options") to select a movement direction each tick.
"""
import time

import laya

QUESTIONS = {
    "direction": {
        "type": "choice",
        "instructions": (
            "Given the snake game state, which direction should the snake move "
            "next? Correct answer must move toward the food's direction "
            "according to the food_is_*_of_head flags, and must not be a "
            "danger_* direction that is true."
        ),
        "criteria": {
            "up": "correct if food_is_above_head is true and danger_up is false",
            "down": "correct if food_is_below_head is true and danger_down is false",
            "left": "correct if food_is_left_of_head is true and danger_left is false",
            "right": "correct if food_is_right_of_head is true and danger_right is false",
        },
    }
}


def heuristic_direction(game, candidates=None):
    """Free, instant fallback move: among non-fatal directions, the one that
    reduces Manhattan distance to the food. No model call. Used both as
    Laya's safety net and as the between-decision mover (see server.py) so
    the snake doesn't have to wait on a ~100ms Laya call every tick.
    """
    from game import DIRS

    candidates = candidates if candidates is not None else DIRS
    safe_dirs = [d for d in candidates if not game.is_fatal(d)]
    if not safe_dirs:
        return game.direction  # no safe move; step() will end the game

    hx, hy = game.head()
    fx, fy = game.food if game.food else (hx, hy)

    def dist_after(d):
        dx, dy = DIRS[d]
        return abs((hx + dx) - fx) + abs((hy + dy) - fy)

    return min(safe_dirs, key=dist_after)


class LayaBrain:
    def __init__(self, device=None):
        t0 = time.time()
        self.agent = laya.load(device=device)
        self.load_seconds = time.time() - t0
        self.override_count = 0
        self.decision_count = 0
        self.last_latency = 0.0

    def decide(self, game):
        """Ask Laya for a direction; returns (direction, probabilities, overridden, latency_s)."""
        state = game.to_state_dict()
        t0 = time.time()
        result = self.agent.predict(state, QUESTIONS)
        latency = time.time() - t0
        self.last_latency = latency
        self.decision_count += 1

        ans = result["answers"]["direction"]
        probs = ans["probabilities"]
        laya_choice = ans["choice"]

        chosen = laya_choice
        overridden = False
        if game.is_fatal(laya_choice):
            # Safety net: Laya was never trained on grid/collision spatial
            # reasoning, so it can (and does) sometimes pick a direction that
            # is immediately fatal.
            chosen = heuristic_direction(game, candidates=probs)
            overridden = True
            self.override_count += 1

        return chosen, probs, overridden, latency, ans.get("confidence")
