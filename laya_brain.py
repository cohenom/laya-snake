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
            "Given the snake game state, which direction should the snake "
            "move next to eat the food while avoiding walls and its own body?"
        ),
        "criteria": {
            "up": "move up (decrease y): safe only if danger_up is false",
            "down": "move down (increase y): safe only if danger_down is false",
            "left": "move left (decrease x): safe only if danger_left is false",
            "right": "move right (increase x): safe only if danger_right is false",
        },
    }
}


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
            # is immediately fatal. Fall back to the safest heuristic move —
            # among non-fatal directions, the one that reduces Manhattan
            # distance to the food, else any non-fatal direction, else give up.
            safe_dirs = [d for d in probs if not game.is_fatal(d)]
            if safe_dirs:
                hx, hy = game.head()
                fx, fy = game.food if game.food else (hx, hy)

                def dist_after(d):
                    from game import DIRS

                    dx, dy = DIRS[d]
                    return abs((hx + dx) - fx) + abs((hy + dy) - fy)

                chosen = min(safe_dirs, key=dist_after)
                overridden = True
                self.override_count += 1

        return chosen, probs, overridden, latency, ans.get("confidence")
