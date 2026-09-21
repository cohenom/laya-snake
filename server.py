"""Local web server: runs an arena of many parallel Snake games and streams
all of them to the browser over one WebSocket. Single-user local demo, no
auth.
"""
import json
import time

from flask import Flask, send_from_directory
from flask_sock import Sock

from game import SnakeGame
from laya_brain import LayaBrain, heuristic_direction

app = Flask(__name__, static_folder="static")
sock = Sock(app)

# Laya's own forward pass floors at ~100ms per game regardless of batch size
# (measured: batching many games into one forward pass gave NO speedup on
# this CPU — ~95-100ms/game at N=1 and N=64 alike, unlike the GPU-only
# "7.2ms/q batched" figure in Laya's README). So instead of batching, one
# arena game gets a real Laya decision per tick, round-robin, while the rest
# move on the free heuristic_direction() fallback. The tick period is
# whatever that one real Laya call takes — no artificial sleep — so this
# runs literally as fast as Laya can produce a decision.
ARENA_SIZE = 8

print("Loading Laya agent (downloads weights on first run)...")
brain = LayaBrain()
print(f"Laya loaded in {brain.load_seconds:.1f}s")


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/health")
def health():
    return {
        "laya_loaded": True,
        "load_seconds": round(brain.load_seconds, 2),
        "decisions_made": brain.decision_count,
        "overrides_fired": brain.override_count,
        "last_latency_s": round(brain.last_latency, 4),
        "arena_size": ARENA_SIZE,
    }


class Slot:
    """One arena board: a game plus its running stats."""

    def __init__(self, slot_id):
        self.id = slot_id
        self.game = SnakeGame()
        self.probs = {}
        self.confidence = None
        self.chosen = None
        self.overridden = False
        self.best_score = 0
        self.deaths = 0
        self.alive_ticks = 0

    def to_json(self, is_active, latency_ms):
        g = self.game
        return {
            "id": self.id,
            "active": is_active,
            "body": [list(seg) for seg in g.body],
            "food": list(g.food) if g.food else None,
            "alive": g.alive,
            "score": g.score,
            "best_score": self.best_score,
            "deaths": self.deaths,
            "moves": g.moves,
            "probabilities": self.probs,
            "chosen": self.chosen,
            "overridden": self.overridden,
            "latency_ms": latency_ms,
        }


@sock.route("/ws")
def ws_game(wsock):
    slots = [Slot(i) for i in range(ARENA_SIZE)]
    tick = 0
    while True:
        t0 = time.time()
        active = tick % ARENA_SIZE
        active_latency_ms = 0.0
        for s in slots:
            g = s.game
            if s.id == active:
                chosen, probs, overridden, latency, confidence = brain.decide(g)
                s.probs, s.confidence, s.overridden = probs, confidence, overridden
                active_latency_ms = round(latency * 1000, 1)
            else:
                chosen = g.direction
                overridden = False
                if g.is_fatal(chosen):
                    chosen = heuristic_direction(g)
                    overridden = True
                s.overridden = overridden
            s.chosen = chosen
            g.step(chosen)
            s.best_score = max(s.best_score, g.score)
            s.alive_ticks += 1
            if not g.alive:
                s.deaths += 1
                g.reset()
                s.alive_ticks = 0

        frame = {
            "tick": tick,
            "active": active,
            "grid_width": 16,
            "grid_height": 16,
            "tick_ms": round((time.time() - t0) * 1000, 1),
            "decision_count": brain.decision_count,
            "override_count": brain.override_count,
            "slots": [s.to_json(s.id == active, active_latency_ms if s.id == active else 0.0) for s in slots],
        }
        wsock.send(json.dumps(frame))
        tick += 1


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8787, debug=False)
