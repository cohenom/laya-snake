"""Local web server: runs the Snake game loop, asks the real Laya agent for a
move each tick, and streams state + probabilities to the browser over a
WebSocket. Single-user local demo, no auth.
"""
import json
import time

from flask import Flask, send_from_directory
from flask_sock import Sock

from game import SnakeGame
from laya_brain import LayaBrain, heuristic_direction

app = Flask(__name__, static_folder="static")
sock = Sock(app)

# Laya's own forward pass takes ~100ms; calling it every tick makes movement
# feel exactly as slow as the model. Call it every DECISION_EVERY ticks for
# strategy, and use the free heuristic_direction() fallback (no model call)
# on the ticks in between, so the snake moves at TICK_SECONDS regardless of
# Laya's latency.
DECISION_EVERY = 4
TICK_SECONDS = 0.12

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
    }


def build_frame(game, chosen, probs, overridden, latency, confidence, laya_this_tick):
    return {
        "grid_width": 16,
        "grid_height": 16,
        "body": [list(seg) for seg in game.body],
        "food": list(game.food) if game.food else None,
        "direction": game.direction,
        "alive": game.alive,
        "score": game.score,
        "moves": game.moves,
        "probabilities": probs,
        "laya_confidence": confidence,
        "chosen": chosen,
        "overridden": overridden,
        "override_count": brain.override_count,
        "decision_count": brain.decision_count,
        "latency_ms": round(latency * 1000, 1),
        "laya_this_tick": laya_this_tick,
    }


@sock.route("/ws")
def ws_game(wsock):
    game = SnakeGame()
    probs, confidence = {}, None
    tick = 0
    while True:
        t0 = time.time()
        laya_this_tick = tick % DECISION_EVERY == 0
        if laya_this_tick:
            chosen, probs, overridden, latency, confidence = brain.decide(game)
        else:
            chosen = game.direction
            overridden = False
            if game.is_fatal(chosen):
                chosen = heuristic_direction(game)
                overridden = True
            latency = time.time() - t0
        game.step(chosen)
        frame = build_frame(game, chosen, probs, overridden, latency, confidence, laya_this_tick)
        wsock.send(json.dumps(frame))
        tick += 1
        if not game.alive:
            time.sleep(1.5)
            game.reset()
            tick = 0
        else:
            time.sleep(max(0.0, TICK_SECONDS - (time.time() - t0)))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8787, debug=False)
