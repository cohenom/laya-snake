"""Local web server: runs the Snake game loop, asks the real Laya agent for a
move each tick, and streams state + probabilities to the browser over a
WebSocket. Single-user local demo, no auth.
"""
import json
import time

from flask import Flask, send_from_directory
from flask_sock import Sock

from game import SnakeGame
from laya_brain import LayaBrain

app = Flask(__name__, static_folder="static")
sock = Sock(app)

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


def build_frame(game, chosen, probs, overridden, latency, confidence):
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
    }


@sock.route("/ws")
def ws_game(wsock):
    game = SnakeGame()
    while True:
        chosen, probs, overridden, latency, confidence = brain.decide(game)
        game.step(chosen)
        frame = build_frame(game, chosen, probs, overridden, latency, confidence)
        wsock.send(json.dumps(frame))
        if not game.alive:
            time.sleep(1.5)
            game.reset()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8787, debug=False)
