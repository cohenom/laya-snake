"""Local web server: runs an arena of many parallel Snake games and streams
all of them to the browser over one WebSocket. Single-user local demo, no
auth.

Movement and Laya's decisions are DECOUPLED on purpose (see README for the
full story): a BFS/flood-fill heuristic drives every game's actual moves
(fast, and measurably far better play — mean score ~41 in isolated testing
vs ~4-11 when Laya's own spatial guesses drove movement, since Laya was
never trained on grid/collision reasoning). A separate background thread
calls the real Laya model continuously, round-robin across boards, purely to
show its live vote/confidence and track how often it agrees with the actual
(smart) move — that's the honest way to keep Laya's real decisions visible
without letting them make the game worse.
"""
import json
import threading
import time
from collections import deque

from flask import Flask, send_from_directory
from flask_sock import Sock

from game import SnakeGame
from laya_brain import LayaBrain, heuristic_direction

app = Flask(__name__, static_folder="static")
sock = Sock(app)

ARENA_SIZE = 8
TICK_SECONDS = 0.08  # movement never waits on Laya, so this can be fast

print("Loading Laya agent (downloads weights on first run)...")
brain = LayaBrain()
print(f"Laya loaded in {brain.load_seconds:.1f}s")

lock = threading.Lock()


class Slot:
    """One arena board: a game, its stats, and Laya's latest (advisory) vote."""

    def __init__(self, slot_id):
        self.id = slot_id
        self.game = SnakeGame()
        self.chosen = None
        self.probs = {}
        self.confidence = None
        self.laya_choice = None
        self.laya_agreed = None
        self.best_score = 0
        self.deaths = 0
        self.agree_count = 0
        self.vote_count = 0

    def to_json(self, is_active_vote):
        g = self.game
        return {
            "id": self.id,
            "voting": is_active_vote,
            "body": [list(seg) for seg in g.body],
            "food": list(g.food) if g.food else None,
            "alive": g.alive,
            "score": g.score,
            "best_score": self.best_score,
            "deaths": self.deaths,
            "moves": g.moves,
            "chosen": self.chosen,
            "probabilities": self.probs,
            "laya_choice": self.laya_choice,
            "laya_agreed": self.laya_agreed,
            "agree_pct": round(100 * self.agree_count / self.vote_count, 1) if self.vote_count else None,
        }


slots = [Slot(i) for i in range(ARENA_SIZE)]
calls_per_sec = deque(maxlen=50)  # rolling timestamps of completed Laya calls
currently_voting_id = 0


def laya_voter_loop():
    """Runs forever in the background: asks Laya for its real opinion on one
    board at a time, round-robin, as fast as the model can answer. Never
    touches game.step() — purely advisory, feeds the UI's confidence graph.
    """
    global currently_voting_id
    i = 0
    while True:
        currently_voting_id = i % ARENA_SIZE
        s = slots[i % ARENA_SIZE]
        with lock:
            g = s.game
            heuristic_pick = heuristic_direction(g)
        try:
            chosen, probs, overridden, latency, confidence = brain.decide(g)
        except Exception:
            time.sleep(0.05)
            i += 1
            continue
        with lock:
            s.probs = probs
            s.confidence = confidence
            s.laya_choice = chosen
            s.laya_agreed = chosen == heuristic_pick
            s.vote_count += 1
            if s.laya_agreed:
                s.agree_count += 1
        calls_per_sec.append(time.time())
        i += 1


threading.Thread(target=laya_voter_loop, daemon=True).start()


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/health")
def health():
    with lock:
        return {
            "laya_loaded": True,
            "load_seconds": round(brain.load_seconds, 2),
            "decisions_made": brain.decision_count,
            "last_latency_s": round(brain.last_latency, 4),
            "arena_size": ARENA_SIZE,
        }


def current_calls_per_sec():
    now = time.time()
    recent = [t for t in calls_per_sec if now - t < 3.0]
    return round(len(recent) / 3.0, 1) if recent else 0.0


@sock.route("/ws")
def ws_game(wsock):
    tick = 0
    while True:
        t0 = time.time()
        with lock:
            for s in slots:
                g = s.game
                s.chosen = heuristic_direction(g)
                g.step(s.chosen)
                s.best_score = max(s.best_score, g.score)
                if not g.alive:
                    s.deaths += 1
                    g.reset()
            frame = {
                "tick": tick,
                "grid_width": 16,
                "grid_height": 16,
                "tick_ms": round((time.time() - t0) * 1000, 1),
                "calls_per_sec": current_calls_per_sec(),
                "decision_count": brain.decision_count,
                "slots": [s.to_json(currently_voting_id == s.id) for s in slots],
            }
        wsock.send(json.dumps(frame))
        tick += 1
        time.sleep(max(0.0, TICK_SECONDS - (time.time() - t0)))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8787, debug=False, threaded=True)
