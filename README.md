# Laya Snake

Snake, controlled every tick by a real forward pass of [Laya](https://github.com/NandhaKishorM/laya)
(`pip install laya`, by Convai Innovations).

## What Laya actually is (read this before you judge the gameplay)

Laya is **not** a game-playing AI or a reinforcement-learning agent. It's a
non-autoregressive, BERT-style classifier (ModernBERT-large, ~421M params)
trained to answer typed questions about a JSON "state" object in a single
forward pass — the kind of thing you'd use for email triage, routing, or
moderation decisions. Its core primitive is `choice`: give it a state, a set
of labeled `criteria`, and it returns a probability distribution over the
options plus a top pick and a confidence score.

This project repurposes that `choice` primitive to pick a Snake movement
direction each tick. Every tick we serialize the board (head position,
direction, food position, body segments, and a `danger_<direction>` boolean
per direction) into a JSON dict and ask Laya:

> "Given the snake game state, which direction should the snake move next to
> eat the food while avoiding walls and its own body?" — options: up / down /
> left / right.

Laya was never trained on spatial/grid reasoning or Snake specifically, so it
sometimes picks a direction that is immediately fatal. When that happens, a
small safety-net heuristic (`laya_brain.py`) overrides it with the nearest
safe direction toward the food, and the UI shows a red "safety override"
badge plus a running override count so you can see exactly how often Laya's
raw pick would have killed the snake.

This is an honest, slightly silly demo of stretching a text-decision model
onto a task it wasn't built for — not a claim that Laya is a game AI.

## What was actually measured on this machine (Apple Silicon Mac, CPU)

- Model load (first run, downloads ~a few hundred MB from Hugging Face):
  ~22–37s. Cached loads are faster.
- Steady-state inference latency per `choice` call, in the running server:
  **~90–100ms per tick** (an isolated microbenchmark without Flask/WebSocket
  overhead measured ~45ms/call after warmup). This is CPU inference — no GPU
  numbers are claimed because none were measured.
- Over a 60-tick sample run, Laya's raw pick was immediately fatal (safety
  override fired) roughly **10% of the time**. It survived 60+ ticks without
  dying in that run, but it visibly wanders — it does not path-plan toward
  food, it answers a one-shot classification question with no lookahead or
  memory of past ticks.

## Running it locally

Requires Python 3.12 (Laya's torch/transformers pins are not yet compatible
with 3.14, which is this machine's default `python3`).

```bash
cd laya-snake
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install laya flask flask-sock
python server.py
```

Then open **http://127.0.0.1:8787/** in a browser. The first request to load
the model downloads Laya's weights from Hugging Face (a few hundred MB to
~1GB), so the server takes ~20-40s to become ready — watch the terminal log.

`GET /health` returns a small JSON status (decisions made, override count,
last latency) if you want to check the server without opening the UI.

## Architecture

- `game.py` — pure Snake grid mechanics (no AI). Also builds the JSON state
  dict handed to Laya and a `danger_<direction>` map used both by Laya's
  question criteria and by the safety-net override.
- `laya_brain.py` — loads the real Laya agent (`laya.load()`), asks the
  `choice` question each tick, and applies the safety-net override.
- `server.py` — Flask + `flask-sock` WebSocket server. Runs the game loop
  per-connection and streams a JSON frame (board state, Laya's probabilities,
  chosen direction, override flag, score, latency) to the browser every tick.
- `static/index.html` — single-page vanilla-JS UI: canvas grid for the board,
  a live confidence bar per direction, and simple score/override/latency
  stats. No build step.

## Honesty note on hosting

This cannot be hosted as a GitHub Pages static demo — the backend needs a
Python process to load a ~421M-parameter PyTorch model and run inference
every tick, which GitHub Pages (static file hosting only) cannot do. Run it
locally as described above.
