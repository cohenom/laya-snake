"""Wraps the real Laya agent's `choice` primitive as a Snake move-picker.

Laya (convaiinnovations/laya) is a non-autoregressive BERT-style classifier
trained to answer typed questions over a JSON state in a single forward pass.
It was never trained on Snake, or on any spatial/game task — we're repurposing
its general `choice` primitive ("given this state, pick the best of N labeled
options") to select a movement direction each tick.
"""
import time

import numpy as np
import torch
import laya
import laya.agent as _agent_mod

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

    def decide_batch(self, games):
        """Batch many independent Snake games into ONE Laya forward pass.

        Laya's public `predict()` only accepts a single state per call, but
        its own batching primitives (collate_items/build_sequence) work over
        an arbitrary number of independent items — that's the same mechanism
        the README's "10 questions batched -> 7.2ms/q" numbers come from,
        just applied across game states instead of across questions on one
        state. This reaches into laya.agent's internals since there's no
        public multi-state API; if that internal shape ever changes, fall
        back to N sequential decide() calls rather than crashing the arena.

        Returns a list of (chosen, probs, overridden, confidence) aligned to
        `games`, plus the batched latency in seconds for the whole call.
        """
        t0 = time.time()
        try:
            results = self._decide_batch_native(games)
            latency = time.time() - t0
            self.decision_count += len(games)
            self.last_latency = latency
            return results, latency
        except Exception:
            out = []
            for g in games:
                chosen, probs, overridden, _lat, conf = self.decide(g)
                out.append((chosen, probs, overridden, conf))
            return out, time.time() - t0

    def _decide_batch_native(self, games):
        agent = self.agent
        tok = agent.tok
        max_len = agent.cfg.get("max_len", 512)
        head_max_len = agent.cfg.get("head_max_len", 192)
        q = agent._to_internal(QUESTIONS["direction"])
        keys = list(q["crit"].keys())

        batch = []
        for g in games:
            state = g.to_state_dict()
            seq, markers = _agent_mod.build_sequence(tok, state, q, max_len, head_max_len)
            batch.append([{"ids": seq, "markers": markers, "qtype": _agent_mod.QTYPES["choice"]}])

        b = _agent_mod.collate_items(batch, tok.pad_token_id)
        use_amp = agent.device.type == "cuda"
        with torch.no_grad():
            with torch.autocast(device_type=agent.device.type, dtype=agent.dtype, enabled=use_amp):
                logits, _act = agent.model(
                    b["input_ids"].to(agent.device),
                    b["attention_mask"].to(agent.device),
                    b["marker_pos"].to(agent.device),
                    b["marker_mask"].to(agent.device),
                    b["qtype"].to(agent.device),
                )
        logits = logits.float().cpu().numpy()

        k = len(keys)
        choice_qtype = _agent_mod.QTYPES["choice"]
        t_scale = agent.temperature_by_options.get(
            _agent_mod.temp_bucket(choice_qtype, k), agent.temperature[choice_qtype]
        )

        results = []
        for i, g in enumerate(games):
            z = logits[i, :k] / t_scale
            p = np.exp(z - z.max())
            p = p / p.sum()
            probs = {kk: round(float(v), 4) for kk, v in zip(keys, p)}
            laya_choice = keys[int(p.argmax())]
            confidence = round(_agent_mod.confidence_from_probs(p, k), 4)

            chosen = laya_choice
            overridden = False
            if g.is_fatal(laya_choice):
                chosen = heuristic_direction(g, candidates=probs)
                overridden = True
                self.override_count += 1
            results.append((chosen, probs, overridden, confidence))
        return results
