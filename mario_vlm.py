"""
Mario + Vision-Language Model (perception) + Nimble (decision).

Pipeline per step:
  1. Grab the Pygame frame from mario_game.py and encode it as base64 PNG.
  2. VLM returns object boxes (mario, goombas, pipes, coins, question blocks)
     and a one-line description of Mario's state. Default backend is OpenRouter
     (google/gemini-3.1-flash-lite, needs OPENROUTER_API_KEY); --vlm-backend
     ollama uses a local qwen2.5vl:7b instead.
  3. Boxes are turned into a compact text state: dangers ahead of Mario and how
     far, collectibles above/ahead. Pits are found with a pixel scan of the
     bottom row because the VLMs can't reliably see missing ground.
  4. Nimble (local Ollama, /v1/systemone) scores every allowed action and the
     highest-probability one is executed as a short key-press macro.

Modes:
  auto    - the agent plays turn-based: the game freezes while the models think,
            then the chosen action runs for a fraction of a second.
  advise  - you play; the agent watches in a background thread and its advice
            is shown on screen.

Run:
  python mario_vlm.py --mode auto
  python mario_vlm.py --mode advise
  python mario_vlm.py --mode auto --headless --steps 40 --out runs/test
  python mario_vlm.py --mode auto --policy rules     # hand-written policy instead of Nimble
  python mario_vlm.py --vlm google/gemma-4-26b-a4b-it  # slower but tighter boxes
  python mario_vlm.py --vlm-backend ollama           # fully local (slow on 8 GB GPUs)
"""
import argparse
import base64
import io
import json
import os
import sys
import threading
import time
import urllib.request

if "--headless" in sys.argv:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

import numpy as np
import pygame

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mario_game as mg  # noqa: E402

OLLAMA = "http://localhost:11434"
OPENROUTER_DEFAULT_VLM = 'google/gemini-3-flash-preview'# "google/gemini-3.1-flash-lite"
LABELS = ["mario", "goomba", "pipe", "coin", "question_block"]

# Box coordinate space each backend answers in (measured against game ground truth):
# OpenRouter models (Gemini, Gemma 4, Qwen3-VL) answer on a 0-1000 grid; Ollama's
# qwen2.5vl upscales 800x600 to 1036x756 (28px patches) and answers in those pixels.
BBOX_SPACE = {"openrouter": (1000, 1000), "ollama": (1036, 756)}
COORD_HINT = {"openrouter": "normalized to 0-1000 on both axes (x right, y down)",
              "ollama": "in pixels"}

VLM_PROMPT = """You are the vision system of an agent playing a 2D side-scrolling Mario game.
Legend: mario = small character with a red cap and blue overalls; goomba = brown mushroom enemy with white eyes walking on the ground; pipe = green tube standing on the ground; coin = small yellow oval; question_block = orange square with a '?'. Green triangles are background hills, white blobs are clouds, red rectangles are bricks: do not report them.
Return:
- objects: every mario, goomba, pipe, coin and question_block you see, with bbox_2d [x1, y1, x2, y2] {coords}.
- mario_state: one short sentence about Mario (on the ground or in the air, what is right in front of him)."""

VLM_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "objects": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "label": {"type": "string", "enum": LABELS},
                    "bbox_2d": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["label", "bbox_2d"],
            },
        },
        "mario_state": {"type": "string"},
    },
    "required": ["objects", "mario_state"],
}

ACTIONS = {
    "right": "Walk right: no danger (goomba, pipe or pit) within about 150 px ahead and no "
             "reachable coin or question block within about 250 px ahead.",
    "jump_right": "Jump forward: a goomba or a pipe is within about 150 px ahead, or an enemy is touching Mario.",
    "run_jump": "Sprint and long jump: a pit (gap in the ground) starts within about 220 px ahead.",
    "collect_coin": "Go get it: a reachable coin or question block is within about 250 px ahead "
                    "and no danger is closer than it.",
    "jump": "Jump straight up: a coin or question block is directly above Mario, no danger is "
            "within 150 px ahead, and the previous action was not jump.",
    "left": "Back off: a goomba is closer than 20 px and Mario is in the air or cannot jump.",
    "stay": "Wait: none of the above makes sense.",
}
NIMBLE_QUESTIONS = {
    "action": {
        "type": "choice",
        "instructions": "You control Mario in a side-scrolling platformer. Goal: reach the flag far to "
                        "the right while collecting as many coins as possible; avoiding goombas and "
                        "pits comes first, collecting coins comes second, plain walking comes last. "
                        "Distances are in pixels from Mario's front (right side); Mario walks about "
                        "110 px per move. Pick Mario's next move.",
        "criteria": ACTIONS,
    }
}
MAX_JUMP_REACH = 160  # px above Mario's head a full jump can touch (jump apex is ~175 px)
BLOCKING_COLORS = [mg.BRICK_RED, mg.BRICK_DARK, mg.USED_GRAY, mg.PIPE_GREEN, mg.PIPE_DARK]


def _post(path, body, timeout=300):
    req = urllib.request.Request(OLLAMA + path, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


# ------------------------------------------------------- perception (VLM)
def encode_frame(surface):
    """Pygame surface -> base64 PNG string."""
    buf = io.BytesIO()
    pygame.image.save(surface, buf, "png")
    return base64.b64encode(buf.getvalue()).decode()


_openrouter = None


def _vlm_ollama(image_b64, model, prompt):
    body = {
        "model": model,
        "stream": False,
        "think": False,
        "format": VLM_SCHEMA,
        "options": {"temperature": 0, "num_predict": 2000},
        "keep_alive": "30m",
        "messages": [{"role": "user", "content": prompt, "images": [image_b64]}],
    }
    return _post("/api/chat", body)["message"]["content"]


def _vlm_openrouter(image_b64, model, prompt):
    global _openrouter
    if _openrouter is None:
        from openrouter import OpenRouter
        key = os.getenv("OPENROUTER_API_KEY")
        if not key:
            sys.exit("Set OPENROUTER_API_KEY (or use --vlm-backend ollama).")
        _openrouter = OpenRouter(api_key=key)
    res = _openrouter.chat.send(
        model=model,
        temperature=0,
        max_tokens=800,
        response_format={"type": "json_schema",
                         "json_schema": {"name": "scene", "strict": True, "schema": VLM_SCHEMA}},
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
        ]}],
    )
    return res.choices[0].message.content


def query_vlm(image_b64, backend, model, retries=2):
    """Returns {"objects": [{"label", "box" (screen px)}], "mario_state", "latency"}."""
    prompt = VLM_PROMPT.format(coords=COORD_HINT[backend])
    call = _vlm_openrouter if backend == "openrouter" else _vlm_ollama
    t0 = time.time()
    for attempt in range(retries + 1):
        try:
            out = json.loads(call(image_b64, model, prompt))
            break
        except (json.JSONDecodeError, TypeError):  # truncated or empty output
            if attempt == retries:
                out = {"objects": [], "mario_state": "unknown (VLM output unparsable)"}
    bw, bh = BBOX_SPACE[backend]
    sx, sy = mg.SCREEN_WIDTH / bw, mg.SCREEN_HEIGHT / bh
    objs = [{"label": o["label"], "box": [b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy]}
            for o in out["objects"] if len(b := o.get("bbox_2d", [])) == 4]
    return {"objects": objs, "mario_state": out.get("mario_state", ""),
            "latency": round(time.time() - t0, 2)}


def find_pits(surface, min_width=20):
    """Screen-x intervals where the bottom row of the frame is sky."""
    row = pygame.surfarray.pixels3d(surface)[:, -1, :]
    sky = np.all(row == mg.SKY_BLUE, axis=1)
    pits, start = [], None
    for x, s in enumerate(np.append(sky, False)):
        if s and start is None:
            start = x
        elif not s and start is not None:
            if x - start >= min_width:
                pits.append((start, x))
            start = None
    return pits


def find_mario(surface, objects=()):
    """Mario's box from his exact cap/shirt colour, else from the VLM (he blinks after a hit)."""
    px = pygame.surfarray.pixels3d(surface)
    xs, ys = np.nonzero(np.all(px == mg.MARIO_RED, axis=2))
    if len(xs):
        return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.min()) + 36]
    boxes = [o["box"] for o in objects if o["label"] == "mario"]
    return boxes[0] if boxes else None


def blocked_from_below(surface, x, y_top, y_bottom):
    """True if a brick, used block or pipe sits between a collectible and Mario's head."""
    if surface is None or y_bottom <= y_top:
        return False
    px = pygame.surfarray.pixels3d(surface)
    x = int(np.clip(x, 1, px.shape[0] - 2))
    col = px[x - 1:x + 2, max(0, int(y_top)):int(y_bottom)].reshape(-1, 3)
    return any(np.all(col == c, axis=1).any() for c in BLOCKING_COLORS)


def build_state(perception, pits, mario, last_action, screen=None):
    """Compact, Mario-relative text state for the decision model.

    Returns (state, target): `target` is the nearest reachable collectible ahead
    in screen pixels (or None); it drives the collect_coin macro and is kept out
    of the text state."""
    state = {"mario": perception["mario_state"], "previous_action": last_action or "none",
             "dangers_ahead": [], "touching": [], "collectibles_above": [],
             "collectibles_ahead": []}
    targets = []
    if mario is None:
        state["mario"] = "not visible"
        return state, None
    mx1, my1, mx2, my2 = mario
    mcx = (mx1 + mx2) / 2
    for x1, x2 in pits:
        if x1 - mx2 > -5:
            state["dangers_ahead"].append({"object": "pit", "distance_px": int(max(0, x1 - mx2)),
                                           "width_px": int(x2 - x1)})
    for o in perception["objects"]:
        x1, y1, x2, y2 = o["box"]
        lab = o["label"]
        if lab == "mario":
            continue
        if lab in ("coin", "question_block"):
            height = int(my1 - y2)
            cx = (x1 + x2) / 2
            if height > MAX_JUMP_REACH or blocked_from_below(screen, cx, y2, my1):
                continue
            if x1 - 15 < mcx < x2 + 15:
                state["collectibles_above"].append({"object": lab, "height_px": max(0, height)})
            elif x1 > mx2 and x1 - mx2 <= 300:
                dist = int(x1 - mx2)
                state["collectibles_ahead"].append({"object": lab, "distance_px": dist,
                                                    "height_px": max(0, height)})
                targets.append((dist, {"x": cx, "label": lab}))
        elif x1 - mx2 > 0:
            entry = {"object": lab, "distance_px": int(x1 - mx2)}
            if lab == "pipe":
                entry["height_px"] = int(y2 - y1)
            state["dangers_ahead"].append(entry)
        elif x2 > mx1:
            state["touching"].append(lab)
    for k in ("dangers_ahead", "collectibles_ahead"):
        state[k] = sorted(state[k], key=lambda e: e["distance_px"])[:4]
    target = min(targets, key=lambda t: t[0])[1] if targets else None
    return state, target


# ---------------------------------------------------------- decision
def query_nimble(state, model="nimble:latest"):
    t0 = time.time()
    ans = _post("/v1/systemone", {"model": model, "state": state,
                                  "questions": NIMBLE_QUESTIONS, "keep_alive": "30m"})
    a = ans["answers"]["action"]
    return a["probabilities"], round(time.time() - t0, 2)


def pick(probs, state):
    """Highest-probability action that is consistent with the state.

    Jumping in place only makes sense under a coin/block, and never twice in a
    row (used blocks give nothing); collect_coin needs a collectible ahead."""
    ranked = sorted(probs, key=probs.get, reverse=True)
    for action in ranked:
        if action == "jump" and (not state["collectibles_above"]
                                 or state["previous_action"] == "jump"):
            continue
        if action == "collect_coin" and not state["collectibles_ahead"]:
            continue
        return action
    return "right"


def rules_policy(state):
    """Hand-written baseline used with --policy rules."""
    for e in state["dangers_ahead"]:
        if e["object"] == "pit" and e["distance_px"] <= 220:
            return "run_jump"
        if e["object"] in ("goomba", "pipe") and e["distance_px"] <= 150:
            return "jump_right"
    if state["touching"]:
        return "jump_right"
    if state["collectibles_above"] and state["previous_action"] != "jump":
        return "jump"
    if state["collectibles_ahead"]:
        return "collect_coin"
    return "right"


# ---------------------------------------------------------- execution
class Keys:
    def __init__(self, held):
        self.held = held

    def __getitem__(self, k):
        return k in self.held


def pit_gap(surface):
    m = find_mario(surface)
    if m is None:
        return None
    ahead = [x1 - m[2] for x1, _ in find_pits(surface) if x1 - m[2] > -5]
    return min(ahead) if ahead else None


def action_plan(action, game, target=None, lead=70):
    """Key-press macro: yields the set of keys held for each frame.

    `target` (screen x at decision time) is tracked through camera scrolling so
    collect_coin can stop right under the coin/block and jump straight up."""
    R, L, J = pygame.K_RIGHT, pygame.K_LEFT, pygame.K_SPACE
    if action == "right":
        yield from [{R}] * 20
    elif action == "left":
        yield from [{L}] * 15
    elif action == "jump":
        yield from [{J}] * 25
    elif action == "jump_right":
        yield from [{R, J}] * 35
    elif action == "run_jump":
        for _ in range(120):  # sprint until the edge is `lead` px away
            d = pit_gap(game.screen)
            if d is None or d < lead:
                break
            yield {R}
        yield from [{R, J}] * 45
    elif action == "collect_coin":
        if target is not None:
            world_x = target["x"] + game.camera_x
            for _ in range(90):  # walk until under it; friction slides ~20 px more
                m = find_mario(game.screen)
                if m is None or world_x - game.camera_x - (m[0] + m[2]) / 2 < 20:
                    break
                yield {R}
            yield from [set()] * 4
        yield from [{J}] * 25
    else:
        yield from [set()] * 10


# ---------------------------------------------------------- agent step
def settle(game, max_frames=8):
    """Mario blinks for 2 s after a hit; idle a few frames until he is drawn."""
    game.draw()
    for _ in range(max_frames):
        if find_mario(game.screen) is not None:
            break
        game.update(Keys(set()))
        game.draw()


def think(screen, args, last_action):
    """Frame -> VLM perception -> text state -> Nimble decision. Returns info dict."""
    perception = query_vlm(encode_frame(screen), args.vlm_backend, args.vlm)
    pits = find_pits(screen)
    mario = find_mario(screen, perception["objects"])
    state, target = build_state(perception, pits, mario, last_action, screen)
    info = {**perception, "pits": pits, "state": state, "target": target}
    if args.policy == "nimble":
        probs, info["nimble_latency"] = query_nimble(state, args.nimble)
        info["probs"] = probs
        info["decision"] = pick(probs, state)
    else:
        info["probs"], info["nimble_latency"] = {}, 0
        info["decision"] = rules_policy(state)
    return info


BOX_COLORS = {"mario": (255, 0, 0), "goomba": (255, 120, 0), "pipe": (0, 255, 0),
              "coin": (255, 255, 0), "question_block": (255, 0, 255)}


def draw_overlay(screen, font, info, boxes=False):
    if not info:
        return
    if boxes:
        for o in info.get("objects", []):
            x1, y1, x2, y2 = map(int, o["box"])
            pygame.draw.rect(screen, BOX_COLORS[o["label"]], (x1, y1, x2 - x1, y2 - y1), 2)
        for x1, x2 in info.get("pits", []):
            pygame.draw.rect(screen, (255, 0, 0), (x1, mg.SCREEN_HEIGHT - 12, x2 - x1, 12), 3)
    probs = info.get("probs", {})
    top = "  ".join(f"{k}:{v:.2f}" for k, v in sorted(probs.items(), key=lambda kv: -kv[1])[:3])
    lines = [f"ACTION: {info.get('decision', '?')}   [{top}]",
             f"VLM {info.get('latency', 0)}s  Nimble {info.get('nimble_latency', 0)}s",
             f"State: {info.get('mario_state', '')}"]
    panel = pygame.Surface((mg.SCREEN_WIDTH, 22 * len(lines) + 10), pygame.SRCALPHA)
    panel.fill((0, 0, 0, 170))
    screen.blit(panel, (0, 50))
    for i, line in enumerate(lines):
        screen.blit(font.render(line[:110], True, (255, 255, 255)), (8, 56 + 22 * i))


# ---------------------------------------------------------- loops
def run_headless(args):
    game = mg.Game()
    font = pygame.font.SysFont(None, 22)
    os.makedirs(args.out, exist_ok=True)
    log = open(os.path.join(args.out, "log.jsonl"), "w")
    last = None
    for step in range(args.steps):
        settle(game)
        try:
            info = think(game.screen, args, last)
        except Exception as e:
            print(f"stopping at step {step}: {type(e).__name__}: {e}")
            break
        snap = game.screen.copy()
        draw_overlay(snap, font, info, boxes=args.show_boxes)
        pygame.image.save(snap, os.path.join(args.out, f"step_{step:03d}.png"))
        rec = {"step": step, "world_x": round(game.mario.x), "lives": game.lives,
               "decision": info["decision"],
               "probs": {k: round(v, 3) for k, v in info["probs"].items()},
               "state": info["state"], "vlm_s": info["latency"], "nimble_s": info["nimble_latency"]}
        print(json.dumps(rec))
        log.write(json.dumps(rec) + "\n")
        last = info["decision"]
        for held in action_plan(last, game, info["target"]):
            game.update(Keys(held))
            game.draw()
        if game.state != "playing":
            break
    print(f"finished: state={game.state} x={game.mario.x:.0f} score={game.score} lives={game.lives}")


def run_window(args):
    game = mg.Game()
    font = pygame.font.SysFont(None, 22)
    info, last, worker, plan = {}, None, None, None
    lock = threading.Lock()

    def work(screen_copy, last_action):
        nonlocal info
        try:
            res = think(screen_copy, args, last_action)
        except Exception as e:  # keep the game alive if Ollama hiccups
            res = {"decision": "stay", "mario_state": f"error: {e}"}
        with lock:
            info = res
            info["fresh"] = True

    def start_worker():
        nonlocal worker
        if args.mode == "auto":
            settle(game)
        else:
            game.draw()
        worker = threading.Thread(target=work, args=(game.screen.copy(), last), daemon=True)
        worker.start()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                game.reset()
                plan = None

        if args.mode == "advise":
            if worker is None or not worker.is_alive():
                start_worker()
            game.update(pygame.key.get_pressed())
        else:  # auto: freeze while thinking, then execute the macro
            if plan is None:
                with lock:
                    fresh = info.pop("fresh", False)
                if fresh:
                    last = info["decision"]
                    plan = action_plan(last, game, info.get("target"))
                elif worker is None or not worker.is_alive():
                    start_worker()
            if plan is not None:
                held = next(plan, None)
                if held is None:
                    plan = None
                else:
                    game.update(Keys(held))

        game.draw()
        with lock:
            draw_overlay(game.screen, font, info,
                         boxes=args.show_boxes and args.mode == "auto" and plan is None)
        if args.mode == "auto" and plan is None:
            game.screen.blit(font.render("thinking...", True, (255, 255, 0)), (8, 130))
        pygame.display.flip()
        game.clock.tick(mg.FPS)
    pygame.quit()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["auto", "advise"], default="auto")
    ap.add_argument("--vlm-backend", choices=["openrouter", "ollama"], default="openrouter")
    ap.add_argument("--vlm", default=None,
                    help=f"vision model (default: {OPENROUTER_DEFAULT_VLM} on openrouter, qwen2.5vl:7b on ollama)")
    ap.add_argument("--nimble", default="nimble:latest", help="Ollama decision model")
    ap.add_argument("--policy", choices=["nimble", "rules"], default="nimble")
    ap.add_argument("--headless", action="store_true", help="no window (auto mode only)")
    ap.add_argument("--show-boxes", action="store_true", help="draw VLM boxes and pits (debugging)")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--out", default="runs/headless")
    args = ap.parse_args()
    if args.vlm is None:
        args.vlm = OPENROUTER_DEFAULT_VLM if args.vlm_backend == "openrouter" else "qwen2.5vl:7b"
    if args.headless:
        run_headless(args)
    else:
        run_window(args)


if __name__ == "__main__":
    main()
