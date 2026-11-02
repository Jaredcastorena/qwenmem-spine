"""Small, fixed vocabulary. Generated trace tokens carry the navigation decisions."""
CAPACITY = 64
MAX_STEPS = 8
OPEN, MOVE, ALT, APEX = 0, 64, 128, 192
ROUTE = 256
PEEK, LEFT, RIGHT, READY, INVALID, LIMIT, LEAF, PAD = range(264, 272)
VOCAB_SIZE = 272
STOP = {READY, INVALID, LIMIT, LEAF}


def name(token):
    for base, label in ((OPEN, "open"), (MOVE, "move"), (ALT, "alternate"), (APEX, "apex")):
        if base <= token < base + CAPACITY:
            return f"{label}:{token-base}"
    if ROUTE <= token < ROUTE + MAX_STEPS:
        return f"route:{token-ROUTE}"
    return dict(zip(range(264, 272), ("peek", "left", "right", "ready", "invalid", "limit", "leaf", "pad")))[token]


def command(text):
    if text in ("peek", "left", "right"):
        return {"peek": PEEK, "left": LEFT, "right": RIGHT}[text]
    kind, _, value = text.partition(":")
    if kind not in ("open", "route") or not value.isdigit():
        raise ValueError("Expected peek, left, right, open:SLOT or route:INDEX")
    n = int(value)
    bound = CAPACITY if kind == "open" else MAX_STEPS
    if not 0 <= n < bound:
        raise ValueError("Address outside compiled capacity")
    return (OPEN if kind == "open" else ROUTE) + n

