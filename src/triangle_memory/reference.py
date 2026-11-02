"""Independent integer oracle. It is never called by the compiled controller."""
from . import protocol as p


class Reference:
    def __init__(self, forest):
        self.forest = forest
        self.cursor = None
        self.pending = None
        self.steps = 0
        self.routes = {}

    def tick(self, token):
        if p.OPEN <= token < p.MOVE:
            self.cursor = token-p.OPEN
            self.steps = 0
            self.routes = {}
        elif p.MOVE <= token < p.ALT:
            self.cursor = token-p.MOVE
        elif p.ALT <= token < p.APEX:
            return p.INVALID if self.pending is None else p.MOVE+self.pending
        elif p.ROUTE <= token < p.PEEK:
            return p.MOVE+self.routes[token-p.ROUTE] if token-p.ROUTE in self.routes else p.INVALID
        elif token not in (p.PEEK, p.LEFT, p.RIGHT):
            return p.READY
        record = self.forest.records.get(self.cursor)
        if token in (p.LEFT, p.RIGHT):
            self.steps += 1
        if record is None:
            return p.INVALID
        if token not in (p.LEFT, p.RIGHT):
            return p.APEX+record.apex
        if not record.triangle:
            return p.LEAF
        if self.steps > p.MAX_STEPS:
            return p.LIMIT
        self.pending = record.left if token == p.LEFT else record.right
        alternate = record.right if token == p.LEFT else record.left
        self.routes[self.steps-1] = alternate
        return p.ALT+alternate

    def execute(self, token):
        trace = [token]
        for _ in range(8):
            token = self.tick(token)
            trace.append(token)
            if token in p.STOP:
                return trace
        raise RuntimeError("Oracle failed to terminate")

