"""Exact compiler evaluator used for differential checks, never mislabeled as an LLM."""
from . import circuit, protocol as p
from chipformer.q35.ir import evaluate


class IRRunner:
    def __init__(self, forest):
        self.forest = forest
        self.circuit = circuit.build()
        circuit.install(self.circuit, forest)
        self.carry = {}

    def tick(self, token):
        if not 0 <= token < p.VOCAB_SIZE:
            raise ValueError("Invalid circuit token")
        mat = evaluate(self.circuit, [token], carry=self.carry, check=True)
        return circuit.predicted(self.circuit, mat)

    def execute(self, token):
        trace = [token]
        for _ in range(8):
            token = self.tick(token)
            trace.append(token)
            if token in p.STOP:
                return trace
        raise RuntimeError("Compiled trace exceeded host limit")

