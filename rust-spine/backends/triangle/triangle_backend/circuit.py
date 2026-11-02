"""Compile triangle records/cursor/routes into exact SwiGLU + Gated DeltaNet layers."""
from .upstream import enable
enable()

import numpy as np
from chipformer.q35.ir import A_BIT, A_REF, K, Head, Target
from chipformer.q35.prims import Circ
from . import protocol as p


def build(target=None):
    c = Circ(target or Target())
    one = c.one
    f = c.token_feature
    op = {key: f(key, lambda t, lo=lo, hi=hi: int(lo <= t < hi)) for key, lo, hi in (
        ("open", 0, 64), ("move", 64, 128), ("alt", 128, 192), ("apex", 192, 256),
        ("route", 256, 264), ("peek", 264, 265), ("left", 265, 266), ("right", 266, 267))}
    bits = [f(f"input{i}", lambda t, i=i: (t >> i) & 1) for i in range(6)]
    code = [f(f"code{i}", lambda t, i=i: 128*(2*((t >> i)&1)-1)) for i in range(9)]
    step = op["left"]+op["right"]
    addressed = op["open"]+op["move"]+op["peek"]+step
    cursor = c.slot_head("cursor", {0: op["open"]+op["move"]}, {0: one},
                         {2: one, **{3+i: b for i,b in enumerate(bits)}}, one)
    cursor_valid, cursor_bits = cursor[0], cursor[1:]
    queries = {slot: c.and_([addressed, cursor_valid, *[
        b if slot >> i & 1 else one-b for i,b in enumerate(cursor_bits)]], f"select{slot}")
        for slot in range(p.CAPACITY)}
    # Read-only installed forest. Reads and navigation happen in weights; installation is
    # a verified heart-to-cache transfer, like loading Qwenputer's firmware/RAM snapshot.
    record = c.slot_head("records", {}, queries, {col: 0*one for col in range(2,22)}, 0*one)
    records_head = next(n for n in c.nodes if isinstance(n, Head) and n.name == "records")
    found, triangle = record[:2]
    apex, left, right = record[2:8], record[8:14], record[14:20]
    # Separate non-decaying ballast and decaying data rows retain exact powers of two.
    (counter,) = c.head("step_counter", {0: K*(one-op["open"]), 1: K*(one-op["open"])},
        {0: K*op["open"], 1: K*(one-op["open"])},
        {0: A_REF*(one-op["open"]), 1: A_REF*(one-op["open"]), 2: A_BIT*op["open"]},
        110*step+238*op["open"], {2: K*one}, [(2, 2.0**7)], tiny=2.0**-40)
    exceeded = c.node(c.t_lt(2**(p.MAX_STEPS+1)*counter, 2, 0, 512), "limit")
    valid_step = c.and_([step, found, triangle, one-exceeded], "valid_step")
    selected = [c.node(c.t_and([op["left"], l])+c.t_and([op["right"], r]), f"selected{i}")
                for i,(l,r) in enumerate(zip(left,right))]
    alternate = [c.node(c.t_and([op["left"], r])+c.t_and([op["right"], l]), f"alternate{i}")
                 for i,(l,r) in enumerate(zip(left,right))]
    pending = c.slot_head("pending", {0: valid_step}, {0: op["alt"]},
                          {2: one, **{3+i:b for i,b in enumerate(selected)}}, one)
    # One slot per allowed descent. OPEN clears routes; alternate routes survive a chat reset.
    positions = []
    for i in range(p.MAX_STEPS):
        # Counter equals 2^-(i+1) on the corresponding step. Threshold bands are disjoint.
        low = one-c.node(c.t_lt(2**(i+2)*counter, 2, 0, 512), f"step_ge{i}")
        high = one-c.node(c.t_lt(2**(i+1)*counter, 2, 0, 512), f"step_gt{i}")
        positions.append(c.and_([valid_step, low, one-high], f"step_slot{i}"))
    routes = c.slot_head("routes", dict(enumerate(positions)),
        {i: f(f"route{i}", lambda t,i=i: int(t == p.ROUTE+i)) for i in range(p.MAX_STEPS)},
        {2: one, **{3+i:b for i,b in enumerate(alternate)}}, one, a=238*op["open"])
    c.outputs = dict(found=found, triangle=triangle, apex=apex, left=left, right=right,
                     cursor=cursor_bits, counter=counter)
    cases = [
        ([addressed, one-found], p.INVALID),
        ([addressed-step, found], [*apex, 1, 1, 0]),
        ([step, found, one-triangle], p.LEAF),
        ([step, found, triangle, exceeded], p.LIMIT),
        ([valid_step], [*alternate, 0, 1, 0]),
        ([op["alt"], pending[0]], [*pending[1:], 1, 0, 0]),
        ([op["alt"], one-pending[0]], p.INVALID),
        ([op["route"], routes[0]], [*routes[1:], 1, 0, 0]),
        ([op["route"], one-routes[0]], p.INVALID),
        ([one-addressed-op["alt"]-op["route"]], p.READY),
    ]
    nxt = c.select_bits(cases, 9, "next")
    for i in range(9):
        c.ffn([(8192,one,nxt[i]),(-4096,one,one),(-1,one,code[i]),(1,one,-code[i])],
              f"exit.code{i}", into=code[i])
    c.code = code
    c.keep = {next(iter(e.terms)) for e in code}
    c.records_head = records_head
    return c


def install(circuit, forest):
    rows = {}
    for slot, r in forest.records.items():
        row = np.zeros(128)
        row[:2] = A_REF
        values = [1, int(r.triangle)] + [(v >> i)&1 for v in (r.apex,r.left,r.right) for i in range(6)]
        row[2:22] = np.array(values)*A_BIT
        rows[slot] = row
    circuit.records_head.init = [rows]


def predicted(circuit, matrix):
    from chipformer.q35.ir import ev
    return sum(int(ev(e, matrix[-1:])[0] > 0) << i for i,e in enumerate(circuit.code))

