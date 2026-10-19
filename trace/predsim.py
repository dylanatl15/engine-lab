#!/usr/bin/env python3
# can layer L's experts be predicted from layer L-1's selection (same token)? co-occurrence table trained on the first half, tested on the second
import sys, numpy as np, collections
raw = np.fromfile(sys.argv[1], dtype=np.uint16); i = 0; dec = collections.defaultdict(list)
while i + 3 <= len(raw):
    L, nt, nu = int(raw[i]), int(raw[i+1]), int(raw[i+2]); i += 3
    if i + nt*nu > len(raw): break
    ids = raw[i:i+nt*nu].reshape(nt, nu); i += nt*nu
    if nt == 1: dec[L].append(ids[0].copy())
layers = sorted(dec); T = min(len(dec[L]) for L in layers); NE = 512
A = {L: np.array(dec[L][:T]) for L in layers}; half = T//2
cap = 64
res = {M: [0, 0] for M in (10, 20, 40)}
rng = np.random.default_rng(0)
for L in layers[1:]:
    C = np.zeros((NE, NE), np.float32)       # C[a,b]: a selected at L-1, b at L
    for t in range(half):
        for a in A[L-1][t]: C[a, A[L][t]] += 1
    C /= np.maximum(C.sum(1, keepdims=True), 1)
    # simulate a per-layer LRU of cap entries over the test half to find misses; predictor must name them
    cache = collections.OrderedDict()
    for t in range(half):
        for e in A[L][t].tolist():
            cache[e] = 1; cache.move_to_end(e)
            if len(cache) > cap: cache.popitem(last=False)
    for t in range(half, T):
        sc = C[A[L-1][t]].sum(0)
        miss = [e for e in A[L][t].tolist() if e not in cache]
        order = [e for e in np.argsort(-sc).tolist() if e not in cache]
        for M in res:
            top = set(order[:M]); res[M][0] += sum(e in top for e in miss); res[M][1] += len(miss)
        for e in A[L][t].tolist():
            cache[e] = 1; cache.move_to_end(e)
            if len(cache) > cap: cache.popitem(last=False)
for M, (h, n) in res.items(): print(f"predict top-{M} non-cached candidates from layer L-1 selection: recall of actual misses {h/n*100:.1f}%  ({n} misses)")
