#!/usr/bin/env python3
# eviction policy simulation on a decode routing trace: slots shared across all layers (global pool) vs per-layer
import sys, numpy as np, collections, heapq
raw = np.fromfile(sys.argv[1], dtype=np.uint16); i = 0; dec = collections.defaultdict(list)
while i + 3 <= len(raw):
    L, nt, nu = int(raw[i]), int(raw[i+1]), int(raw[i+2]); i += 3
    if i + nt*nu > len(raw): break
    ids = raw[i:i+nt*nu].reshape(nt, nu); i += nt*nu
    if nt == 1: dec[L].append(ids[0].copy())
layers = sorted(dec); T = min(len(dec[L]) for L in layers); NL = len(layers)
# access sequence in execution order: token-major, layer-minor
seq = [(L, int(e)) for t in range(T) for L in layers for e in dec[L][t]]
n = len(seq)
def lru(cap):
    c = collections.OrderedDict(); h = 0
    for k in seq:
        if k in c: h += 1; c.move_to_end(k)
        else:
            c[k] = 1
            if len(c) > cap: c.popitem(last=False)
    return h/n
def belady(cap):
    nxt = {}; nx = [0]*n
    for j in range(n-1, -1, -1): nx[j] = nxt.get(seq[j], 1<<60); nxt[seq[j]] = j
    c = {}; heap = []; h = 0
    for j, k in enumerate(seq):
        if k in c: h += 1
        elif len(c) >= cap:
            while True:
                _, kk = heapq.heappop(heap)
                if kk in c and c[kk] == -_ : break
            del c[kk]
        c[k] = nx[j]; heapq.heappush(heap, (-nx[j], k))
    return h/n
def lfu_lru(cap, decay_every):  # frequency with aging; evict min score
    c = {}; h = 0
    for j, k in enumerate(seq):
        if k in c: h += 1; c[k][0] += 1; c[k][1] = j
        else:
            if len(c) >= cap:
                v = min(c, key=lambda x: (c[x][0], c[x][1])); del c[v]
            c[k] = [1, j]
        if j % decay_every == 0:
            for x in c.values(): x[0] *= 0.5
    return h/n
def slru(cap, prot=0.5):  # segmented LRU: promote on second hit
    pc = int(cap*prot); a = collections.OrderedDict(); b = collections.OrderedDict(); h = 0
    for k in seq:
        if k in b: h += 1; b.move_to_end(k)
        elif k in a:
            h += 1; del a[k]; b[k] = 1
            if len(b) > pc: o, _ = b.popitem(last=False); a[o] = 1
        else:
            a[k] = 1
        while len(a) + len(b) > cap:
            if a: a.popitem(last=False)
            else: b.popitem(last=False)
    return h/n
for per in (48, 64, 96):
    cap = per*NL
    print(f"{per}/layer avg ({cap} slots): LRU {lru(cap)*100:.1f}%  SLRU50 {slru(cap)*100:.1f}%  SLRU25 {slru(cap,0.25)*100:.1f}%  LFU(decay5k) {lfu_lru(cap,5000)*100:.1f}%  Belady {belady(cap)*100:.1f}%")
