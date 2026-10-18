#!/usr/bin/env python3
import sys, numpy as np, collections
fn = sys.argv[1] if len(sys.argv) > 1 else "routing.bin"
raw = np.fromfile(fn, dtype=np.uint16)
i = 0; dec = collections.defaultdict(list); pre = collections.defaultdict(lambda: collections.Counter()); n_used = 0
while i + 3 <= len(raw):
    L, nt, nu = int(raw[i]), int(raw[i+1]), int(raw[i+2]); i += 3
    if i + nt*nu > len(raw): break
    ids = raw[i:i+nt*nu].reshape(nt, nu); i += nt*nu; n_used = nu
    if nt == 1: dec[L].append(ids[0].copy())
    else:
        for e in ids.ravel(): pre[L][int(e)] += 1
layers = sorted(dec)
if not layers: sys.exit("no decode records yet")
NE = int(max(max(int(x.max()) for x in dec[L]) for L in layers)) + 1
NE = max(NE, 1 << (NE - 1).bit_length()) if NE > 1 else NE
T = min(len(dec[L]) for L in layers)
print(f"layers with MoE records: {len(layers)} (ids {layers[0]}..{layers[-1]}), experts/layer ~{NE}, experts used/token {n_used}, decode tokens {T}, prefill records {sum(sum(c.values()) for c in pre.values())//max(n_used,1)} tok-layers")
A = {L: np.array(dec[L][:T]) for L in layers}   # T x n_used
half = T // 2
def cover(L, K, train, test):
    cnt = np.bincount(A[L][train].ravel(), minlength=NE); top = set(np.argsort(-cnt)[:K].tolist())
    hit = np.isin(A[L][test], list(top)).sum(); return hit / A[L][test].size
print("\nstatic per-layer top-K (train on first half of decode, test on second half), hit rate of expert selections:")
for frac in (0.05, 0.10, 0.125, 0.25, 0.375, 0.5, 0.75):
    K = max(1, int(NE * frac))
    h = np.mean([cover(L, K, slice(0, half), slice(half, T)) for L in layers])
    o = np.mean([cover(L, K, slice(half, T), slice(half, T)) for L in layers])
    print(f"  cache {frac*100:5.1f}% of experts ({K:3d}/layer): held-out hit {h*100:5.1f}%   oracle {o*100:5.1f}%   random baseline {frac*100:5.1f}%")
print("\nLRU per layer (capacity as % of experts), hit rate over decode sequence:")
for frac in (0.05, 0.125, 0.25, 0.5):
    K = max(1, int(NE * frac)); hits = 0; tot = 0
    for L in layers:
        cache = collections.OrderedDict()
        for row in A[L]:
            for e in row.tolist():
                tot += 1
                if e in cache: hits += 1; cache.move_to_end(e)
                else:
                    cache[e] = 1
                    if len(cache) > K: cache.popitem(last=False)
    print(f"  LRU {frac*100:5.1f}% ({K}/layer): {hits/tot*100:5.1f}%")
ov = np.mean([len(set(A[L][t].tolist()) & set(A[L][t-1].tolist())) / n_used for L in layers for t in range(1, T)])
print(f"\nconsecutive-token expert overlap (same layer): {ov*100:.1f}%  (random would be {n_used/NE*100:.1f}%)")
ent = []
for L in layers:
    p = np.bincount(A[L].ravel(), minlength=NE).astype(float); p /= p.sum(); p = p[p > 0]; ent.append(-(p*np.log2(p)).sum())
print(f"routing entropy per layer: mean {np.mean(ent):.2f} bits of max {np.log2(NE):.2f}")
# global budget: greedy allocation of VRAM slots across layers by marginal held-out gain
cnts = {L: np.bincount(A[L][:half].ravel(), minlength=NE) for L in layers}
test = {L: np.bincount(A[L][half:].ravel(), minlength=NE) for L in layers}
order = []
for L in layers:
    for rank, e in enumerate(np.argsort(-cnts[L])): order.append((cnts[L][e], L, int(e)))
order.sort(reverse=True)
tot_sel = sum(test[L].sum() for L in layers)
print("\nglobal greedy placement (one pool of slots across all layers), held-out hit rate:")
sel = collections.defaultdict(set); k = 0
marks = [int(len(layers)*NE*f) for f in (0.05, 0.10, 0.125, 0.25, 0.375, 0.5)]
for _, L, e in order:
    sel[L].add(e); k += 1
    if k in marks:
        h = sum(test[Lx][list(sel[Lx])].sum() for Lx in layers if sel[Lx]) / tot_sel
        print(f"  {k/(len(layers)*NE)*100:5.1f}% of all experts in VRAM ({k} slots): {h*100:5.1f}%")
