# scripts

**Run this one:** `bench.sh` is self-contained. It starts `llama-server`, warms the expert cache, generates, and prints decode t/s. See the top of the file for the variables (`MODEL`, `BIN`, `CACHE_MIB`, `NCMOE`).

**Everything else is lab notebook, specific to my machine.** `window*.sh` take a lease on a GPU broker that only exists on my server and hard-code my paths. `det.py`, `needle60k.py`, `ctx31k.py` and `ppl.txt` are the determinism, long-context and perplexity probes they call. Read them for what was measured, adapt them if you want to repeat it.
