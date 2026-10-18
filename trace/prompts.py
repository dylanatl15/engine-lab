#!/usr/bin/env python3
"""Send a varied agent-like workload to a local llama-server; used with EXPERT_TRACE to record MoE routing."""
import json, glob, os, random, sys, time, urllib.request
URL = os.environ.get("LLAMA_URL", "http://127.0.0.1:8097")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
random.seed(7)
files = []
for root in ["~/my-server/agent-stack", "~/my-server/dylantamayo", "~/my-server/AIClass"]:
    for e in ("py", "js", "mjs", "ts", "yml", "sh", "md"):
        files += glob.glob(os.path.expanduser(root) + f"/**/*.{e}", recursive=True)
files = [f for f in sorted(set(files)) if "node_modules" not in f and ".git/" not in f and 1500 < os.path.getsize(f) < 9000]
random.shuffle(files)
generic = [
 "Explain how a B-tree differs from a LSM tree and when to pick each. Be concrete.",
 "Write a Python function that merges overlapping intervals, with tests.",
 "A train leaves at 3:15pm at 80 km/h; another leaves the same station at 4:00pm at 110 km/h in the same direction. When does the second catch up? Show your reasoning.",
 "Draft a polite email declining a vendor's contract renewal and asking for a 30-day extension.",
 "Summarize the main tradeoffs between PostgreSQL and SQLite for a small self-hosted app.",
 "Write a short story (200 words) about a lighthouse keeper who finds a message in a bottle.",
 "Translate to Spanish and French: 'The deployment finished, but two health checks are still failing.'",
 "What are the steps to rotate a TLS certificate behind nginx with zero downtime?",
 "Given the JSON {\"orders\":[{\"id\":1,\"total\":30},{\"id\":2,\"total\":45}]}, write a jq command computing the sum of totals, then explain it.",
 "Solve: integral of x*e^(2x) dx. Show the integration by parts steps.",
 "Write a SQL query returning the top 3 customers by revenue per month from tables orders(customer_id, amount, created_at).",
 "Give me a plan to debug a Node.js service whose memory grows 50MB per hour.",
 "Compare Rust and Go for writing a high-throughput TCP proxy.",
 "Write a bash one-liner that finds the 10 largest files under /var and prints sizes in human-readable form.",
 "Explain the CAP theorem with a real-world example involving a shopping cart.",
 "You are a QA agent. Given a change that adds rate limiting to /login, list 8 test cases including edge cases.",
 "Rewrite this sentence to be more concise: 'In the event that the system is unable to connect to the database, it will make an attempt to retry the connection a total of three times before giving up.'",
 "What is the difference between a mutex and a semaphore? Include a small code example in C++.",
]
tools = [{"type":"function","function":{"name":"read_file","description":"Read a file","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}},
         {"type":"function","function":{"name":"run_shell","description":"Run a shell command","parameters":{"type":"object","properties":{"cmd":{"type":"string"}},"required":["cmd"]}}}]
jobs = []
for i in range(N):
    k = i % 4
    if k == 0 and files:
        f = files[i % len(files)]
        jobs.append(([{"role": "user", "content": f"File {f}:\n```\n{open(f, errors='ignore').read()}\n```\nReview this file: list any bugs or risks and suggest one improvement."}], None))
    elif k == 1 and files:
        f = files[(i * 3) % len(files)]
        jobs.append(([{"role": "system", "content": "You are an autonomous engineering agent. Use tools when needed."}, {"role": "user", "content": f"Check what {f} does and whether its tests pass, then report."}], tools))
    else:
        jobs.append(([{"role": "user", "content": generic[i % len(generic)]}], None))
def post(body):
    r = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=3600))
tot = 0
for i, (m, t) in enumerate(jobs):
    b = {"messages": m, "max_tokens": int(os.environ.get("MAXTOK", "350")), "temperature": 0.7}
    if t: b["tools"] = t
    t0 = time.time(); r = post(b); tm = r.get("timings", {}); tot += tm.get("predicted_n", 0)
    print(f"{i+1}/{len(jobs)} {time.time()-t0:.0f}s pp_n={tm.get('prompt_n')} tg_n={tm.get('predicted_n')} tg={tm.get('predicted_per_second',0):.1f} total_decode={tot}", flush=True)
