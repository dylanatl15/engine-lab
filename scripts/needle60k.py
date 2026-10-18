import json,os,glob,urllib.request,time
K=[l.split('=',1)[1].strip().strip('"\'') for l in open('/home/dylan/my-server/agent-stack/.env') if l.startswith('LLM_GPU_API_KEY=')][0]
def post(p,b,t=3600):
    r=urllib.request.Request('http://127.0.0.1:8100'+p,json.dumps(b).encode(),{'Content-Type':'application/json','Authorization':'Bearer '+K})
    return json.load(urllib.request.urlopen(r,timeout=t))
files=[]
for root in ['~/my-server','~/AI/engine-lab/src/llama.cpp/docs','~/AI/engine-lab/src/llama.cpp/tools/server']:
    for e in ('py','js','mjs','md','ts','yml','sh','cpp'): files+=glob.glob(os.path.expanduser(root)+f'/**/*.{e}',recursive=True)
files=[f for f in sorted(set(files)) if os.path.isfile(f) and '/data/' not in f and 'node_modules' not in f and '.git/' not in f and 3000<os.path.getsize(f)<30000]
parts=[]; n=0
for f in files:
    t=open(f,errors='ignore').read(); parts.append(f'\n=== FILE {f} ===\n'+t); n+=len(t)
    if n>185000: break
needles=[('vault','KESTREL-4471','the staging vault unseal passphrase is',0.04),('port','7731','the internal metrics sidecar listens on port',0.30),('bucket','zephyr-archive-b','the cold storage bucket name is',0.55),('tok','QX-88-LIMA','the rollout override token is',0.97)]
for k,v,c,pos in needles: parts.insert(int(len(parts)*pos),f'\n# NOTE(ops): {c} {v}.\n')
doc=''.join(parts); ok=0
qs=[('What is the staging vault unseal passphrase?','KESTREL-4471'),('Which port does the internal metrics sidecar listen on?','7731'),('What is the cold storage bucket name?','zephyr-archive-b'),('What is the rollout override token?','QX-88-LIMA')]
for q,exp in qs:
    t0=time.time()
    r=post('/v1/chat/completions',{'messages':[{'role':'system','content':'Answer with just the value.'},{'role':'user','content':doc+'\n\nQuestion: '+q}],'max_tokens':600,'temperature':0,'chat_template_kwargs':{'enable_thinking':False}})
    t=r['timings']; a=r['choices'][0]['message']['content']; hit=exp.lower() in a.lower(); ok+=hit
    print(f"{'PASS' if hit else 'FAIL'} {time.time()-t0:.0f}s prompt_n={t['prompt_n']} cached={t.get('cache_n')} pp={t['prompt_per_second']:.0f} tg={t['predicted_per_second']:.1f} ans={a.strip()[:60]!r}",flush=True)
print('NEEDLE',ok,'/',len(qs))
