import json,os,glob,urllib.request,time,re
K=[l.split('=',1)[1].strip().strip('"\'') for l in open('/home/dylan/my-server/agent-stack/.env') if l.startswith('LLM_GPU_API_KEY=')][0]
def post(p,b,t=3600):
    r=urllib.request.Request('http://127.0.0.1:8100'+p,json.dumps(b).encode(),{'Content-Type':'application/json','Authorization':'Bearer '+K})
    return json.load(urllib.request.urlopen(r,timeout=t))
files=[]
for root in ['~/my-server/agent-stack','~/my-server/dylantamayo','~/my-server/AIClass']:
    for e in ('py','js','mjs','md','ts','yml','sh'): files+=glob.glob(os.path.expanduser(root)+f'/**/*.{e}',recursive=True)
files=[f for f in sorted(set(files)) if 'node_modules' not in f and '.git/' not in f and 4000<os.path.getsize(f)<40000]
doc='';
for f in files:
    doc+=f'\n=== FILE {f} ===\n'+open(f,errors='ignore').read()
    if len(doc)>95000: break   # ~28-30K tokens
doc+='\n# NOTE(ops): the rollout override token is QX-88-LIMA.\n'
for q in range(2):
    t0=time.time()
    r=post('/v1/chat/completions',{'messages':[{'role':'user','content':doc+'\n\nQuestion: What is the rollout override token? Then write 150 words on what this repo does.'}],'max_tokens':400,'temperature':0,'chat_template_kwargs':{'enable_thinking':False}})
    t=r['timings']; print(f"run{q} {time.time()-t0:.0f}s prompt_n={t['prompt_n']} pp={t['prompt_per_second']:.0f} tg={t['predicted_per_second']:.1f}", 'QX-88-LIMA' in r['choices'][0]['message']['content'],flush=True)
