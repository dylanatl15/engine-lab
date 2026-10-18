import json,sys,urllib.request,time
url="http://127.0.0.1:8098/v1/chat/completions"
b={"messages":[{"role":"user","content":"Explain in four sentences how a hash map handles collisions."}],"max_tokens":int(__import__("os").environ.get("DT","80")),"temperature":0,"seed":1,"cache_prompt":False,"chat_template_kwargs":{"enable_thinking":False}}
r=urllib.request.Request(url,json.dumps(b).encode(),{"Content-Type":"application/json"})
t=time.time(); j=json.load(urllib.request.urlopen(r,timeout=900))
print(json.dumps({"text":j["choices"][0]["message"]["content"],"tg":j["timings"]["predicted_per_second"],"n":j["timings"]["predicted_n"]}))
