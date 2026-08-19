import csv, os, time
from datetime import datetime
from pathlib import Path
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE=Path(__file__).resolve().parent; RESULTS=BASE/"results"; RESULTS.mkdir(exist_ok=True)
MODEL=os.getenv("MODEL_NAME","Qwen/Qwen2.5-0.5B-Instruct"); RUNS=int(os.getenv("RUNS","3")); MAX_NEW=int(os.getenv("MAX_NEW_TOKENS","120"))
PROMPTS={
"factual":"Explain in two sentences what an AI gateway does.",
"structured_extraction":"Extract these fields as JSON: Company: Example Bank; Country: UK; Employees: 12000; Product: Payment Platform.",
"classification":"Classify this request as LOW, MEDIUM, or HIGH complexity: Design an enterprise architecture for an AI agent that can access customer records and approve financial transactions.",
"creative":"Write a short imaginative analogy explaining model routing to a software architect."
}
CONFIGS=[
("greedy",False,None,None,None),
("temperature_0_3",True,0.3,None,None),
("temperature_0_7",True,0.7,None,None),
("temperature_1_0",True,1.0,None,None),
("top_k_20",True,0.7,20,None),
("top_p_0_8",True,0.7,None,0.8)
]

def load():
    tok=AutoTokenizer.from_pretrained(MODEL); mdl=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype="auto",device_map="auto"); mdl.eval(); return tok,mdl
def fmt(tok,text):
    return tok.apply_chat_template([{"role":"user","content":text}],tokenize=False,add_generation_prompt=True) if getattr(tok,"chat_template",None) else text
def run(tok,mdl,text,do_sample,temp,topk,topp):
    x=tok(fmt(tok,text),return_tensors="pt"); dev=next(mdl.parameters()).device; x={k:v.to(dev) for k,v in x.items()}; n=x["input_ids"].shape[1]
    kw={"max_new_tokens":MAX_NEW,"do_sample":do_sample,"pad_token_id":tok.eos_token_id}
    if temp is not None: kw["temperature"]=temp
    if topk is not None: kw["top_k"]=topk
    if topp is not None: kw["top_p"]=topp
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t=time.perf_counter()
    with torch.inference_mode(): y=mdl.generate(**x,**kw)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    lat=time.perf_counter()-t; o=max(y.shape[1]-n,0)
    return n,o,lat,tok.decode(y[0][n:],skip_special_tokens=True)
def main():
    tok,mdl=load(); stamp=datetime.now().strftime("%Y%m%d_%H%M%S"); rows=[]
    for pn,p in PROMPTS.items():
        for name,ds,temp,topk,topp in CONFIGS:
            for r in range(1,RUNS+1):
                try:
                    n,o,lat,out=run(tok,mdl,p,ds,temp,topk,topp)
                    err=""
                except Exception as e: n=o=lat=""; out=""; err=f"{type(e).__name__}: {e}"
                rows.append({"timestamp":stamp,"model":MODEL,"prompt_name":pn,"configuration":name,"run":r,"input_tokens":n,"output_tokens":o,"latency_seconds":round(lat,4) if lat!="" else "","output_tokens_per_second":round(o/lat,2) if lat else "","output":out.replace("\n"," "),"error":err})
    cp=RESULTS/f"results_{stamp}.csv"; tp=RESULTS/f"results_{stamp}.txt"
    with cp.open("w",newline="",encoding="utf-8") as f: w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
    with tp.open("w",encoding="utf-8") as f:
        f.write(f"Decoding Behaviour\nModel: {MODEL}\nTimestamp: {stamp}\n\n"); [f.write(f"{r}\n") for r in rows]
    print(cp); print(tp)
if __name__=="__main__": main()
