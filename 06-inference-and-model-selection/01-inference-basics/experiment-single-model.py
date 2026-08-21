import csv, os, time
from datetime import datetime
from pathlib import Path
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE=Path(__file__).resolve().parent
RESULTS=BASE/"results"; RESULTS.mkdir(exist_ok=True)
MODEL=os.getenv("MODEL_NAME","Qwen/Qwen2.5-0.5B-Instruct")
RUNS=int(os.getenv("RUNS","3")); MAX_NEW=int(os.getenv("MAX_NEW_TOKENS","128"))

PROMPTS={
"short_input_short_output":"In one sentence, explain what an API gateway does.",
"short_input_long_output":"Explain the role of an API gateway in an enterprise AI architecture. Cover routing, authentication, rate limiting, observability, cost controls, retries, and model selection.",
"long_input_short_output":"Summarise this in one sentence: An enterprise application sends requests to an AI gateway. The gateway authenticates callers, applies quotas, selects an appropriate model, retrieves context, invokes tools where permitted, records telemetry, applies safety policies, and returns the model response.",
"long_input_long_output":"Explain the following architecture and its trade-offs: An enterprise application sends requests to an AI gateway. The gateway authenticates callers, applies quotas, selects an appropriate model, retrieves context, invokes tools where permitted, records telemetry, applies safety policies, and returns the model response. The system supports models with different latency, cost and capability characteristics."
}

def load():
    tok=AutoTokenizer.from_pretrained(MODEL)
    mdl=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype="auto",device_map="auto")
    mdl.eval(); return tok,mdl

def prompt(tok,text):
    if getattr(tok,"chat_template",None):
        return tok.apply_chat_template([{"role":"user","content":text}],tokenize=False,add_generation_prompt=True)
    return text

def run(tok,mdl,text):
    x=tok(prompt(tok,text),return_tensors="pt")
    dev=next(mdl.parameters()).device; x={k:v.to(dev) for k,v in x.items()}
    n=x["input_ids"].shape[1]
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t=time.perf_counter()
    with torch.inference_mode(): y=mdl.generate(**x,max_new_tokens=MAX_NEW,do_sample=False,pad_token_id=tok.eos_token_id)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    elapsed=time.perf_counter()-t; out=max(y.shape[1]-n,0)
    return n,out,elapsed,tok.decode(y[0][n:],skip_special_tokens=True)

def main():
    tok,mdl=load(); stamp=datetime.now().strftime("%Y%m%d_%H%M%S"); rows=[]
    run(tok,mdl,"Say hello in one sentence.")
    for name,text in PROMPTS.items():
        for r in range(1,RUNS+1):
            n,o,lat,out=run(tok,mdl,text)
            rows.append({"timestamp":stamp,"model":MODEL,"prompt_name":name,"run":r,"input_tokens":n,"output_tokens":o,"latency_seconds":round(lat,4),"output_tokens_per_second":round(o/lat if lat else 0,2),"output":out.replace("\n"," ")})
    csvp=RESULTS/f"results_{stamp}.csv"; txtp=RESULTS/f"results_{stamp}.txt"
    with csvp.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
    with txtp.open("w",encoding="utf-8") as f:
        f.write(f"Inference Basics\nModel: {MODEL}\nTimestamp: {stamp}\n\n")
        for r in rows: f.write(f"{r}\n")
    print(csvp); print(txtp)

if __name__=="__main__": main()
