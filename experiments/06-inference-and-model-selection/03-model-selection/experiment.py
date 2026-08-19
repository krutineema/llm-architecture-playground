import csv, os, time
from datetime import datetime
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE=Path(__file__).resolve().parent; RESULTS=BASE/"results"; RESULTS.mkdir(exist_ok=True)
DEFAULT="Qwen/Qwen2.5-0.5B-Instruct,Qwen/Qwen2.5-1.5B-Instruct,Qwen/Qwen2.5-3B-Instruct"
MODELS=[x.strip() for x in os.getenv("MODEL_NAMES",DEFAULT).split(",") if x.strip()]
MAX_NEW=int(os.getenv("MAX_NEW_TOKENS","160"))
CASES=[
("classification","Classify as LOW, MEDIUM, or HIGH complexity: Summarise a two-page policy."),
("classification","Classify as LOW, MEDIUM, or HIGH complexity: Design a multi-region AI platform."),
("classification","Classify as LOW, MEDIUM, or HIGH complexity: Extract a customer name from a document."),
("classification","Classify as LOW, MEDIUM, or HIGH complexity: Decide whether a transaction should require human approval."),
("extraction","Extract company, country and employee count: Example Bank, UK, 12000 employees."),
("extraction","Extract product and sector: Payment Platform, Financial Services."),
("extraction","Extract the risk level: The transaction requires manager approval."),
("extraction","Extract the date: Policy review completed on 18 August 2026."),
("reasoning","Why might RAG be preferable to fine-tuning for frequently changing policy documents?"),
("reasoning","Why can adding more agents increase latency and failure modes?"),
("reasoning","Why is the most capable model not necessarily the best production model?"),
("reasoning","Explain the difference between latency and throughput."),
("summarisation","Summarise: An AI gateway authenticates requests, applies quotas, routes requests to models, records telemetry, and provides fallback handling."),
("summarisation","Summarise: RAG retrieves relevant external information and places it into model context before generation."),
("summarisation","Summarise: Model routing can reduce cost by sending simple requests to smaller models while reserving larger models for complex tasks.")
]
def load(name):
    tok=AutoTokenizer.from_pretrained(name); mdl=AutoModelForCausalLM.from_pretrained(name,torch_dtype="auto",device_map="auto"); mdl.eval(); return tok,mdl
def fmt(tok,p):
    return tok.apply_chat_template([{"role":"user","content":p}],tokenize=False,add_generation_prompt=True) if getattr(tok,"chat_template",None) else p
def run(tok,mdl,p):
    x=tok(fmt(tok,p),return_tensors="pt"); dev=next(mdl.parameters()).device; x={k:v.to(dev) for k,v in x.items()}; n=x["input_ids"].shape[1]
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t=time.perf_counter()
    with torch.inference_mode(): y=mdl.generate(**x,max_new_tokens=MAX_NEW,do_sample=False,pad_token_id=tok.eos_token_id)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    lat=time.perf_counter()-t; o=max(y.shape[1]-n,0); return n,o,lat,tok.decode(y[0][n:],skip_special_tokens=True)
def main():
    stamp=datetime.now().strftime("%Y%m%d_%H%M%S"); rows=[]
    for name in MODELS:
        print("Loading",name)
        try: tok,mdl=load(name)
        except Exception as e:
            rows.append({"timestamp":stamp,"model":name,"case_id":"","task":"","prompt":"","input_tokens":"","output_tokens":"","latency_seconds":"","output_tokens_per_second":"","quality_score_manual":"","output":"","error":f"MODEL LOAD ERROR: {type(e).__name__}: {e}"}); continue
        for i,(task,p) in enumerate(CASES,1):
            try:
                n,o,lat,out=run(tok,mdl,p); err=""
            except Exception as e: n=o=lat=""; out=""; err=f"{type(e).__name__}: {e}"
            rows.append({"timestamp":stamp,"model":name,"case_id":i,"task":task,"prompt":p,"input_tokens":n,"output_tokens":o,"latency_seconds":round(lat,4) if lat!="" else "","output_tokens_per_second":round(o/lat,2) if lat else "","quality_score_manual":"","output":out.replace("\n"," "),"error":err})
        del mdl,tok
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    cp=RESULTS/f"results_{stamp}.csv"; tp=RESULTS/f"results_{stamp}.txt"
    with cp.open("w",newline="",encoding="utf-8") as f: w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
    with tp.open("w",encoding="utf-8") as f:
        f.write("Model Selection Shootout\n"); f.write("IMPORTANT: manually score quality_score_manual using a task-specific rubric.\n\n")
        [f.write(f"{r}\n") for r in rows]
    print(cp); print(tp)
if __name__=="__main__": main()
