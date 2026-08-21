import csv, os, time
from datetime import datetime
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE=Path(__file__).resolve().parent; RESULTS=BASE/"results"; RESULTS.mkdir(exist_ok=True)
MODELS={"small":os.getenv("SMALL_MODEL","Qwen/Qwen2.5-0.5B-Instruct"),"medium":os.getenv("MEDIUM_MODEL","Qwen/Qwen2.5-1.5B-Instruct"),"large":os.getenv("LARGE_MODEL","Qwen/Qwen2.5-3B-Instruct")}
MAX_NEW=int(os.getenv("MAX_NEW_TOKENS","160"))
REQUESTS=[
("simple","Extract the country: Example Bank operates in the UK."),
("simple","Does RAG retrieve external information? Answer YES or NO."),
("simple","Summarise in one sentence: Model routing sends different requests to different models."),
("normal","Explain how RAG works in an enterprise application and identify three architecture trade-offs."),
("normal","Explain latency vs throughput for an AI service and why both matter."),
("normal","Explain when an AI gateway is useful in enterprise architecture."),
("complex","Design an enterprise architecture for an AI agent with private-document retrieval, business APIs, human approval, model fallback and observability. Explain trade-offs."),
("complex","Evaluate one large model versus model routing for classification, summarisation and complex reasoning. Discuss cost, latency, quality and operational complexity."),
("complex","Explain how RAG, tool calling, model routing and human-in-the-loop controls can be combined in an enterprise AI system.")
]
def load(mid):
    tok=AutoTokenizer.from_pretrained(mid); mdl=AutoModelForCausalLM.from_pretrained(mid,torch_dtype="auto",device_map="auto"); mdl.eval(); return tok,mdl
def run(tok,mdl,p):
    text=tok.apply_chat_template([{"role":"user","content":p}],tokenize=False,add_generation_prompt=True) if getattr(tok,"chat_template",None) else p
    x=tok(text,return_tensors="pt"); dev=next(mdl.parameters()).device; x={k:v.to(dev) for k,v in x.items()}; n=x["input_ids"].shape[1]
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t=time.perf_counter()
    with torch.inference_mode(): y=mdl.generate(**x,max_new_tokens=MAX_NEW,do_sample=False,pad_token_id=tok.eos_token_id)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    lat=time.perf_counter()-t; o=max(y.shape[1]-n,0)
    return n,o,lat,tok.decode(y[0][n:],skip_special_tokens=True)
def architecture(name,loaded):
    rows=[]
    for i,(complexity,p) in enumerate(REQUESTS,1):
        role="large" if name=="single_large_model" else {"simple":"small","normal":"medium","complex":"large"}[complexity]
        tok,mdl=loaded[role]; n,o,lat,out=run(tok,mdl,p)
        rows.append({"architecture":name,"request_id":i,"complexity":complexity,"selected_model_role":role,"selected_model":MODELS[role],"input_tokens":n,"output_tokens":o,"latency_seconds":round(lat,4),"output_tokens_per_second":round(o/lat,2) if lat else 0,"quality_score_manual":"","routing_correct_manual":"","output":out.replace("\n"," ")})
    return rows
def main():
    stamp=datetime.now().strftime("%Y%m%d_%H%M%S"); loaded={}
    for role,mid in MODELS.items(): print("Loading",role,mid); loaded[role]=load(mid)
    rows=architecture("single_large_model",loaded)+architecture("model_routing",loaded)
    cp=RESULTS/f"results_{stamp}.csv"; tp=RESULTS/f"results_{stamp}.txt"
    with cp.open("w",newline="",encoding="utf-8") as f: w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
    with tp.open("w",encoding="utf-8") as f:
        f.write("Model Routing Experiment\nIMPORTANT: manually score quality and routing correctness.\n\n"); [f.write(f"{r}\n") for r in rows]
    print(cp); print(tp)
if __name__=="__main__": main()
