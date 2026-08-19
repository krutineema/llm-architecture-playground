import csv, os, time
from datetime import datetime
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE=Path(__file__).resolve().parent; RESULTS=BASE/"results"; RESULTS.mkdir(exist_ok=True)
MODEL=os.getenv("MODEL_NAME","Qwen/Qwen2.5-0.5B-Instruct")
# Use actual comparable variants available in your environment.
CONFIGS={}
for item in os.getenv("MODEL_CONFIGS",f"baseline={MODEL}").split(","):
    label,mid=item.split("=",1); CONFIGS[label.strip()]=mid.strip()
RUNS=int(os.getenv("RUNS","3")); MAX_NEW=int(os.getenv("MAX_NEW_TOKENS","128"))
PROMPT="Explain the trade-off between a larger and smaller model in an enterprise AI application, covering quality, latency, cost and operations."

def load(mid):
    tok=AutoTokenizer.from_pretrained(mid); mdl=AutoModelForCausalLM.from_pretrained(mid,torch_dtype="auto",device_map="auto"); mdl.eval(); return tok,mdl
def main():
    stamp=datetime.now().strftime("%Y%m%d_%H%M%S"); rows=[]
    for label,mid in CONFIGS.items():
        print("Loading",label,mid)
        try: tok,mdl=load(mid)
        except Exception as e:
            rows.append({"timestamp":stamp,"configuration":label,"model":mid,"run":"","input_tokens":"","output_tokens":"","latency_seconds":"","output_tokens_per_second":"","peak_gpu_memory_mb":"","quality_score_manual":"","output":"","error":f"{type(e).__name__}: {e}"}); continue
        text=tok.apply_chat_template([{"role":"user","content":PROMPT}],tokenize=False,add_generation_prompt=True) if getattr(tok,"chat_template",None) else PROMPT
        x=tok(text,return_tensors="pt"); dev=next(mdl.parameters()).device; x={k:v.to(dev) for k,v in x.items()}; n=x["input_ids"].shape[1]
        with torch.inference_mode(): mdl.generate(**x,max_new_tokens=16,do_sample=False,pad_token_id=tok.eos_token_id)
        for r in range(1,RUNS+1):
            if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
            t=time.perf_counter()
            with torch.inference_mode(): y=mdl.generate(**x,max_new_tokens=MAX_NEW,do_sample=False,pad_token_id=tok.eos_token_id)
            if torch.cuda.is_available(): torch.cuda.synchronize()
            lat=time.perf_counter()-t; o=max(y.shape[1]-n,0); out=tok.decode(y[0][n:],skip_special_tokens=True)
            mem=round(torch.cuda.max_memory_allocated()/1024**2,2) if torch.cuda.is_available() else ""
            rows.append({"timestamp":stamp,"configuration":label,"model":mid,"run":r,"input_tokens":n,"output_tokens":o,"latency_seconds":round(lat,4),"output_tokens_per_second":round(o/lat,2) if lat else 0,"peak_gpu_memory_mb":mem,"quality_score_manual":"","output":out.replace("\n"," "),"error":""})
        del mdl,tok
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    cp=RESULTS/f"results_{stamp}.csv"; tp=RESULTS/f"results_{stamp}.txt"
    with cp.open("w",newline="",encoding="utf-8") as f: w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
    with tp.open("w",encoding="utf-8") as f:
        f.write("Quantisation / Precision Experiment\nIMPORTANT: compare only genuinely comparable configurations and manually score quality.\n\n"); [f.write(f"{r}\n") for r in rows]
    print(cp); print(tp)
if __name__=="__main__": main()
