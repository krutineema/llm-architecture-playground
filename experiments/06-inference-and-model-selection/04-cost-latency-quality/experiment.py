import csv, os
from datetime import datetime
from pathlib import Path
from statistics import mean

BASE=Path(__file__).resolve().parent; RESULTS=BASE/"results"; RESULTS.mkdir(exist_ok=True)
#SOURCE=Path(os.getenv("INPUT_CSV",""))
#if not SOURCE:
#   candidates=sorted((BASE.parent/"03-model-selection"/"results").glob("results_*.csv"))
#    if not candidates: raise FileNotFoundError("Run 03-model-selection first.")
#    SOURCE=candidates[-1]

input_csv = os.getenv("INPUT_CSV")

if input_csv:
    SOURCE = Path(input_csv)
else:
    candidates = sorted(
        (BASE.parent / "03-model-selection" / "results")
        .glob("results_*.csv")
    )

    if not candidates:
        raise FileNotFoundError(
            "Run 03-model-selection first or set INPUT_CSV."
        )

    SOURCE = candidates[-1]
    
MONTHLY=int(os.getenv("MONTHLY_REQUESTS","100000"))
IN_PRICE=float(os.getenv("INPUT_PRICE_PER_MILLION","1.0"))
OUT_PRICE=float(os.getenv("OUTPUT_PRICE_PER_MILLION","3.0"))
QUALITY=float(os.getenv("QUALITY_THRESHOLD","1.5"))

def nums(rows,k):
    out=[]
    for r in rows:
        try: out.append(float(r[k]))
        except: pass
    return out
def main():
    rows=list(csv.DictReader(SOURCE.open(encoding="utf-8"))); stamp=datetime.now().strftime("%Y%m%d_%H%M%S"); summary=[]
    for model in sorted({r["model"] for r in rows if r.get("model")}):
        rs=[r for r in rows if r.get("model")==model and not r.get("error")]
        ins=nums(rs,"input_tokens"); outs=nums(rs,"output_tokens"); lats=nums(rs,"latency_seconds"); tps=nums(rs,"output_tokens_per_second"); qs=nums(rs,"quality_score_manual")
        ai=mean(ins) if ins else 0; ao=mean(outs) if outs else 0; al=mean(lats) if lats else 0; at=mean(tps) if tps else 0; aq=mean(qs) if qs else None
        mi=ai*MONTHLY; mo=ao*MONTHLY; cost=mi/1e6*IN_PRICE+mo/1e6*OUT_PRICE
        summary.append({"timestamp":stamp,"model":model,"monthly_requests":MONTHLY,"avg_input_tokens":round(ai,2),"avg_output_tokens":round(ao,2),"avg_latency_seconds":round(al,4),"avg_output_tokens_per_second":round(at,2),"avg_quality_manual":round(aq,3) if aq is not None else "","monthly_input_tokens":round(mi),"monthly_output_tokens":round(mo),"estimated_monthly_cost":round(cost,2),"quality_threshold_met":aq>=QUALITY if aq is not None else ""})
    cp=RESULTS/f"results_{stamp}.csv"; tp=RESULTS/f"results_{stamp}.txt"
    with cp.open("w",newline="",encoding="utf-8") as f: w=csv.DictWriter(f,fieldnames=summary[0]); w.writeheader(); w.writerows(summary)
    with tp.open("w",encoding="utf-8") as f:
        f.write(f"Source: {SOURCE}\nMonthly requests: {MONTHLY:,}\nInput price/1M: {IN_PRICE}\nOutput price/1M: {OUT_PRICE}\nQuality threshold: {QUALITY}\n\n")
        [f.write(f"{r}\n") for r in summary]
    print(cp); print(tp)
if __name__=="__main__": main()
