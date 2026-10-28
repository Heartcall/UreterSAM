import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from ureter_semis.experiment import atomic_json,sha256

if __name__=="__main__":
    config=json.loads((ROOT/"configs/diagnose_refinement_roi_20261028.json").read_text())
    output=ROOT/config["output"]/"matched_costs"
    output.mkdir()
    sources=[ROOT/"tools/benchmark_refinement_proxy.py",ROOT/"tools/diagnose_refinement_roi.py",ROOT/"tools/refinement_roi.py"]
    identity={str(p.relative_to(ROOT)):sha256(p) for p in sources}
    atomic_json(output/"preflight.json",{"sources":identity,"reason":"Repeat after tests completed; identical PIL/native-output full-frame references, no concurrent project tests",
        "names":["full256","full512"]+[p["name"] for p in config["profiles"]]})
    results={}
    for name in ["full256","full512"]+[p["name"] for p in config["profiles"]]:
        atomic_json(output/"job.json",{"phase":"running","name":name,"pid":os.getpid()})
        with (output/f"{name}.log").open("w",encoding="utf-8") as log:
            subprocess.run([sys.executable,"-m","tools.benchmark_refinement_proxy","--name",name,"--output",str(output/f"{name}.json")],cwd=ROOT,
                stdout=log,stderr=subprocess.STDOUT,check=True,creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
        results[name]=json.loads((output/f"{name}.json").read_text())
    assert all(sha256(ROOT/p)==h for p,h in identity.items())
    atomic_json(output/"comparison.json",results)
    atomic_json(output/"job.json",{"phase":"completed","pid":os.getpid(),"variants":len(results)})
    print(json.dumps({n:{k:round(v,2) for k,v in r["timings"]["total_ms"].items()} for n,r in results.items()}))
