"""Pick the smallest policy that is reliable enough and export it to the web.

Rule: among DAgger-trained models, the fewest parameters whose closed-loop
success rate (results/evaluation.json) is >= TARGET; if none, the best one.
Writes src/robot/policyWeights.json (base64 float32), src/robot/policyMeta.json
and results/benchmarks.json.
Run: python training/export.py [model_name]
"""

import base64
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from policies import ACT_ENSEMBLE_M, ALPHA_BAR, DDIM_STEPS, E, H, TEMB

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TARGET = 0.97
OUT = ROOT / "src" / "robot" / "policyWeights.json"
META = ROOT / "src" / "robot" / "policyMeta.json"


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a, dtype="<f4").tobytes()).decode()


def pick(ev):
    cands = {k: v for k, v in ev["models"].items() if not k.endswith("_bc")}
    good = [k for k, v in cands.items() if v["success_rate"] >= TARGET]
    if good:
        return min(good, key=lambda k: (cands[k]["params"], -cands[k]["success_rate"]))
    return max(cands, key=lambda k: cands[k]["success_rate"])


def mlp_layers(sd, prefix=""):
    out = []
    for i in (0, 2, 4):
        W, b = sd[f"{prefix}{i}.weight"].numpy(), sd[f"{prefix}{i}.bias"].numpy()
        out.append({"in": int(W.shape[1]), "out": int(W.shape[0]), "w": b64(W), "b": b64(b)})
    return out


def main():
    ev = json.loads((ROOT / "results" / "evaluation.json").read_text())
    name = sys.argv[1] if len(sys.argv) > 1 else pick(ev)
    ck = torch.load(HERE / "models" / f"{name}.pt", weights_only=False)
    family = ck.get("type", "mlp")
    sd = ck["state_dict"]
    doc = {"type": family, "features": ck["kind"]}
    if family == "mlp":
        doc["layers"] = mlp_layers(sd)
        dims = [doc["layers"][0]["in"]] + [l["out"] for l in doc["layers"]]
        arch = " → ".join(map(str, dims))
    elif family == "diffusion":
        doc["layers"] = mlp_layers(sd, "net.")
        doc.update(H=H, E=E, temb=TEMB, alpha_bar=ALPHA_BAR, ddim_steps=DDIM_STEPS)
        dims = [doc["layers"][0]["in"]] + [l["out"] for l in doc["layers"]]
        arch = " → ".join(map(str, dims))
    else:  # act: every tensor except the training-only CVAE encoder
        tensors = {k: {"shape": list(v.shape), "data": b64(v.numpy())} for k, v in sd.items() if not k.startswith("enc.")}
        d = int(sd["obs_proj.weight"].shape[0])
        n_layers = len({k.split(".")[1] for k in sd if k.startswith("blocks.")})
        doc.update(d=d, heads=2, n_layers=n_layers, H=H, ensemble_m=ACT_ENSEMBLE_M, tensors=tensors)
        arch = f"transformer, {n_layers} layers, d={d}, 2 heads"
    params = sum(v.numel() for k, v in sd.items() if not k.startswith("enc.") and k not in ("temb", "ab"))
    r = ev["models"][name]
    log = {}
    for f in ("train_log.json", "train_log_chunked.json"):
        if (HERE / "models" / f).exists():
            log.update(json.loads((HERE / "models" / f).read_text()))
    log = log.get(name, {})
    pc = ev.get("pymunk_crosscheck", {})
    doc["meta"] = {
        "model": name,
        "type": family,
        "architecture": arch,
        "params": params,
        "weight_bytes": params * 4,
        "success_rate": round(r["success_rate"], 4),
        "eval_episodes": ev["n_episodes"],
        "median_steps": r["median_steps_to_success"],
        "features": ck["kind"],
        "dagger_rounds": log.get("dagger_rounds", 0),
        "train_samples": log.get("train_samples", 0),
        "pymunk_success_rate": round(pc[name]["policy"]["success_rate"], 4) if name in pc else None,
        "exported": time.strftime("%Y-%m-%d"),
    }
    text = json.dumps(doc, separators=(",", ":"))
    OUT.write_text(text)
    # metadata alone, so the page copy can show it without loading the weights
    META.write_text(json.dumps(doc["meta"], indent=2, ensure_ascii=False) + "\n")
    bench = json.loads((ROOT / "results" / "benchmarks.json").read_text()) if (ROOT / "results" / "benchmarks.json").exists() else {}
    bench.update({
        "model": name,
        "type": family,
        "params": params,
        "raw_float32_bytes": params * 4,
        "json_bytes": len(text.encode()),
        "json_gzip_bytes": len(gzip.compress(text.encode(), 9)),
        "success_rate": r["success_rate"],
    })
    (ROOT / "results" / "benchmarks.json").write_text(json.dumps(bench, indent=2))
    print(json.dumps({k: v for k, v in bench.items()}, indent=2))


if __name__ == "__main__":
    main()
