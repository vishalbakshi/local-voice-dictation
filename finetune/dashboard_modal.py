"""Live dashboard for fine-tuning runs, served from Modal at a fixed URL.

  uv run modal deploy finetune/dashboard_modal.py

Reads every /vol/runs/<run>/metrics.csv (synced by train_modal.py after each
epoch) and log.json (written when a run finishes), so it shows runs live and
keeps the history. Nothing here touches training. The URL is public but only
serves loss/WER numbers and run settings, no audio or transcripts.
"""
import csv
import io
import json

import modal

image = modal.Image.debian_slim(python_version="3.12").pip_install("fastapi[standard]")
vol = modal.Volume.from_name("local-voice-dictation", create_if_missing=True)
app = modal.App("finetune-dashboard", image=image)

# Round 1 mixed in a CTC loss ~50x larger than TDT, so its losses aren't on the
# same scale as later runs; the page says so.
NOTES = {
    "full": "round 1 · loss includes CTC (not comparable)",
    "frozen": "round 1 · loss includes CTC (not comparable)",
    "smoke": "setup check",
}


def read_run(name):
    base = f"/vol/runs/{name}"
    out = {"name": name, "note": NOTES.get(name, ""), "train": [], "val": [], "log": None}
    try:
        out["log"] = json.load(open(f"{base}/log.json"))
    except OSError:
        pass
    try:
        rows = list(csv.DictReader(open(f"{base}/metrics.csv")))
    except OSError:
        return out
    num = lambda r, k: float(r[k]) if r.get(k) not in (None, "") else None
    for r in rows:
        if num(r, "train_loss") is not None:
            out["train"].append({"step": int(float(r["step"])), "loss": num(r, "train_loss"),
                                 "lr": num(r, "learning_rate")})
        # Round 3 validates on two sets: val_val_* (personal) and blog_val_* (forgetting watch).
        wer = num(r, "val_val_wer") if "val_val_wer" in r else num(r, "val_wer")
        if wer is not None:
            loss = num(r, "val_val_loss") if "val_val_loss" in r else num(r, "val_loss")
            out["val"].append({"epoch": int(r["epoch"]), "loss": loss, "wer": wer, "blog": num(r, "blog_val_wer")})
    # A finished run re-validates its best checkpoint at the end; that last row
    # isn't a training epoch, so leave it out of the curve.
    if out["log"] and len(out["val"]) > 1:
        out["val"] = out["val"][:-1]
    return out


@app.function(volumes={"/vol": vol}, min_containers=0, scaledown_window=300)
@modal.asgi_app()
def web():
    import os
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, JSONResponse

    api = FastAPI()

    @api.get("/")
    def page():
        return HTMLResponse(PAGE)

    @api.get("/api/runs")
    def runs():
        vol.reload()  # pick up epochs committed since the last request
        names = sorted(os.listdir("/vol/runs")) if os.path.isdir("/vol/runs") else []
        return JSONResponse([read_run(n) for n in names])

    return api


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fine-tuning Runs</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
:root{--bg:#fcfcfb;--fg:#0b0b0b;--fg2:#52514e;--muted:#898781;--grid:#e1e0d9;--card:#fff}
@media (prefers-color-scheme:dark){:root{--bg:#1a1a19;--fg:#fff;--fg2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--card:#222220}}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 -apple-system,system-ui,sans-serif}
main{max-width:1200px;margin:0 auto;padding:16px}
h1{font-size:18px;margin:0 0 4px}.muted{color:var(--muted);font-size:12px}
.runs{display:flex;flex-wrap:wrap;gap:8px;margin:12px 0}
.run{display:flex;gap:8px;align-items:flex-start;border:1px solid var(--grid);border-radius:8px;padding:8px 10px;background:var(--card);min-width:220px}
.sw{width:10px;height:10px;border-radius:2px;margin-top:5px;flex:none}
.run b{font-size:13px}.run div{font-size:12px;color:var(--fg2)}
.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px}
.card{border:1px solid var(--grid);border-radius:8px;padding:10px;background:var(--card)}
.card h2{font-size:13px;margin:0 0 6px;font-weight:600}
canvas{width:100%!important;height:240px!important}
</style></head><body><main>
<h1>Fine-tuning runs</h1>
<div class="muted" id="updated">loading…</div>
<div class="runs" id="runs"></div>
<div class="charts">
  <div class="card"><h2>Training loss (smoothed)</h2><canvas id="c_train"></canvas></div>
  <div class="card"><h2>Validation loss</h2><canvas id="c_vloss"></canvas></div>
  <div class="card"><h2>Validation error % (case + punctuation counted)</h2><canvas id="c_vwer"></canvas></div>
  <div class="card"><h2>Blog-read error % (forgetting watch, round 3+)</h2><canvas id="c_blog"></canvas></div>
</div>
</main>
<script>
// Categorical slots in fixed order; a run keeps its color by name.
const COLORS=["#2a78d6","#eb6834","#1baf7a","#eda100","#e87ba4","#008300","#4a3aa7","#e34948"];
const css=k=>getComputedStyle(document.documentElement).getPropertyValue(k).trim();
const shown=JSON.parse(localStorage.getItem("shown")||"null")||{};
let charts={}, colorOf={};
function ema(xs,a=0.25){let m=null;return xs.map(x=>m=(m===null?x:a*x+(1-a)*m))}
function opts(xlabel){return{animation:false,responsive:true,maintainAspectRatio:false,parsing:false,
  interaction:{mode:"nearest",axis:"x",intersect:false},
  plugins:{legend:{display:false},tooltip:{callbacks:{label:c=>`${c.dataset.label}: ${c.parsed.y.toFixed(3)}`}}},
  scales:{x:{type:"linear",title:{display:true,text:xlabel,color:css("--muted")},grid:{color:css("--grid")},ticks:{color:css("--muted")}},
          y:{grid:{color:css("--grid")},ticks:{color:css("--muted")}}}}}
function chart(id,xlabel){if(!charts[id])charts[id]=new Chart(document.getElementById(id),{type:"line",data:{datasets:[]},options:opts(xlabel)});return charts[id]}
function status(r){const blog=r.log&&r.log.blog_wer_before!==undefined?` · blog ${(r.log.blog_wer_before*100).toFixed(1)}% → ${(r.log.blog_wer_after*100).toFixed(1)}%`:"";
  if(r.log)return`finished · ${r.log.epochs_run} epochs · val error ${(r.log.val_wer_before*100).toFixed(1)}% → ${(r.log.val_wer_after*100).toFixed(1)}%${blog}`;
  const v=r.val.at(-1);return v?`running · epoch ${v.epoch} · val error ${(v.wer*100).toFixed(1)}%`:"starting…"}
async function refresh(){
  const runs=await (await fetch("api/runs")).json();
  runs.forEach((r,i)=>{if(!(r.name in colorOf))colorOf[r.name]=COLORS[Object.keys(colorOf).length%COLORS.length]; if(!(r.name in shown))shown[r.name]=r.name!=="smoke"});
  document.getElementById("runs").innerHTML=runs.map(r=>{const c=r.log?.config||{};
    const cfg=c.lr?`lr ${c.lr} · ${c.train||(c.freeze_encoder?"heads":"all")}${c.ctc_weight!==undefined?` · ctc ${c.ctc_weight}`:""}`:"";
    return`<label class="run"><input type="checkbox" data-run="${r.name}" ${shown[r.name]?"checked":""}><span class="sw" style="background:${colorOf[r.name]}"></span>
    <span><b>${r.name}</b><div>${status(r)}</div><div>${cfg}${r.note?` · ${r.note}`:""}</div></span></label>`}).join("");
  document.querySelectorAll("[data-run]").forEach(el=>el.onchange=()=>{shown[el.dataset.run]=el.checked;localStorage.setItem("shown",JSON.stringify(shown));draw(runs)});
  draw(runs);
  document.getElementById("updated").textContent=`updated ${new Date().toLocaleTimeString()} · refreshes every 30 s`;
}
function draw(runs){
  const vis=runs.filter(r=>shown[r.name]);
  const ds=(r,pts)=>({label:r.name,data:pts,borderColor:colorOf[r.name],backgroundColor:colorOf[r.name],borderWidth:2,pointRadius:pts.length<80?3:0,tension:0});
  const t=chart("c_train","step"); t.data.datasets=vis.map(r=>{const s=ema(r.train.map(p=>p.loss));return ds(r,r.train.map((p,i)=>({x:p.step,y:s[i]})))}); t.update();
  const vl=chart("c_vloss","epoch"); vl.data.datasets=vis.map(r=>ds(r,r.val.filter(p=>p.loss!==null).map(p=>({x:p.epoch,y:p.loss})))); vl.update();
  const vw=chart("c_vwer","epoch"); vw.data.datasets=vis.map(r=>ds(r,r.val.map(p=>({x:p.epoch,y:p.wer*100})))); vw.update();
  const bl=chart("c_blog","epoch"); bl.data.datasets=vis.filter(r=>r.val.some(p=>p.blog!==null)).map(r=>ds(r,r.val.filter(p=>p.blog!==null).map(p=>({x:p.epoch,y:p.blog*100})))); bl.update();
}
refresh(); setInterval(refresh,30000);
</script></body></html>"""
