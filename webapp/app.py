"""
webapp/app.py
=============
FastAPI backend for the smart-irrigation advisory tool.

Endpoints
---------
GET  /                      -> farm-facing page (static index.html)
GET  /research              -> research/admin page (status, audit, manual refine)
POST /api/recommend         -> {moisture_pct, hour, planting_date} -> recommendation
POST /api/upload            -> multipart CSV from SD card; kicks off background refine
GET  /api/status            -> current model status (version, state, last refine)
GET  /api/audit             -> recent audit-log entries (research view)

Run locally:
    cd webapp
    uvicorn app:app --reload --port 8000
Then open http://localhost:8000
"""

from __future__ import annotations

import shutil
from datetime import datetime, date, timezone
from pathlib import Path

import os, secrets
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Header
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from model_service import (ModelService, read_status, AUDIT_LOG, UPLOADS, MODELS)

HERE = Path(__file__).resolve().parent
app = FastAPI(title="Smart Irrigation Advisor (T2)")

service = ModelService()


def _days_after(planting_date: str) -> int:
    from zoneinfo import ZoneInfo
    try:
        pd = datetime.strptime(planting_date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(400, "planting_date must be YYYY-MM-DD")
    # Farm-local (IST) date; the server clock is UTC, which is one day
    # behind between midnight and 05:30 IST.
    today_ist = datetime.now(ZoneInfo("Asia/Kolkata")).date()
    return max((today_ist - pd).days, 0)


@app.get("/", response_class=HTMLResponse)
def home():
    return (HERE / "static" / "index.html").read_text()


@app.get("/research", response_class=HTMLResponse)
def research():
    return (HERE / "static" / "research.html").read_text()


@app.post("/api/upload_events")
async def api_upload_events(file: UploadFile = File(...),
                            planting_date: str = Form("2026-06-08")):
    """Upload a table of rain/irrigation events (+ optional farmer judgment).
    Returns each row with the model's recommendation alongside, for comparison.
    No admin key needed — the worker can use this."""
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Please upload a .csv events table.")
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    dest = UPLOADS / f"events_{ts}_{file.filename}"
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    res = service.process_events_csv(str(dest), planting_date)
    return JSONResponse(res)


@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...),
                     events_file: UploadFile = File(None),
                     planting_date: str = Form("2026-06-08"),
                     rained: str = Form("no"),
                     rain_hours: float = Form(0.0)):
    """Single farm action: upload the SD-card sensor CSV (and optionally an
    events-history CSV in the same submit). The app reads the latest moisture
    from the sensor file, advises for the next 9 AM / 2 PM window, and (if an
    events file is given) logs that history with model-vs-farmer comparison."""
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Please upload a .csv from the sensor node.")
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    dest = UPLOADS / f"{ts}_{file.filename}"
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # The recommendation is for the T2 (agent) plot, so it must be driven by
    # that plot's own probe — not whichever node happened to transmit last.
    # (Inter-plot probe offsets are ~4 pts, larger than the day-to-day signal.)
    t2_plot = service.cfg["actuator"].get("t2_plot_id", "P1")
    latest = service.latest_moisture_from_csv(str(dest), plot_filter=t2_plot)
    if not latest["ok"]:
        return JSONResponse({"ok": False, "error": latest["error"]})

    # today's rain toggle
    events = []
    if str(rained).lower() in ("yes", "true", "1") and rain_hours > 0:
        events.append({"type": "rain", "hours": rain_hours})
    if events:
        service.log_events(str(dest), events)

    # optional events-history file uploaded in the same submit
    events_result = None
    if events_file is not None and events_file.filename:
        if events_file.filename.lower().endswith(".csv"):
            ed = UPLOADS / f"events_{ts}_{events_file.filename}"
            with open(ed, "wb") as f:
                shutil.copyfileobj(events_file.file, f)
            events_result = service.process_events_csv(
                str(ed), planting_date, sensor_csv_path=str(dest))

    window = service.next_decision_window()
    dat = _days_after(planting_date)
    rec = service.recommend(moisture_pct=latest["moisture_pct"], hour=window,
                            days_after_transplant=dat)
    started = service.refine_async(str(dest), planting_date)
    return JSONResponse({"ok": True, "saved": dest.name,
                         "latest_reading": latest, "events_logged": events,
                         "events_history": events_result,
                         "recommendation": rec, "refine": started})


@app.post("/api/farmer_judgment")
async def api_farmer_judgment(model_minutes: int = Form(...),
                              farmer_minutes: float = Form(...),
                              moisture_pct: float = Form(...),
                              reason: str = Form(""),
                              date: str = Form(None)):
    """Record today's on-screen farmer judgment next to the model's rec."""
    from datetime import date as _date
    d = date or _date.today().isoformat()
    service.log_farmer_judgment(model_minutes, farmer_minutes, reason,
                                moisture_pct, d)
    return JSONResponse({"ok": True,
                         "message": "Recorded the farmer's choice alongside "
                                    "the model's recommendation."})


@app.post("/api/recommend")
async def api_recommend(moisture_pct: float = Form(...),
                        hour: int = Form(None),
                        planting_date: str = Form("2026-06-08")):
    """Fallback manual path (card not available). Normal flow uses /api/upload."""
    dat = _days_after(planting_date)
    if hour is None:
        hour = service.next_decision_window()
    res = service.recommend(moisture_pct=moisture_pct, hour=hour,
                            days_after_transplant=dat)
    return JSONResponse(res)


@app.get("/api/status")
def api_status():
    return JSONResponse(read_status())


@app.post("/api/upload_model")
async def api_upload_model(file: UploadFile = File(...),
                           x_admin_key: str = Header(...)):
    """
    Researcher uploads a newly trained iql_final.pt from their laptop.
    Protected by a secret key set in the ADMIN_KEY environment variable.
    """
    expected = os.environ.get("ADMIN_KEY", "")
    if not expected or not secrets.compare_digest(x_admin_key, expected):
        raise HTTPException(403, "Invalid admin key.")
    if not file.filename.endswith(".pt"):
        raise HTTPException(400, "Please upload a .pt checkpoint file.")
    import shutil
    dest = MODELS / "live.pt"
    tmp  = MODELS / "incoming.pt"
    with open(tmp, "wb") as f:
        shutil.copyfileobj(file.file, f)
    # Validate: try loading as a state dict
    try:
        import torch
        torch.load(tmp, map_location="cpu", weights_only=False)
    except Exception as e:
        tmp.unlink(missing_ok=True)
        raise HTTPException(400, f"File does not look like a valid checkpoint: {e}")
    tmp.replace(dest)
    # Reload the live model
    service._load_live_or_bootstrap()
    from model_service import audit
    audit("model_uploaded", filename=file.filename)
    return JSONResponse({"ok": True,
                         "message": f"Model updated from {file.filename}. "
                                    f"New version: {read_status().get('model_version',0)}"})


@app.get("/api/audit")
def api_audit(limit: int = 50):
    if not AUDIT_LOG.exists():
        return JSONResponse({"entries": []})
    lines = AUDIT_LOG.read_text().strip().splitlines()[-limit:]
    import json
    return JSONResponse({"entries": [json.loads(x) for x in lines][::-1]})


# --------------------------------------------------------------------------
# T2 decision log
# --------------------------------------------------------------------------
# Field protocol: at each decision window the worker asks the model for a
# recommendation, then records whether it was actually executed (and if not,
# what was done instead and why). Append-only JSONL beside the audit log;
# CSV export for analysis. No admin key — the worker uses this daily.

DECISIONS_LOG = AUDIT_LOG.parent / "decisions.jsonl"


@app.post("/api/decision")
async def api_decision(executed: str = Form(...),
                       recommended_minutes: int = Form(...),
                       actual_minutes: int = Form(None),
                       moisture_pct: float = Form(None),
                       reason: str = Form("")):
    """Log whether the model's recommendation was executed at this window."""
    import json
    from zoneinfo import ZoneInfo
    executed = executed.strip().lower()
    if executed not in ("yes", "no"):
        raise HTTPException(400, "executed must be 'yes' or 'no'")
    if actual_minutes is None:
        actual_minutes = recommended_minutes if executed == "yes" else 0
    now_utc = datetime.now(timezone.utc)
    entry = {
        "ts": now_utc.strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z",
        "date_ist": str(now_utc.astimezone(ZoneInfo("Asia/Kolkata")).date()),
        "executed": executed == "yes",
        "recommended_minutes": recommended_minutes,
        "actual_minutes": actual_minutes,
        "moisture_pct": moisture_pct,
        "reason": reason.strip(),
        "model_version": read_status().get("model_version", 0),
    }
    with open(DECISIONS_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return JSONResponse({"ok": True, "entry": entry})


def _read_decisions(limit: int = None):
    import json
    if not DECISIONS_LOG.exists():
        return []
    lines = DECISIONS_LOG.read_text().strip().splitlines()
    if limit:
        lines = lines[-limit:]
    return [json.loads(x) for x in lines]


@app.get("/api/decision_log")
def api_decision_log(limit: int = 200):
    return JSONResponse({"entries": _read_decisions(limit)[::-1]})


@app.get("/api/decision_log.csv")
def api_decision_log_csv():
    import csv, io
    from fastapi.responses import PlainTextResponse
    cols = ["ts", "date_ist", "executed", "recommended_minutes",
            "actual_minutes", "moisture_pct", "reason", "model_version"]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    for e in _read_decisions():
        w.writerow(e)
    return PlainTextResponse(
        buf.getvalue(), media_type="text/csv",
        headers={"Content-Disposition":
                 "attachment; filename=decision_log.csv"})


_DECIDE_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>T2 decision log</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 480px;
         margin: 1.5rem auto; padding: 0 1rem; color: #143; }
  h1 { font-size: 1.25rem; }
  label { display: block; margin: .8rem 0 .25rem; font-weight: 600; }
  input, textarea { width: 100%; padding: .55rem; font-size: 1rem;
         border: 1px solid #9c9; border-radius: 8px; box-sizing: border-box; }
  button { margin-top: 1rem; padding: .7rem 1.2rem; font-size: 1.05rem;
         border: 0; border-radius: 10px; cursor: pointer; }
  #get  { background: #2a7; color: #fff; }
  #yes  { background: #2a7; color: #fff; }
  #no   { background: #c53; color: #fff; margin-left: .75rem; }
  .card { background: #f2faf4; border: 1px solid #bde3c8; border-radius: 12px;
         padding: 1rem; margin-top: 1.25rem; }
  .big  { font-size: 1.35rem; font-weight: 700; }
  .muted{ color: #567; font-size: .9rem; }
  #followup, #done { display: none; }
</style>
</head>
<body>
<h1>T2 plot — irrigation decision</h1>

<label for="moist">Soil moisture (%)</label>
<input id="moist" type="number" step="0.1" inputmode="decimal">

<button id="get">Get recommendation</button>

<div id="rec" class="card" style="display:none">
  <div class="big" id="recText"></div>
  <div class="muted" id="recMeta"></div>
  <p class="big">Will you execute this?</p>
  <button id="yes">Yes — as recommended</button>
  <button id="no">No / different</button>
  <div id="followup">
    <label for="actual">Minutes actually run (0 if none)</label>
    <input id="actual" type="number" inputmode="numeric">
    <label for="reason">Reason (optional)</label>
    <textarea id="reason" rows="2"></textarea>
    <button id="save" style="background:#2a7;color:#fff">Save decision</button>
  </div>
</div>

<div id="done" class="card">
  <div class="big">Recorded ✔</div>
  <div class="muted">Logged to the decision log.
    <a href="/api/decision_log.csv">Download CSV</a></div>
</div>

<script>
let rec = null;
const el = id => document.getElementById(id);

el('get').onclick = async () => {
  const m = parseFloat(el('moist').value);
  if (isNaN(m)) { alert('Enter the soil moisture reading first.'); return; }
  const fd = new FormData();
  fd.append('moisture_pct', m);
  const r = await fetch('/api/recommend', {method: 'POST', body: fd});
  rec = await r.json();
  if (!rec.ok) { alert(rec.detail || 'Recommendation failed'); return; }
  el('recText').textContent = rec.drip_minutes > 0
      ? `Run drip ${rec.drip_minutes} min (${rec.litres_per_plant} L/plant)`
      : 'No water needed now';
  el('recMeta').textContent =
      `stage: ${rec.stage} | day ${rec.days_after_transplant}` +
      ` | window ${rec.time} | model v${rec.model_version}`;
  el('rec').style.display = 'block';
  el('done').style.display = 'none';
  el('followup').style.display = 'none';
};

async function save(executed, actual, reason) {
  const fd = new FormData();
  fd.append('executed', executed);
  fd.append('recommended_minutes', rec.drip_minutes);
  if (actual !== null) fd.append('actual_minutes', actual);
  fd.append('moisture_pct', rec.moisture_pct);
  fd.append('reason', reason || '');
  const r = await fetch('/api/decision', {method: 'POST', body: fd});
  const res = await r.json();
  if (!res.ok) { alert(res.detail || 'Save failed'); return; }
  el('rec').style.display = 'none';
  el('done').style.display = 'block';
}

el('yes').onclick = () => save('yes', null, '');
el('no').onclick = () => { el('followup').style.display = 'block'; };
el('save').onclick = () => {
  const a = parseInt(el('actual').value);
  if (isNaN(a)) { alert('Enter the minutes actually run (0 if none).'); return; }
  save('no', a, el('reason').value);
};
</script>
</body>
</html>"""


@app.get("/decide", response_class=HTMLResponse)
def decide():
    return _DECIDE_HTML
