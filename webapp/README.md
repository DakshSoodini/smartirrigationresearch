# PANI Web App: Irrigation Advisor

A FastAPI app that serves the IQL irrigation model to the farm and refines it safely as new sensor data arrives. It is built for a field worker on a phone: one screen and large text.

## Pages

- **`/`**: farm page. The worker uploads the sensor node's SD-card CSV. The app reads plot P3's latest moisture and recommends a dose for the next 09:00 or 14:00 window, in litres per plant and drip minutes. The worker can record what they would have done instead. The irrigation history loads from the farm's field-log Google Sheet.
- **`/decide`**: decision log. At each window the worker records whether the recommendation was carried out, and if not, what was done and why. It can be exported as CSV from `/api/decision_log.csv`.
- **`/research`**: researcher view. It shows the model version, calibration state and the full audit log. It also has an admin-key-protected form for uploading a newly trained checkpoint.

The full endpoint list is in the docstring at the top of `app.py`.

## Safety model

Uploading data **cannot**, on its own, change what the farm is told. Each upload goes through five steps:

1. It is validated (moisture and temperature in range, enough rows). Bad data is **quarantined**.
2. It recalibrates the simulator from the new field data.
3. A **candidate** model is fine-tuned.
4. The candidate is **shadow-tested** against the live model on simulated seasons.
5. The candidate is **promoted only if it beats the live model**. Otherwise the live model keeps serving.

Separately, every recommendation passes through an **FAO‑56 floor**. If the model says "no water" while the soil is past the FAO‑56 depletion limit, the app serves the FAO‑56 dose instead and tags the response `source: "fao56_floor"`.

Every step is written to `webapp/storage/logs/audit.jsonl`.

## Run locally

```bash
# from the repository root
pip install -r requirements.txt
cd webapp
uvicorn app:app --reload --port 8000
```

Open <http://localhost:8000> for the farm page and <http://localhost:8000/research> for the researcher view. The app bootstraps from the packaged checkpoint in `artifacts/ckpts/`.

## Configuration

Secrets and links to private resources are read from environment variables and never committed:

| Variable | Purpose |
|---|---|
| `ADMIN_KEY` | Required to upload a new checkpoint from `/research`. Unset means uploads are refused. |
| `SHEET_CSV_URL` | CSV export link of the farm's field-log Google Sheet, used by "Refresh from sheet". Unset means sheet sync is disabled. |
| `JUDGMENT_FORM_URL`, `JUDGMENT_FORM_FIELDS` | Optional. Relays each farmer judgment to a Google Form so it survives restarts on hosts with ephemeral disks. |

Field settings to confirm before use:
- `PLANTING` (the transplant date) in `static/index.html` and `static/events.html`.
- `emitter_rate_L_per_h` and `t2_plot_id` in `configs/config.yaml`.

## Deploying

The included `Procfile` works on PaaS hosts such as Render, Railway or Fly.io:

```
web: cd webapp && uvicorn app:app --host 0.0.0.0 --port $PORT
```

Use an instance with at least about 1 GB of RAM, since PyTorch and fine-tuning run in-process. `webapp/storage/` holds the live model, uploads and logs. On hosts whose disk is wiped on restart, mount it as a persistent volume or configure the Google Form relay above.

Refinement runs on a background thread. That is enough for occasional uploads. Heavier use would move it to a separate worker process.
