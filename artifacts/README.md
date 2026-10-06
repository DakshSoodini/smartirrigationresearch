# Artifacts

`ckpts/` holds the model the web app serves:

- **`ckpts/iql_final.pt`**: the deployed IQL agent's weights. They are trained for the two-window, litres-per-plant action space described in the main README.
- **`ckpts/sim_cfg.yaml`**: the calibrated simulator parameters the checkpoint was trained against.

The two files are a pair. The web app loads them together on start-up (`webapp/model_service.py`), and a checkpoint should never be shipped without its matching `sim_cfg.yaml`.

Intermediate checkpoints, cached buffers and training logs are written here too, but are git-ignored.

### Reproduce the checkpoint

From the repository root:

```bash
./run_training.sh                 # 200k steps in resumable 25k-step chunks
python benchmark.py --episodes 30 # compare T0 / T1 / T2 in the simulator
```
