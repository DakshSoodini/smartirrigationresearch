"""
plot_season.py — one simulated season under each policy (T0 / T1 / T2).

Uses the same simulator and policies as benchmark.py, with the deployed
checkpoint and its paired simulator parameters. Writes a light and a dark PNG
to docs/images/ for the README.

    python docs/plot_season.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from benchmark import build
from src.env import IrrigationEnv
from src.iql import IQLAgent, IQLConfig

SEED = 7
POLICIES = [("T0", "Fixed schedule"), ("T1", "FAO-56 rule"), ("T2", "IQL agent")]

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", grid="#e4e3df",
                  series=["#2a78d6", "#eb6834", "#1baf7a"]),
    "dark": dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", grid="#383835",
                 series=["#3987e5", "#d95926", "#199e70"]),
}


def run_season(env, policy):
    obs = env.reset()
    days, moisture, litres = [], [], []
    done = False
    while not done:
        obs, _, done, info = env.step(policy(obs, env))
        days.append(env.sim.state.day + env.sim.state.hour / 24.0)
        moisture.append(info["M_root"])
        litres.append(info["litres_per_plant"])
    return np.array(days), np.array(moisture), np.cumsum(litres)


def main():
    cfg = yaml.safe_load(open(ROOT / "configs" / "config.yaml"))
    sim_cfg, env_cfg, ip = build(cfg, str(ROOT / "artifacts" / "ckpts" / "sim_cfg.yaml"))
    action_litres = cfg["actuator"]["action_litres_per_plant"]
    snap = lambda v: min(range(len(action_litres)), key=lambda i: abs(action_litres[i] - v))

    probe = IrrigationEnv(env_cfg, sim_cfg, rng=np.random.default_rng(0))
    agent = IQLAgent(IQLConfig(obs_dim=probe.OBS_DIM, n_actions=probe.n_actions,
                               hidden_dim=cfg["iql"]["hidden_dim"],
                               n_hidden=cfg["iql"]["hidden_layers"]))
    agent.load_state_dict(torch.load(ROOT / "artifacts" / "ckpts" / "iql_final.pt",
                                     map_location="cpu", weights_only=False))

    policies = {
        "T0": lambda obs, env: snap(0.5),
        "T1": lambda obs, env: snap(ip.recommend_litres(
            M_surf=env.sim.state.M_surf, M_deep=env.sim.state.M_deep, day=env.sim.state.day)),
        "T2": lambda obs, env: agent.act(obs, deterministic=True),
    }
    # Same seed for every policy, so all three face identical weather noise
    runs = {k: run_season(IrrigationEnv(env_cfg, sim_cfg, rng=np.random.default_rng(SEED)), p)
            for k, p in policies.items()}

    # Optimal moisture band per growth stage, as a step function over days
    bands, edges, day0 = cfg["crop"]["optimal_moisture_band"], [], 0
    for stage in ["initial", "development", "mid", "late"]:
        n = cfg["crop"]["stage_days"][stage]
        edges.append((day0, day0 + n, *bands[stage]))
        day0 += n

    out = ROOT / "docs" / "images"
    out.mkdir(parents=True, exist_ok=True)
    for name, t in THEMES.items():
        plt.rcParams.update({"font.size": 11, "axes.edgecolor": t["grid"],
                             "axes.labelcolor": t["ink2"], "xtick.color": t["ink2"],
                             "ytick.color": t["ink2"], "text.color": t["ink"]})
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6.6), sharex=True,
                                       gridspec_kw={"height_ratios": [3, 2], "hspace": 0.32})
        fig.patch.set_facecolor(t["surface"])
        for ax in (ax1, ax2):
            ax.set_facecolor(t["surface"])
            ax.grid(axis="y", color=t["grid"], linewidth=0.8)
            ax.set_axisbelow(True)
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            ax.tick_params(length=0)

        # Bottom of the target band for each growth stage: below it, the
        # reward counts water stress
        xs = [e[0] for e in edges] + [edges[-1][1]]
        ys = [e[2] for e in edges] + [edges[-1][2]]
        ax1.step(xs, ys, where="post", color=t["ink2"], linewidth=1, linestyle=(0, (4, 3)))
        ax1.text(31, 57.75, "Stress threshold", color=t["ink2"], fontsize=10, va="top")

        for (key, label), color in zip(POLICIES, t["series"]):
            d, m, c = runs[key]
            ax1.plot(d, m, color=color, linewidth=1.6, label=f"{key} · {label}")
            ax2.plot(d, c, color=color, linewidth=2)
            ax2.annotate(f"{key} · {c[-1]:.0f} L", (d[-1], c[-1]), xytext=(6, 0),
                         textcoords="offset points", va="center", color=t["ink"], fontsize=10)

        ax1.set_ylabel("Root-zone moisture (sensor %)")
        ax1.set_title("One simulated season under each policy", loc="left",
                      color=t["ink"], fontsize=13, pad=26)
        ax1.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, frameon=False,
                   fontsize=10, labelcolor=t["ink"], borderaxespad=0.2, handlelength=1.6)
        ax2.set_ylabel("Water applied\n(L per plant, cumulative)")
        ax2.set_xlabel("Days after transplant")
        ax2.set_xlim(0, max(r[0][-1] for r in runs.values()) * 1.1)
        ax2.set_ylim(bottom=0)
        fig.savefig(out / f"simulated-season-{name}.png", dpi=150, bbox_inches="tight",
                    facecolor=t["surface"])
        plt.close(fig)

    for key, _ in POLICIES:
        d, m, c = runs[key]
        print(f"{key}: water {c[-1]:.1f} L/plant, moisture {m.min():.1f}-{m.max():.1f}")


if __name__ == "__main__":
    main()
