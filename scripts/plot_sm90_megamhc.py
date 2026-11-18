"""Render latency and sampled-warp timelines from benchmark JSON evidence.

python scripts/plot_sm90_megamhc.py artifacts/sm90-mega-mhc/benchmark-final.json \
  --profiles artifacts/sm90-mega-mhc/sm90-megamhc-4096-cta0.json \
             artifacts/sm90-mega-mhc/sm90-megamhc-32-cta0.json
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


def save(figure, output, name):
    for extension in ("svg", "png"):
        figure.savefig(output / f"{name}.{extension}", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def latency(result, output):
    cases = result["cases"]
    fig, axes = plt.subplots(1, len(cases), figsize=(11, 4.2), squeeze=False)
    for ax, case in zip(axes[0], cases):
        labels = ["SGLang baseline", "SM90 Mega mHC"]
        values = [case["timing"][k]["median_us"] for k in ("baseline", "mega")]
        ax.bar(labels, values, color=["#a4b2c5", "#416cc2"], width=.62)
        for i, (key, value) in enumerate(zip(("baseline", "mega"), values)):
            samples = case["timing"][key]["samples_us"]
            ax.scatter(np.full(len(samples), i), samples, s=10, color="#26354b", zorder=3, alpha=.5)
            ax.text(i, value * 1.05, f"{value:.2f} us", ha="center", fontsize=12)
        ax.set_ylim(0, max(values) * 1.28)
        ax.set_ylabel("Boundary latency (us), lower is better")
        ax.set_title(f"{case['tokens']} tokens / rank  |  {case['speedup']:.3f}x")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
    fig.suptitle("H200: shifted mHC + next stats + RMSNorm, H=5120, hc=4", fontsize=14)
    fig.text(.5, -.01, "Real checkpoint weights; synthetic activations. CUDA Graph medians; dots are independent replay samples.", ha="center", fontsize=9)
    fig.tight_layout()
    save(fig, output, "sm90-mega-mhc-latency")


def timeline(paths, output):
    colors = {"role": "#edf0f5", "Post": "#4b9d82", "wait": "#e3b25b",
              "WGMMA completion gate": "#9771c4", "WGMMA drain": "#9771c4"}
    tracks = [(0, "IO TMA"), (1, "Fn TMA"), (3, "Store/coeff"), (4, "Post WG1"),
              (8, "Post WG2"), (12, "WGMMA WG3"), (16, "Mix WG4"), (20, "Norm WG5")]
    fig, axes = plt.subplots(len(paths), 1, figsize=(13, 4.6 * len(paths)), squeeze=False)
    for ax, path in zip(axes[:, 0], paths):
        trace = json.loads(path.read_text())
        events = trace["traceEvents"]
        for y, (warp, _name) in enumerate(tracks):
            selected = [e for e in events if e["pid"] == 0 and e["tid"] == warp]
            for label in ("role", "wait", "Post", "WGMMA drain", "WGMMA completion gate"):
                intervals = [(e["ts"], e["dur"]) for e in selected if e["name"] == label]
                ax.broken_barh(intervals, (y - .3, .6), facecolors=colors[label], linewidth=0)
        ax.set_yticks(range(len(tracks)), [name for _warp, name in tracks])
        ax.invert_yaxis()
        ax.set_xlim(left=0)
        ax.set_xlabel("Globaltimer time from first sampled warp (us)")
        ax.set_title(f"{trace['metadata']['tokens']} tokens: CTA 0, one sampled warp per role (instrumented)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", alpha=.18)
    fig.legend(handles=[Patch(facecolor=colors[k], label=label) for k, label in (
        ("role", "Role span, includes uninstrumented work / waits"), ("wait", "A / IO / Fn wait"),
        ("Post", "Post compute + publish"), ("WGMMA completion gate", "WGMMA completion gate"))],
        loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.suptitle("SM90 Mega mHC pipeline: sampled %globaltimer intervals", fontsize=14)
    fig.tight_layout(rect=(0, .075, 1, .97))
    save(fig, output, "sm90-mega-mhc-timeline")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--profiles", type=Path, nargs="*")
    parser.add_argument("--output", type=Path, default=Path("docs/figures"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"svg.fonttype": "none", "font.size": 10})
    latency(json.loads(args.results.read_text()), args.output)
    if args.profiles:
        timeline(args.profiles, args.output)


if __name__ == "__main__":
    main()
