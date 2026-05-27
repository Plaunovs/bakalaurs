"""
analyze_results.py — Analizē visas sistēmas no results/raw/ JSON failiem.

Automātiski iekļauj jebkuru sistēmu (arī nākamo hibrīdu) bez koda izmaiņām —
cilpo pa VISIEM *.json failiem mapē results/raw/.

Palaišana:
    conda activate env_analysis
    python analysis/analyze_results.py
"""

import sys
import os
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import roc_curve, auc

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from shared.shared_utils import (
    RESULTS_RAW_DIR, RESULTS_FIGURES_DIR, RESULTS_TABLES_DIR,
    save_csv,
)

# ---------------------------------------------------------------------------
# Matplotlib stils
# ---------------------------------------------------------------------------

matplotlib.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "legend.fontsize":   10,
    "figure.dpi":        100,   # ekrānam; saglabāšanai izmantojam 300
    "savefig.dpi":       300,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
})

# Konsekventas krāsas — paplašināmas automātiski jaunām sistēmām
_FIXED_COLORS = {
    "face_recognition": "#1565C0",   # zilā
    "deepface":         "#BF360C",   # oranžsarkanā
    "insightface":      "#2E7D32",   # zaļā
}
_COLOR_CYCLE = plt.cm.tab10.colors


def system_color(system: str, idx: int = 0) -> str:
    return _FIXED_COLORS.get(system, _COLOR_CYCLE[idx % len(_COLOR_CYCLE)])


# Iznākumu krāsas un latviskās etiķetes
OUTCOME_COLORS = {
    "correct":  "#43A047",
    "wrong":    "#E53935",
    "rejected": "#FB8C00",
    "no_face":  "#757575",
}
OUTCOME_LABELS = {
    "correct":  "Pareiza ID",
    "wrong":    "Kļūdaina ID",
    "rejected": "Noraidīta",
    "no_face":  "Seja nav atrasta",
}

METRIC_LABELS = {
    "execution_time_ms_per_image": "Izpildes laiks (ms/attēls)",
    "throughput_fps":              "Caurlaidspēja (FPS)",
    "correct_pct":                 "Pareiza identifikācija (%)",
    "wrong_pct":                   "Kļūdaina identifikācija (%)",
    "rejected_pct":                "Noraidīta identifikācija (%)",
    "no_face_pct":                 "Seja nav atrasta (%)",
    "peak_mb":                     "Atmiņas patēriņš — peak (MB)",
    "delta_mb":                    "Atmiņas delta (MB)",
}


# ---------------------------------------------------------------------------
# Datu ielāde
# ---------------------------------------------------------------------------

def load_all_results() -> dict[str, list[dict]]:
    """
    Ielādē visus *.json failus no results/raw/.
    Atgriež {sistēma: [run1_dict, run2_dict, ...]}.
    Darbojas ar jebkuru sistēmu skaitu — nav hardkodētu nosaukumu.
    """
    results: dict[str, list[dict]] = defaultdict(list)
    json_files = sorted(RESULTS_RAW_DIR.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(
            f"Nav JSON failu mapē {RESULTS_RAW_DIR}. "
            "Palaiž vispirms run_*.py skriptus."
        )
    for jf in json_files:
        with open(jf, encoding="utf-8") as f:
            data = json.load(f)
        system = data.get("system", jf.stem)
        results[system].append(data)
    print(f"Ielādētas sistēmas: {sorted(results)}")
    return dict(results)


# ---------------------------------------------------------------------------
# Statistika
# ---------------------------------------------------------------------------

SUMMARY_KEYS = [
    "execution_time_ms_per_image", "throughput_fps",
    "correct_pct", "wrong_pct", "rejected_pct", "no_face_pct",
    "peak_mb", "baseline_mb", "delta_mb",
]


def compute_stats(runs: list[dict]) -> dict[str, dict]:
    """Aprēķina mean + std katram rādītājam pār palaišanām."""
    stats: dict[str, dict] = {}
    for key in SUMMARY_KEYS:
        vals = [r["summary"].get(key, 0.0) for r in runs]
        stats[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=0))}
    thresholds = [r.get("threshold", 0.0) for r in runs]
    stats["threshold"] = {"mean": float(np.mean(thresholds)),
                          "std":  float(np.std(thresholds, ddof=0))}
    return stats


# ---------------------------------------------------------------------------
# CSV tabulas
# ---------------------------------------------------------------------------

def save_per_system_csv(system: str, stats: dict, runs: list[dict]):
    """Saglabā results/tables/{system}.csv ar Kritērijs/Vērtība formātu."""
    rows = [
        ("Sistēma",                         system),
        ("Palaišanas skaits",               str(len(runs))),
        ("Izpildes laiks (ms)",             f"{stats['execution_time_ms_per_image']['mean']:.4f} ± {stats['execution_time_ms_per_image']['std']:.4f}"),
        ("Caurlaidspēja (fps)",             f"{stats['throughput_fps']['mean']:.4f} ± {stats['throughput_fps']['std']:.4f}"),
        ("Pareiza identifikācija (%)",      f"{stats['correct_pct']['mean']:.4f} ± {stats['correct_pct']['std']:.4f}"),
        ("Kļūdaina identifikācija (%)",     f"{stats['wrong_pct']['mean']:.4f} ± {stats['wrong_pct']['std']:.4f}"),
        ("Noraidīta identifikācija (%)",    f"{stats['rejected_pct']['mean']:.4f} ± {stats['rejected_pct']['std']:.4f}"),
        ("Seja nav atrasta (%)",            f"{stats['no_face_pct']['mean']:.4f} ± {stats['no_face_pct']['std']:.4f}"),
        ("Atmiņas patēriņš — peak (MB)",   f"{stats['peak_mb']['mean']:.2f} ± {stats['peak_mb']['std']:.2f}"),
        ("Atmiņas delta (MB)",              f"{stats['delta_mb']['mean']:.2f} ± {stats['delta_mb']['std']:.2f}"),
        ("Slieksnis T",                     f"{stats['threshold']['mean']:.5f} ± {stats['threshold']['std']:.5f}"),
    ]
    path = RESULTS_TABLES_DIR / f"{system}.csv"
    save_csv(rows, path)


def save_summary_csv(all_systems: dict[str, list[dict]],
                     all_stats: dict[str, dict]):
    """Saglabā results/tables/summary.csv ar visām sistēmām blakus."""
    systems = sorted(all_systems.keys())
    header_row = ["Rādītājs"] + systems
    metric_rows = []

    for key, label in {
        "execution_time_ms_per_image": "Izpildes laiks (ms)",
        "throughput_fps":              "Caurlaidspēja (fps)",
        "correct_pct":                 "Pareiza identifikācija (%)",
        "wrong_pct":                   "Kļūdaina identifikācija (%)",
        "rejected_pct":                "Noraidīta identifikācija (%)",
        "no_face_pct":                 "Seja nav atrasta (%)",
        "peak_mb":                     "Atmiņas patēriņš (MB)",
        "threshold":                   "Slieksnis T",
    }.items():
        row = [label]
        for sys in systems:
            s = all_stats[sys].get(key, {"mean": 0, "std": 0})
            row.append(f"{s['mean']:.4f} ± {s['std']:.4f}")
        metric_rows.append(row)

    path = RESULTS_TABLES_DIR / "summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        f.write(",".join(header_row) + "\n")
        for row in metric_rows:
            f.write(",".join(row) + "\n")
    print(f"  Kopsavilkuma CSV saglabāts: {path}")


# ---------------------------------------------------------------------------
# Palīgfunkcijas grafiku saglabāšanai
# ---------------------------------------------------------------------------

def savefig(fig: plt.Figure, name: str):
    """Saglabā figūru PNG un PDF formātos mapē results/figures/."""
    RESULTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        p = RESULTS_FIGURES_DIR / f"{name}.{ext}"
        fig.savefig(p, bbox_inches="tight")
    print(f"  Grafiks saglabāts: {RESULTS_FIGURES_DIR / name}.{{png,pdf}}")


# ---------------------------------------------------------------------------
# Grafiks 1: Iznākumu sadalījums (katrai sistēmai)
# ---------------------------------------------------------------------------

def plot_outcome_distribution(system: str, runs: list[dict], color: str):
    """Stabiņu grafiks: iznākumu % katrai palaišanai + vidējais."""
    outcomes = ["correct", "wrong", "rejected", "no_face"]
    n_runs = len(runs)
    x = np.arange(n_runs + 1)  # +1 priekš vidējā
    width = 0.18

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, outcome in enumerate(outcomes):
        vals = [r["summary"].get(f"{outcome}_pct", 0.0) for r in runs]
        mean_val = float(np.mean(vals))
        all_vals = vals + [mean_val]
        bars = ax.bar(x + i * width, all_vals, width,
                      label=OUTCOME_LABELS[outcome],
                      color=OUTCOME_COLORS[outcome],
                      alpha=0.85)

    x_labels = [f"Palaišana {r['run']}" for r in runs] + ["Vidējais"]
    ax.set_xticks(x + width * (len(outcomes) - 1) / 2)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("Proporcija (%)")
    ax.set_title(f"{system} — Iznākumu sadalījums")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 110)
    fig.tight_layout()
    savefig(fig, f"{system}_outcome_distribution")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Grafiks 2: Sliekšņa kalibrācijas līkne (katrai sistēmai)
# ---------------------------------------------------------------------------

def plot_calibration_curve(system: str, runs: list[dict], color: str):
    """Rāda iznākumu likmes kā funkciju no T (no 1. palaišanas sweep)."""
    sweep_run = next((r for r in runs if r.get("threshold_sweep")), None)
    if sweep_run is None or not sweep_run["threshold_sweep"]:
        print(f"  BRIDINAJUMS: Nav threshold_sweep datiem {system}.")
        return

    sweep = sweep_run["threshold_sweep"]
    chosen_T = sweep_run["threshold"]
    n_total_val = sum(
        sweep[0].get(k, 0) for k in ("correct", "wrong", "rejected", "no_face")
    )
    if n_total_val == 0:
        return

    T_vals    = np.array([s["T"]        for s in sweep])
    correct_p = np.array([s["correct"]  for s in sweep]) / n_total_val * 100
    wrong_p   = np.array([s["wrong"]    for s in sweep]) / n_total_val * 100
    rejected_p= np.array([s["rejected"] for s in sweep]) / n_total_val * 100
    no_face_p = np.array([s["no_face"]  for s in sweep]) / n_total_val * 100

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(T_vals, correct_p,  color=OUTCOME_COLORS["correct"],
            label=OUTCOME_LABELS["correct"],  lw=2)
    ax.plot(T_vals, wrong_p,    color=OUTCOME_COLORS["wrong"],
            label=OUTCOME_LABELS["wrong"],    lw=2)
    ax.plot(T_vals, rejected_p, color=OUTCOME_COLORS["rejected"],
            label=OUTCOME_LABELS["rejected"], lw=2, ls="--")
    if no_face_p.max() > 0:
        ax.plot(T_vals, no_face_p, color=OUTCOME_COLORS["no_face"],
                label=OUTCOME_LABELS["no_face"], lw=1.5, ls=":")
    ax.axvline(chosen_T, color="black", lw=1.5, ls="-.",
               label=f"Izvēlētais T = {chosen_T:.4f}")
    ax.set_xlabel("Slieksnis T")
    ax.set_ylabel("Proporcija (%)")
    ax.set_title(f"{system} — Sliekšņa kalibrācijas līkne (validācijas kopa)")
    ax.legend(loc="center right")
    fig.tight_layout()
    savefig(fig, f"{system}_calibration_curve")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Grafiks 3: ROC līkne (katrai sistēmai)
# ---------------------------------------------------------------------------

def _compute_roc_from_run(run: dict) -> tuple | None:
    """
    Aprēķina ROC no per_image datiem.
    Pozitīvs = best_gallery_match == true_label (pareizs top-1).
    Rezultāts: (fpr, tpr, roc_auc) vai None, ja nav datu.
    """
    metric_type = run.get("metric_type", "cosine_similarity")
    y_true, y_score = [], []

    for img in run.get("per_image", []):
        bgm = img.get("best_gallery_match")
        score = img.get("score")
        if bgm is None or score is None:
            continue
        y_true.append(1 if bgm == img["true_label"] else 0)
        # Kosinusam: lielāks = labāks; attālumam: negatīvs, lai arī lielāks = labāks
        y_score.append(score if metric_type == "cosine_similarity" else -score)

    if len(y_true) < 2 or len(set(y_true)) < 2:
        return None

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc


def plot_roc_curve(system: str, runs: list[dict], color: str):
    """ROC līkne visām 3 palaišanām vienā grafikā."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Nejaušs klasifikators")

    aucs = []
    for run in runs:
        res = _compute_roc_from_run(run)
        if res is None:
            continue
        fpr, tpr, roc_auc = res
        aucs.append(roc_auc)
        ax.plot(fpr, tpr, color=color, lw=1.5, alpha=0.6,
                label=f"Palaišana {run['run']} (AUC = {roc_auc:.3f})")

    if not aucs:
        plt.close(fig)
        print(f"  BRIDINAJUMS: Nav ROC datu {system}.")
        return

    ax.set_xlabel("Kļūdainas pieņemšanas likme (FPR)")
    ax.set_ylabel("Pareizas pieņemšanas likme (TPR)")
    ax.set_title(f"{system} — ROC līkne\n"
                 f"Vidējais AUC = {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
    ax.legend(loc="lower right")
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    fig.tight_layout()
    savefig(fig, f"{system}_roc_curve")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Grafiks 4: Kopsavilkuma grupētie stabiņi
# ---------------------------------------------------------------------------

def plot_summary_grouped_bars(all_systems: dict[str, list[dict]],
                               all_stats: dict[str, dict]):
    """Četri grupēti stabiņu grafiki: laiks, FPS, pareiza ID %, atmiņa."""
    metrics_to_plot = [
        ("execution_time_ms_per_image", "Izpildes laiks (ms/attēls)"),
        ("throughput_fps",              "Caurlaidspēja (FPS)"),
        ("correct_pct",                 "Pareiza identifikācija (%)"),
        ("peak_mb",                     "Atmiņas patēriņš — peak (MB)"),
    ]

    systems = sorted(all_systems.keys())
    n_sys = len(systems)
    x = np.arange(len(metrics_to_plot))
    width = 0.8 / n_sys

    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(14, 5))
    if len(metrics_to_plot) == 1:
        axes = [axes]

    for ax_i, (ax, (key, label)) in enumerate(zip(axes, metrics_to_plot)):
        for j, sys in enumerate(systems):
            s = all_stats[sys].get(key, {"mean": 0, "std": 0})
            color = system_color(sys, j)
            ax.bar(j, s["mean"], width * n_sys * 0.8,
                   color=color, alpha=0.85,
                   yerr=s["std"], capsize=5,
                   label=sys if ax_i == 0 else "")
        ax.set_xticks(range(n_sys))
        ax.set_xticklabels(systems, rotation=15, ha="right")
        ax.set_title(label)
        ax.set_ylabel(label.split("(")[-1].rstrip(")") if "(" in label else "")

    # Leģenda tikai pirmajā panelī
    handles = [mpatches.Patch(color=system_color(s, i), label=s)
               for i, s in enumerate(systems)]
    fig.legend(handles=handles, loc="upper center", ncol=n_sys,
               bbox_to_anchor=(0.5, 1.02), title="Sistēma")
    fig.suptitle("Kopsavilkums — visas sistēmas (vidējais ± std, 3 palaišanas)",
                 y=1.05)
    fig.tight_layout()
    savefig(fig, "summary_grouped_bars")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Grafiks 5: Apvienotā ROC
# ---------------------------------------------------------------------------

def plot_combined_roc(all_systems: dict[str, list[dict]]):
    """Visas sistēmas vienā ROC grafikā (vidējā palaišana)."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)

    systems = sorted(all_systems.keys())
    for j, system in enumerate(systems):
        color = system_color(system, j)
        aucs, fprs, tprs = [], [], []
        for run in all_systems[system]:
            res = _compute_roc_from_run(run)
            if res is None:
                continue
            fpr, tpr, roc_auc = res
            aucs.append(roc_auc)
            fprs.append(fpr)
            tprs.append(tpr)

        if not aucs:
            continue

        # Interpolē uz kopīgu FPR bāzi un aprēķina vidējo TPR
        base_fpr = np.linspace(0, 1, 200)
        interp_tprs = [np.interp(base_fpr, f, t) for f, t in zip(fprs, tprs)]
        mean_tpr = np.mean(interp_tprs, axis=0)
        mean_auc = float(np.mean(aucs))
        std_auc  = float(np.std(aucs))

        ax.plot(base_fpr, mean_tpr, color=color, lw=2,
                label=f"{system} (AUC = {mean_auc:.3f} ± {std_auc:.3f})")
        if len(interp_tprs) > 1:
            std_tpr = np.std(interp_tprs, axis=0)
            ax.fill_between(base_fpr,
                            np.clip(mean_tpr - std_tpr, 0, 1),
                            np.clip(mean_tpr + std_tpr, 0, 1),
                            color=color, alpha=0.15)

    ax.set_xlabel("Kļūdainas pieņemšanas likme (FPR)")
    ax.set_ylabel("Pareizas pieņemšanas likme (TPR)")
    ax.set_title("Apvienotā ROC līkne — visas sistēmas")
    ax.legend(loc="lower right")
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    fig.tight_layout()
    savefig(fig, "combined_roc_curve")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Grafiks 6: Rezultātu sadalījums — horizontāli stabiņi, visas sistēmas
# ---------------------------------------------------------------------------

def plot_results_distribution_horizontal(all_systems: dict, all_stats: dict):
    """
    Horizontāls sadalīts stabiņu grafiks: viena josla uz sistēmu.
    X ass: 0–100 %. Krāsas: pareizi / kļūdaini / noraidīts / seja nav atrasta.
    """
    systems  = sorted(all_systems.keys())
    outcomes = ["correct", "wrong", "rejected", "no_face"]
    n = len(systems)

    fig, ax = plt.subplots(figsize=(11, max(3, n * 1.4 + 1.5)))

    lefts = np.zeros(n)
    for outcome in outcomes:
        vals = np.array([
            all_stats[s].get(f"{outcome}_pct", {"mean": 0})["mean"]
            for s in systems
        ])
        ax.barh(range(n), vals, left=lefts,
                color=OUTCOME_COLORS[outcome],
                label=OUTCOME_LABELS[outcome],
                edgecolor="white", linewidth=0.4)
        for i, (v, l) in enumerate(zip(vals, lefts)):
            if v >= 4:
                ax.text(l + v / 2, i, f"{v:.1f}%",
                        ha="center", va="center",
                        color="white", fontsize=9, fontweight="bold")
        lefts += vals

    ax.set_yticks(range(n))
    ax.set_yticklabels(systems, fontsize=11)
    ax.set_xlabel("Proporcija (%)")
    ax.set_xlim(0, 100)
    ax.set_title("Rezultātu sadalījums (%)", fontsize=13, pad=10)
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    savefig(fig, "results_distribution")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Grafiks 7: Izpildes laiks un caurlaidspēja — salīdzinājums
# ---------------------------------------------------------------------------

def plot_execution_time_comparison(all_systems: dict, all_stats: dict):
    """
    Divi paneļi: kreisais — izpildes laiks ms/attēls; labais — FPS.
    Kļūdas joslas = std no palaišanām.
    """
    systems = sorted(all_systems.keys())
    colors  = [system_color(s, i) for i, s in enumerate(systems)]
    x       = np.arange(len(systems))

    times    = [all_stats[s]["execution_time_ms_per_image"]["mean"] for s in systems]
    times_sd = [all_stats[s]["execution_time_ms_per_image"]["std"]  for s in systems]
    fps      = [all_stats[s]["throughput_fps"]["mean"] for s in systems]
    fps_sd   = [all_stats[s]["throughput_fps"]["std"]  for s in systems]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    bars1 = ax1.bar(x, times, yerr=times_sd, capsize=6,
                    color=colors, alpha=0.85, width=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(systems, rotation=15, ha="right")
    ax1.set_ylabel("Izpildes laiks (ms/attēls)")
    ax1.set_title("Izpildes laiks")
    for bar, v in zip(bars1, times):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + max(times) * 0.01,
                 f"{v:.1f}", ha="center", va="bottom", fontsize=9)

    bars2 = ax2.bar(x, fps, yerr=fps_sd, capsize=6,
                    color=colors, alpha=0.85, width=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(systems, rotation=15, ha="right")
    ax2.set_ylabel("Caurlaidspēja (attēli/s)")
    ax2.set_title("Caurlaidspēja (FPS)")
    for bar, v in zip(bars2, fps):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + max(fps) * 0.01,
                 f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Izpildes laika salīdzinājums", fontsize=13)
    fig.tight_layout()
    savefig(fig, "execution_time_comparison")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Grafiks 8: Iznākumi pa personām — viena sistēma
# ---------------------------------------------------------------------------

def _per_person_mean(runs: list[dict]) -> tuple[list[str], dict]:
    """
    Aprēķina vidējo iznākumu procentu pa personām pāri visām palaišanām.
    Atgriež (sorted_persons, {outcome: [pct, ...]}).
    """
    from collections import defaultdict

    persons_acc: dict[str, dict[str, list]] = defaultdict(
        lambda: {"correct": [], "wrong": [], "rejected": [], "no_face": []}
    )
    for run in runs:
        per_run: dict[str, dict] = defaultdict(
            lambda: {"correct": 0, "wrong": 0, "rejected": 0, "no_face": 0, "total": 0}
        )
        for img in run.get("per_image", []):
            p = img["true_label"]
            per_run[p][img["outcome"]] += 1
            per_run[p]["total"] += 1
        for person, counts in per_run.items():
            total = counts["total"] or 1
            for o in ("correct", "wrong", "rejected", "no_face"):
                persons_acc[person][o].append(counts[o] / total * 100)

    persons = sorted(persons_acc.keys(),
                     key=lambda x: int(x.replace("person", "")))
    data = {
        o: [float(np.mean(persons_acc[p][o])) for p in persons]
        for o in ("correct", "wrong", "rejected", "no_face")
    }
    return persons, data


def plot_per_person_outcomes(system: str, runs: list[dict], color: str):
    """Sadalīts stabiņu grafiks: iznākumi pa personām (vidējais pāri palaišanām)."""
    persons, data = _per_person_mean(runs)
    x = np.arange(len(persons))

    fig, ax = plt.subplots(figsize=(14, 5))
    bottoms = np.zeros(len(persons))
    for outcome in ("correct", "wrong", "rejected", "no_face"):
        vals = np.array(data[outcome])
        ax.bar(x, vals, bottom=bottoms,
               color=OUTCOME_COLORS[outcome],
               label=OUTCOME_LABELS[outcome],
               edgecolor="white", linewidth=0.3)
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels(persons, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Proporcija (%)")
    ax.set_ylim(0, 108)
    ax.set_title(f"{system} — Iznākumi pa personām (%)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    savefig(fig, f"{system}_per_person")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Tabula 2: Personas × iznākumi — CSV
# ---------------------------------------------------------------------------

def save_per_person_table(system: str, runs: list[dict]):
    """
    Saglabā CSV: persona, pareizi, kļūdaini, noraidīts, seja_nav_atrasta, kopā.
    Izmanto vidējos skaitus pāri palaišanām (noapaļo uz veselu skaitli).
    """
    from collections import defaultdict

    persons_acc: dict[str, dict[str, list]] = defaultdict(
        lambda: {"correct": [], "wrong": [], "rejected": [], "no_face": []}
    )
    for run in runs:
        per_run: dict[str, dict] = defaultdict(
            lambda: {"correct": 0, "wrong": 0, "rejected": 0, "no_face": 0}
        )
        for img in run.get("per_image", []):
            per_run[img["true_label"]][img["outcome"]] += 1
        for person, counts in per_run.items():
            for o in ("correct", "wrong", "rejected", "no_face"):
                persons_acc[person][o].append(counts[o])

    persons = sorted(persons_acc.keys(),
                     key=lambda x: int(x.replace("person", "")))

    path = RESULTS_TABLES_DIR / f"{system}_per_person.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        f.write("Persona,Pareizi,Kludaini,Noraidits,Seja nav atrasta,Kopā\n")
        for p in persons:
            c  = round(np.mean(persons_acc[p]["correct"]))
            w  = round(np.mean(persons_acc[p]["wrong"]))
            r  = round(np.mean(persons_acc[p]["rejected"]))
            nf = round(np.mean(persons_acc[p]["no_face"]))
            total = c + w + r + nf
            f.write(f"{p},{c},{w},{r},{nf},{total}\n")
    print(f"  Personu tabula saglabata: {path}")


# ---------------------------------------------------------------------------
# Galvenā funkcija
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", type=str, default=None,
                        help="Analizē tikai šo sistēmu (piem. face_recognition)")
    args = parser.parse_args()

    print("\n" + "=" * 55)
    print("  Sejas Atpazīšanas Sistēmu Analīze")
    print("=" * 55)

    all_systems = load_all_results()

    if args.system:
        if args.system not in all_systems:
            print(f"KĻŪDA: '{args.system}' nav atrasta. Pieejamas: {sorted(all_systems)}")
            return
        all_systems = {args.system: all_systems[args.system]}

    all_stats: dict[str, dict] = {}

    for j, (system, runs) in enumerate(sorted(all_systems.items())):
        color = system_color(system, j)
        print(f"\n--- {system} ({len(runs)} palaišanas) ---")

        stats = compute_stats(runs)
        all_stats[system] = stats

        # CSV
        save_per_system_csv(system, stats, runs)

        # Grafiki katrai sistēmai
        plot_outcome_distribution(system, runs, color)
        plot_calibration_curve(system, runs, color)
        plot_roc_curve(system, runs, color)
        plot_per_person_outcomes(system, runs, color)
        save_per_person_table(system, runs)

    # Kopsavilkuma faili
    print("\n--- Kopsavilkums ---")
    save_summary_csv(all_systems, all_stats)
    plot_summary_grouped_bars(all_systems, all_stats)
    plot_combined_roc(all_systems)
    plot_results_distribution_horizontal(all_systems, all_stats)
    plot_execution_time_comparison(all_systems, all_stats)

    print("\n" + "=" * 55)
    print(f"  Gatavs. Grafiki: {RESULTS_FIGURES_DIR}")
    print(f"  Tabulas:         {RESULTS_TABLES_DIR}")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
