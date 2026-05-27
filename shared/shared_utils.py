"""
shared_utils.py — Kopīgais modulis visām sejas atpazīšanas sistēmām.

Atkarības: TIKAI numpy, psutil un Python stdlib.
Importējams no jebkuras sistēmas vides (env_face_recognition, env_deepface,
env_insightface, env_analysis).
"""

import os
import json
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import psutil

# ---------------------------------------------------------------------------
# Ceļi
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_RAW_DIR     = _PROJECT_ROOT / "results" / "raw"
RESULTS_TABLES_DIR  = _PROJECT_ROOT / "results" / "tables"
RESULTS_FIGURES_DIR = _PROJECT_ROOT / "results" / "figures"
DEFAULT_DATASET     = _PROJECT_ROOT / "Dataset"

# ---------------------------------------------------------------------------
# Konstantes
# ---------------------------------------------------------------------------

SEED = 42
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# ---------------------------------------------------------------------------
# Atmiņas monitors
# ---------------------------------------------------------------------------

class MemoryMonitor:
    """
    Fona pavediens, kas mēra procesa RSS ik pa INTERVAL_S sekundēm.

    Lietošana:
        mon = MemoryMonitor()
        mon.start()           # piefiksē baseline un sāk mērīšanu
        ... (modeļa ielāde, galerijas veidošana, tests) ...
        mon.stop()            # apstādina un aprēķina peak
        r = mon.result()      # {'baseline_mb', 'peak_mb', 'delta_mb'}
    """

    INTERVAL_S = 0.1

    def __init__(self):
        self._process = psutil.Process(os.getpid())
        self._running = False
        self._thread = None
        self._samples: list[float] = []
        self.baseline_mb: float = 0.0
        self.peak_mb: float = 0.0
        self.delta_mb: float = 0.0

    def _rss_mb(self) -> float:
        return self._process.memory_info().rss / (1024 ** 2)

    def _monitor(self):
        while self._running:
            self._samples.append(self._rss_mb())
            time.sleep(self.INTERVAL_S)

    def start(self):
        self.baseline_mb = self._rss_mb()
        self._samples = []
        self._running = True
        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        self.peak_mb = max(self._samples) if self._samples else self.baseline_mb
        self.delta_mb = self.peak_mb - self.baseline_mb

    def result(self) -> dict:
        return {
            "baseline_mb": round(self.baseline_mb, 2),
            "peak_mb":     round(self.peak_mb,     2),
            "delta_mb":    round(self.delta_mb,     2),
        }


# ---------------------------------------------------------------------------
# Failu palīgfunkcijas
# ---------------------------------------------------------------------------

def list_images(folder) -> list[str]:
    """Atgriež šķirotu sarakstu ar attēlu ceļiem dotajā mapē."""
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(
        str(p) for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


# ---------------------------------------------------------------------------
# Metriku aprēķins
# ---------------------------------------------------------------------------

def compute_metrics(per_image_results: list[dict], total_time_s: float) -> dict:
    """
    Aprēķina 8 rādītājus no per-attēla rezultātu saraksta.

    Katram elementam per_image_results jāsatur lauks 'outcome':
        'correct' | 'wrong' | 'rejected' | 'no_face'

    total_time_s — kopējais testa laiks sekundēs (tikai test kopa, bez galerijas
    un sliekšņa kalibrācijas).
    """
    n = len(per_image_results)
    if n == 0:
        return {}

    counts = {"correct": 0, "wrong": 0, "rejected": 0, "no_face": 0}
    for r in per_image_results:
        key = r.get("outcome", "no_face")
        counts[key] = counts.get(key, 0) + 1

    time_per_img_ms = (total_time_s * 1000.0) / n if n > 0 else 0.0
    fps = n / total_time_s if total_time_s > 0 else 0.0

    return {
        "execution_time_ms_per_image": round(time_per_img_ms, 4),
        "throughput_fps":              round(fps, 4),
        "correct_pct":                 round(counts["correct"]  / n * 100, 4),
        "wrong_pct":                   round(counts["wrong"]    / n * 100, 4),
        "rejected_pct":                round(counts["rejected"] / n * 100, 4),
        "no_face_pct":                 round(counts["no_face"]  / n * 100, 4),
        "n_test_images": n,
        "n_correct":   counts["correct"],
        "n_wrong":     counts["wrong"],
        "n_rejected":  counts["rejected"],
        "n_no_face":   counts["no_face"],
    }


# ---------------------------------------------------------------------------
# JSON shēma un saglabāšana
# ---------------------------------------------------------------------------

def build_result_dict(
    system: str,
    run_number: int,
    metric_type: str,
    threshold: float,
    threshold_sweep: list[dict],
    metrics: dict,
    memory_result: dict,
    per_image_results: list[dict],
) -> dict:
    """
    Izveido pilnu JSON rezultātu vārdnīcu.

    metric_type: 'cosine_similarity' vai 'euclidean_distance'

    threshold_sweep: saraksts ar dikt {'T', 'correct', 'wrong', 'rejected', 'no_face'}
        — visi pārbaudītie T vērtības uz val kopas (kalibrācijas līknei).

    per_image_results: saraksts ar dikt (skat. run skriptus):
        'path', 'true_label', 'predicted_label', 'best_gallery_match',
        'score', 'outcome'
    """
    return {
        "system":           system,
        "run":              run_number,
        "timestamp":        datetime.now().isoformat(),
        "metric_type":      metric_type,
        "threshold":        threshold,
        "threshold_sweep":  threshold_sweep,
        "summary":          {**metrics, **memory_result},
        "per_image":        per_image_results,
    }


def save_json(result_dict: dict, path=None) -> Path:
    """Saglabā rezultātu vārdnīcu JSON failā."""
    if path is None:
        RESULTS_RAW_DIR.mkdir(parents=True, exist_ok=True)
        fname = f"{result_dict['system']}_run{result_dict['run']}.json"
        path = RESULTS_RAW_DIR / fname
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=2)
    print(f"  JSON saglabats: {path}")
    return path


# ---------------------------------------------------------------------------
# CSV izvade
# ---------------------------------------------------------------------------

def metrics_to_csv_rows(summary: dict, threshold: float) -> list[tuple]:
    """Pārveido summary vārdnīcu uz [(kritērijs, vērtība)] sarakstu."""
    return [
        ("Izpildes laiks (ms)",          f"{summary.get('execution_time_ms_per_image', 0):.4f}"),
        ("Caurlaidspeja (fps)",           f"{summary.get('throughput_fps',              0):.4f}"),
        ("Pareiza identifikacija (%)",    f"{summary.get('correct_pct',                 0):.4f}"),
        ("Klūdaina identifikacija (%)",   f"{summary.get('wrong_pct',                   0):.4f}"),
        ("Noraidita identifikacija (%)",  f"{summary.get('rejected_pct',                0):.4f}"),
        ("Seja nav atrasta (%)",          f"{summary.get('no_face_pct',                 0):.4f}"),
        ("Atminas paterings (MB)",        f"{summary.get('peak_mb',                     0):.2f}"),
        ("Slieksnis T",                   f"{threshold:.5f}"),
    ]


def save_csv(rows: list[tuple], path) -> Path:
    """Saglabā CSV ar formatejumu Kritērijs,Vērtība."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        f.write("Kritērijs,Vērtība\n")
        for k, v in rows:
            f.write(f"{k},{v}\n")
    print(f"  CSV saglabats: {path}")
    return path


# ---------------------------------------------------------------------------
# Ekrāna izvade
# ---------------------------------------------------------------------------

def print_metrics_table(metrics: dict, threshold: float, system_name: str):
    """Izdrukā rādītāju tabulu uz ekrāna."""
    sep = "=" * 50
    print(f"\n{sep}")
    print(f"  {system_name} — Rezultati")
    print(sep)
    rows = [
        ("Izpildes laiks (ms)",         f"{metrics.get('execution_time_ms_per_image', 0):.2f}"),
        ("Caurlaidspeja (fps)",          f"{metrics.get('throughput_fps',              0):.2f}"),
        ("Pareiza identifikacija (%)",   f"{metrics.get('correct_pct',                 0):.2f}"),
        ("Kludaina identifikacija (%)",  f"{metrics.get('wrong_pct',                   0):.2f}"),
        ("Noraidita identifikacija (%)", f"{metrics.get('rejected_pct',                0):.2f}"),
        ("Seja nav atrasta (%)",         f"{metrics.get('no_face_pct',                 0):.2f}"),
        ("Atminas paterings (MB)",       f"{metrics.get('peak_mb',                     0):.2f}"),
        ("Slieksnis T",                  f"{threshold:.5f}"),
    ]
    for k, v in rows:
        print(f"  {k:<36} {v}")
    print(sep + "\n")
