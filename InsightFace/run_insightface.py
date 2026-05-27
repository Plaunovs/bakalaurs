"""
run_insightface.py — InsightFace sistēma (RetinaFace detektors + ArcFace 512D).

Metrika: kosinusa līdzība (lielāka = labāka).
Slieksnis:  pieņem, ja sim >= T; izvēlas MAZĀKO T bez kļūdainām ID uz val.

Palaišana:
    conda activate env_insightface
    python InsightFace/run_insightface.py --run 1 [--dataset Dataset]
"""

import sys
import os
import argparse
import time

import numpy as np
import cv2
from insightface.app import FaceAnalysis

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

from shared.shared_utils import (
    SEED, DEFAULT_DATASET, RESULTS_TABLES_DIR,
    MemoryMonitor, list_images,
    compute_metrics, build_result_dict,
    save_json, save_csv, metrics_to_csv_rows, print_metrics_table,
)

np.random.seed(SEED)

SYSTEM_NAME  = "insightface"
METRIC_TYPE  = "cosine_similarity"
DET_THRESH   = 0.50
DET_SIZE     = (640, 640)
MAX_DIM      = 640
N_SWEEP      = 500

# Modelis tiek ielādēts vienu reizi — mēra atmiņu no šī brīža
_app: FaceAnalysis | None = None


def get_app() -> FaceAnalysis:
    global _app
    if _app is None:
        _app = FaceAnalysis(name="buffalo_l",
                            providers=["CPUExecutionProvider"])
        _app.prepare(ctx_id=0, det_thresh=DET_THRESH, det_size=DET_SIZE)
    return _app


# ---------------------------------------------------------------------------
# Attēlu ielāde
# ---------------------------------------------------------------------------

def load_and_downscale_bgr(img_path: str, max_dim: int = MAX_DIM):
    img = cv2.imread(img_path)
    if img is None:
        return None
    h, w = img.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)
    return img


# ---------------------------------------------------------------------------
# Iegultnes
# ---------------------------------------------------------------------------

def l2n(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-10 else v


def encode_one(img_path: str):
    """Atgriež L2-normalizētu 512D ArcFace vektoru vai None."""
    bgr = load_and_downscale_bgr(img_path)
    if bgr is None:
        return None
    app = get_app()
    faces = app.get(bgr)
    if not faces:
        return None
    # Izvēlas lielāko seju pēc bounding-box laukuma
    face = max(faces, key=lambda f: (
        (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
    ))
    return l2n(np.array(face.embedding, dtype=np.float32))


# ---------------------------------------------------------------------------
# Galerija
# ---------------------------------------------------------------------------

def build_gallery(dataset_path: str, persons: list[str]) -> dict:
    gallery = {}
    print("Veidoju galeriju no train attēliem...")
    for person in persons:
        imgs = list_images(os.path.join(dataset_path, person, "train"))
        vecs = [enc for enc in (encode_one(p) for p in imgs) if enc is not None]
        if vecs:
            gallery[person] = l2n(np.mean(vecs, axis=0).astype(np.float32))
            print(f"  {person}: {len(vecs)}/{len(imgs)} atteli -> iegultne")
        else:
            print(f"  BRIDINAJUMS: {person} — nav sejas train datos")
    return gallery


# ---------------------------------------------------------------------------
# Sliekšņa kalibrācija
# ---------------------------------------------------------------------------

def _best_match_sim(enc: np.ndarray, gallery: dict) -> tuple[str, float]:
    best_person, best_sim = None, -2.0
    for person, ref in gallery.items():
        sim = float(np.dot(enc, ref))
        if sim > best_sim:
            best_sim, best_person = sim, person
    return best_person, best_sim


def calibrate_threshold(
    dataset_path: str, persons: list[str], gallery: dict
) -> tuple[float, list[dict]]:
    """
    Kosinusa līdzība: pieņem, ja sim >= T.
    Izvēlas MAZĀKO T, pie kura uz val NAV kļūdainas ID.
    """
    print("Kalibrēju slieksni uz val datiem...")
    val_rows = []
    for person in persons:
        for img_path in list_images(os.path.join(dataset_path, person, "val")):
            enc = encode_one(img_path)
            if enc is None:
                val_rows.append((person, None, None))
            else:
                best_person, best_sim = _best_match_sim(enc, gallery)
                val_rows.append((person, best_person, best_sim))

    valid_sims = [r[2] for r in val_rows if r[2] is not None]
    if not valid_sims:
        print("  BRIDINAJUMS: Nav sejas val datos. Fallback T=0.50")
        return 0.50, []

    T_lo = max(0.0,  min(valid_sims) - 0.05)
    T_hi = min(1.0,  max(valid_sims) + 0.05)
    T_values = np.linspace(T_lo, T_hi, N_SWEEP)

    sweep: list[dict] = []
    best_T: float | None = None

    for T in T_values:
        T = float(T)
        c = w = r = nf = 0
        for true_lbl, best_person, sim in val_rows:
            if best_person is None:
                nf += 1
            elif sim >= T:
                if best_person == true_lbl:
                    c += 1
                else:
                    w += 1
            else:
                r += 1
        sweep.append({"T": round(T, 6), "correct": c, "wrong": w,
                      "rejected": r, "no_face": nf})
        if w == 0 and best_T is None:
            best_T = T

    if best_T is None:
        best_T = float(max(valid_sims))
        print(f"  BRIDINAJUMS: Nav T bez kludam ID. Fallback T={best_T:.5f}")
    print(f"  Izveletais slieksnis T = {best_T:.5f}")
    return best_T, sweep


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def run_test(
    dataset_path: str, persons: list[str], gallery: dict, threshold: float
) -> tuple[list[dict], float]:
    print("Novērtēju testa datus...")
    per_image: list[dict] = []
    t0 = time.perf_counter()

    for person in persons:
        for img_path in list_images(os.path.join(dataset_path, person, "test")):
            enc = encode_one(img_path)
            if enc is None:
                per_image.append({
                    "path": img_path, "true_label": person,
                    "predicted_label": "no_face",
                    "best_gallery_match": None, "score": None,
                    "outcome": "no_face",
                })
                continue
            best_person, best_sim = _best_match_sim(enc, gallery)
            if best_sim >= threshold:
                predicted = best_person
                outcome = "correct" if predicted == person else "wrong"
            else:
                predicted = "rejected"
                outcome = "rejected"
            per_image.append({
                "path": img_path, "true_label": person,
                "predicted_label": predicted,
                "best_gallery_match": best_person,
                "score": round(best_sim, 6),
                "outcome": outcome,
            })

    return per_image, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Galvenā funkcija
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=f"Palaiž {SYSTEM_NAME} 1:N sejas atpazīšanu.")
    parser.add_argument("--run", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--dataset", type=str, default=str(DEFAULT_DATASET))
    args = parser.parse_args()

    dataset_path = args.dataset
    persons = sorted(
        d for d in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, d))
    )
    print(f"\n{'='*50}")
    print(f"  {SYSTEM_NAME} — Palaišana {args.run}/3")
    print(f"  Dataset: {dataset_path}  ({len(persons)} personas)")
    print(f"{'='*50}")

    mem = MemoryMonitor()
    mem.start()

    # Modeļa ielāde notiek šeit — atmiņa jau tiek mērīta
    _ = get_app()

    gallery = build_gallery(dataset_path, persons)
    threshold, sweep = calibrate_threshold(dataset_path, persons, gallery)
    per_image, total_time_s = run_test(dataset_path, persons, gallery, threshold)

    mem.stop()
    mem_result = mem.result()

    metrics = compute_metrics(per_image, total_time_s)
    print_metrics_table({**metrics, **mem_result}, threshold, SYSTEM_NAME)

    result = build_result_dict(
        system=SYSTEM_NAME, run_number=args.run,
        metric_type=METRIC_TYPE, threshold=threshold,
        threshold_sweep=sweep, metrics=metrics,
        memory_result=mem_result, per_image_results=per_image,
    )
    save_json(result)

    RESULTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    save_csv(
        metrics_to_csv_rows({**metrics, **mem_result}, threshold),
        RESULTS_TABLES_DIR / f"{SYSTEM_NAME}_run{args.run}.csv",
    )
    print(f"\n{SYSTEM_NAME} palaišana {args.run} pabeigta.\n")


if __name__ == "__main__":
    main()
