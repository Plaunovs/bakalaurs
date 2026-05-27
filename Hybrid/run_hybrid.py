"""
run_hybrid.py — Hibrīdā sistēma.

Komponentes:
  Detektors:    InsightFace RetinaFace (buffalo_l)  — 0% seja nav atrasta
  Izlīdzināšana: dlib 68 atslēgpunktu modelis       — precīzāka sejas poza
  Iegultnes:    ArcFace 512D (InsightFace buffalo_l) — 98.33% precizitāte
  Metrika:      kosinusa līdzība

Pipeline:
  RetinaFace → bounding box → dlib 68pt → afīnā transf. → 112×112 → ArcFace

Palaišana:
    conda activate env_hybrid_dlib
    python Hybrid/run_hybrid.py --run 1 [--dataset Dataset]
"""

import sys
import os
import argparse
import time

import numpy as np
import cv2
import dlib
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

SYSTEM_NAME = "hybrid"
METRIC_TYPE = "cosine_similarity"
DET_THRESH  = 0.50
DET_SIZE    = (640, 640)
MAX_DIM     = 640
T_MIN       = 0.30
T_MAX       = 0.80
T_STEP      = 0.001

# ArcFace 112×112 šablons (5 atslēgpunkti)
_ARCFACE_DST = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)

# Dlib modeļa ceļš — meklē projektā vai lietotāja norādītā vidē
_DLIB_CANDIDATES = [
    os.path.join(os.path.dirname(_PROJECT_ROOT), "shape_predictor_68_face_landmarks.dat"),
    os.path.join(_PROJECT_ROOT, "shape_predictor_68_face_landmarks.dat"),
]

# Lai vajadzības gadījumā var izmantot pārvaldītu ceļu no vides mainīgā
if os.environ.get("DLIB_SHAPE_PREDICTOR"):
    _DLIB_CANDIDATES.insert(0, os.environ["DLIB_SHAPE_PREDICTOR"])

_app: FaceAnalysis | None = None
_rec_model = None
_predictor: dlib.shape_predictor | None = None


def _find_dlib_model() -> str:
    for p in _DLIB_CANDIDATES:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        "shape_predictor_68_face_landmarks.dat nav atrasts.\n"
        "Lejupielādē: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2\n"
        "Novieto šo failu projektā vai iestatiet vides mainīgo DLIB_SHAPE_PREDICTOR.\n"
        f"Meklēts: {_DLIB_CANDIDATES}"
    )


def get_models():
    global _app, _rec_model, _predictor
    if _app is None:
        _app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        _app.prepare(ctx_id=0, det_thresh=DET_THRESH, det_size=DET_SIZE)

        _rec_model = None
        for _, model in _app.models.items():
            if hasattr(model, "get_feat"):
                _rec_model = model
                break
        if _rec_model is None:
            raise RuntimeError("Nevar atrast ArcFace recognition modeli buffalo_l!")

        dlib_path = _find_dlib_model()
        _predictor = dlib.shape_predictor(dlib_path)
        print(f"  dlib modelis: {dlib_path}")

    return _app.det_model, _rec_model, _predictor


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


def l2n(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-10 else v


def align_face_112(img_bgr: np.ndarray, bbox, predictor) -> tuple:
    """
    Izlīdzina seju uz 112×112 ArcFace šablonu, izmantojot dlib 68pt.
    Atgriež (aligned_bgr, 'ok') vai (None, kļūdas_kods).
    """
    h, w = img_bgr.shape[:2]
    x1 = max(0, int(bbox[0]))
    y1 = max(0, int(bbox[1]))
    x2 = min(w, int(bbox[2]))
    y2 = min(h, int(bbox[3]))

    if x2 <= x1 or y2 <= y1:
        return None, "invalid_bbox"

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    rect = dlib.rectangle(x1, y1, x2, y2)
    shape = predictor(gray, rect)

    if shape.num_parts != 68:
        return None, "dlib_failed"

    pts = np.array([[shape.part(i).x, shape.part(i).y]
                    for i in range(68)], dtype=np.float32)

    # 5 galvenie punkti no 68 atslēgpunktiem
    src_pts = np.array([
        pts[36:42].mean(axis=0),   # kreisās acs centrs
        pts[42:48].mean(axis=0),   # labās acs centrs
        pts[30],                   # deguna gals
        pts[48],                   # kreisais mutes stūris
        pts[54],                   # labais mutes stūris
    ], dtype=np.float32)

    M, _ = cv2.estimateAffinePartial2D(src_pts, _ARCFACE_DST, method=cv2.LMEDS)
    if M is None:
        return None, "transform_failed"

    aligned = cv2.warpAffine(img_bgr, M, (112, 112), borderValue=0)
    return aligned, "ok"


def encode_one(img_path: str):
    """
    Pilnais hibrīda pipeline: RetinaFace → dlib 68pt → afīnā izlīdz. → ArcFace.
    Atgriež L2-normalizētu 512D vektoru vai None.
    """
    det_model, rec_model, predictor = get_models()

    img_bgr = load_and_downscale_bgr(img_path)
    if img_bgr is None:
        return None

    # 1. RetinaFace detekcija
    bboxes, _ = det_model.detect(img_bgr)
    if bboxes is None or bboxes.shape[0] == 0:
        return None

    areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in bboxes]
    bbox  = bboxes[int(np.argmax(areas))]

    # 2. dlib 68pt + afīnā izlīdzināšana → 112×112
    aligned, status = align_face_112(img_bgr, bbox, predictor)
    if aligned is None:
        return None   # dlib neizdevās — NEatkrīt pie 5pt modeļa

    # 3. ArcFace 512D iegultne
    feat = rec_model.get_feat(aligned)   # (1, 512)
    return l2n(feat[0].astype(np.float32))


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
            print(f"  BRIDINAJUMS: {person} — nav derīgu iegultnu train datos")
    return gallery


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
    """Mazākais T, pie kura val kopā wrong == 0."""
    print("Kalibrēju slieksni uz val datiem...")
    val_rows = []
    for person in persons:
        for img_path in list_images(os.path.join(dataset_path, person, "val")):
            enc = encode_one(img_path)
            if enc is None:
                val_rows.append((person, None, None))
            else:
                bp, bs = _best_match_sim(enc, gallery)
                val_rows.append((person, bp, bs))

    T_values = np.arange(T_MIN, T_MAX + T_STEP / 2, T_STEP)
    sweep: list[dict] = []
    best_T: float | None = None

    for T in T_values:
        T = float(round(T, 3))
        c = w = r = nf = 0
        for true_lbl, bp, sim in val_rows:
            if bp is None:
                nf += 1
            elif sim >= T:
                if bp == true_lbl:
                    c += 1
                else:
                    w += 1
            else:
                r += 1
        sweep.append({"T": T, "correct": c, "wrong": w,
                      "rejected": r, "no_face": nf})
        if w == 0 and best_T is None:
            best_T = T

    if best_T is None:
        best_T = T_MAX
        print(f"  BRIDINAJUMS: Nav T bez kludam ID. Fallback T={best_T:.3f}")
    print(f"  Izveletais slieksnis T = {best_T:.5f}")
    return best_T, sweep


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
            bp, bs = _best_match_sim(enc, gallery)
            if bs >= threshold:
                predicted = bp
                outcome = "correct" if predicted == person else "wrong"
            else:
                predicted = "rejected"
                outcome = "rejected"
            per_image.append({
                "path": img_path, "true_label": person,
                "predicted_label": predicted,
                "best_gallery_match": bp,
                "score": round(bs, 6),
                "outcome": outcome,
            })

    return per_image, time.perf_counter() - t0


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

    _ = get_models()

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
