# Sejas atpazīšanas sistēmu salīdzinājums

Bakalaura darba praktiskā daļa. Trīs atvērtā koda sejas atpazīšanas sistēmas salīdzinātas identiskiem apstākļos (viens dators, tikai CPU, viena datu kopa).

Salīdzinātās sistēmas:
- **face_recognition** — dlib CNN detektors + ResNet-34 128D + Eiklīda attālums
- **DeepFace** — MTCNN detektors + VGG-Face 4096D + kosinusa līdzība
- **InsightFace** — RetinaFace detektors + ArcFace 512D + kosinusa līdzība
- **Hibrīds** — RetinaFace detektors + dlib 68pt izlīdzināšana + ArcFace 512D

---

## Prasības

- Windows 10/11
- [Anaconda](https://www.anaconda.com/download) (vai Miniconda)
- ~5 GB brīvas vietas (modeļi + vides)

---

## Uzstādīšana

Katrai sistēmai ir sava conda vide, jo atkarības savstarpēji konfliktē.

```bat
conda env create -f envs/env_face_recognition.yml
conda env create -f envs/env_deepface.yml
conda env create -f envs/env_insightface.yml
conda env create -f envs/env_hybrid.yml
conda env create -f envs/env_analysis.yml
```

Vides izveide aizņem 5–15 minūtes katrai.

### Zināmas problēmas pēc instalācijas

**face_recognition: `No module named 'pkg_resources'`**

Jaunākos conda builtos `setuptools` dažreiz nav pareizi reģistrēts. Risinājums:

```bat
conda activate env_face_recognition
pip install --force-reinstall setuptools==68.2.2
```

Ja kļūda joprojām parādās, jāizlabo fails
`envs\env_face_recognition\Lib\site-packages\face_recognition_models\__init__.py`
— aizstāj saturu ar:

```python
from pathlib import Path
_MODELS = Path(__file__).parent / "models"
def pose_predictor_model_location():
    return str(_MODELS / "shape_predictor_68_face_landmarks.dat")
def pose_predictor_five_point_model_location():
    return str(_MODELS / "shape_predictor_5_face_landmarks.dat")
def face_recognition_model_location():
    return str(_MODELS / "dlib_face_recognition_resnet_model_v1.dat")
def cnn_face_detector_model_location():
    return str(_MODELS / "mmod_human_face_detector.dat")
```

**InsightFace: numpy versiju konflikts**

Ja rodas kļūda par `CV_8U` vai `_ARRAY_API not found`:

```bat
conda activate env_insightface
pip uninstall opencv-python -y
pip install "numpy>=2" "onnxruntime>=1.18" opencv-python-headless --force-reinstall
```

**Hibrīds: `shape_predictor_68_face_landmarks.dat nav atrasts`**

Lejupielādē modeli un novieto projekta mapē:
- Adrese: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
- Atarhivē un kopē `shape_predictor_68_face_landmarks.dat` uz `Daniels_Plaunovs_221RDB422_PIELIKUMS\`

---

## Datu kopa

Mapei `Dataset\` jābūt šādā struktūrā:

```
Dataset\
  person1\
    train\   (atsauces attēli galerijā)
    val\     (sliekšņa kalibrēšanai)
    test\    (gala novērtēšana)
  person2\
    ...
```

Ja datu kopa atrodas citā vietā, pievieno `--dataset` argumentu:

```bat
python face_recognition/run_face_recognition.py --run 1 --dataset C:\mans_ceļš\Dataset
```

---

## Eksperimentu palaišana

Katra sistēma jāpalaiž 3 reizes (rezultāti vidējojas):

```bat
conda activate env_face_recognition
python face_recognition/run_face_recognition.py --run 1
python face_recognition/run_face_recognition.py --run 2
python face_recognition/run_face_recognition.py --run 3

conda activate env_deepface
python DeepFace/run_deepface.py --run 1
python DeepFace/run_deepface.py --run 2
python DeepFace/run_deepface.py --run 3

conda activate env_insightface
python InsightFace/run_insightface.py --run 1
python InsightFace/run_insightface.py --run 2
python InsightFace/run_insightface.py --run 3

conda activate env_hybrid_dlib
python Hybrid/run_hybrid.py --run 1
python Hybrid/run_hybrid.py --run 2
python Hybrid/run_hybrid.py --run 3
```

Rezultāti saglabājas `results\raw\` kā JSON faili.

---

## Grafiku un tabulu ģenerēšana

```bat
conda activate env_analysis
python analysis/analyze_results.py
```

Vai tikai vienas sistēmas grafiki:

```bat
python analysis/analyze_results.py --system face_recognition
```

Izvade:
- `results\figures\` — grafiki PNG un PDF formātā
- `results\tables\` — CSV tabulas

---

## Projekta struktūra

```
Daniels_Plaunovs_221RDB422_PIELIKUMS\
  Dataset\                        datu kopa
  face_recognition\
    run_face_recognition.py       dlib CNN + ResNet-34 128D
  DeepFace\
    run_deepface.py               MTCNN + VGG-Face 4096D
  InsightFace\
    run_insightface.py            RetinaFace + ArcFace 512D
  Hybrid\
    run_hybrid.py                 RetinaFace + dlib 68pt + ArcFace 512D
  shared\                         kopīgs utilities modulis
  analysis\                       grafiku un tabulu ģenerēšana
  envs\                           conda vides YAML faili
  results\
    raw\                          JSON rezultāti (katrai palaišanai)
    figures\                      grafiki
    tables\                       CSV tabulas
```
