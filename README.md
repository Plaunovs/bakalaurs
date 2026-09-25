# Atvērtā koda sejas atpazīšanas sistēmu analīze kontrolētos apstākļos

Bakalaura darba praktiskā daļa · Rīgas Tehniskā universitāte, 2026
Autors: Dāniels Plaunovs · Zinātniskais vadītājs: asoc. prof. Aleksandrs Sisojevs

Trīs atvērtā koda sejas atpazīšanas sistēmas un paša izstrādāts hibrīdrisinājums, salīdzināti identiskos apstākļos: viens dators, tikai CPU, viena datu kopa, vienots sliekšņa kalibrēšanas process.

| Sistēma | Detektors | Izlīdzināšana | Iegultnes | Līdzība |
|---|---|---|---|---|
| face_recognition | dlib CNN | 68 atslēgpunkti | ResNet-34, 128D | Eiklīda attālums |
| DeepFace | MTCNN | 5 atslēgpunkti | VGG-Face, 4096D | Kosinusa līdzība |
| InsightFace | RetinaFace | 5 atslēgpunkti | ArcFace, 512D | Kosinusa līdzība |
| **Hibrīds** | RetinaFace | dlib 68 atslēgpunkti | ArcFace, 512D | Kosinusa līdzība |

---

## Rezultāti

| Kritērijs | face_recognition | DeepFace | InsightFace | Hibrīds |
|---|---|---|---|---|
| Izpildes laiks (ms) | 461,19 | 1759,42 | 482,01 | **333,51** |
| Caurlaidspēja (fps) | 2,18 | 0,57 | 2,07 | **3,00** |
| Pareiza identifikācija (%) | 95,00 | 95,00 | **98,33** | **98,33** |
| Kļūdaina identifikācija (%) | 0,00 | 0,00 | 0,00 | 0,00 |
| Noraidīta identifikācija (%) | 1,67 | 3,33 | 1,67 | 1,67 |
| Seja nav atrasta (%) | 3,33 | 1,67 | **0,00** | **0,00** |
| Atmiņas patēriņš (MB) | **204,04** | 1409,32 | 603,23 | 579,63 |

*Vidējās vērtības no trim palaišanām.*

- Hibrīds sasniedza InsightFace precizitāti un bija **ātrākais** no visām sistēmām, aptuveni 31 % ātrāks par InsightFace.
- **Nevienā sistēmā netika pieļauta kļūdaina identifikācija**, jo slieksnis tika kalibrēts pēc drošības principa.
- Datu kopa ir neliela, tāpēc precizitātes atšķirība starp 95,00 % un 98,33 % atbilst tikai dažiem testa attēliem. Nozīmīgāki ir ātruma rādītāji un kļūdu raksturs.
- Sistēmu kļūdas atšķiras pēc rakstura. face_recognition biežāk neatrod seju, DeepFace biežāk noraida identifikāciju, bet InsightFace atrod sejas visos attēlos. Tas kalpoja par pamatu hibrīda komponenšu izvēlei.

### Kāpēc hibrīds ir ātrāks

Hibrīds izsauc tikai trīs nepieciešamās komponentes (RetinaFace, dlib izlīdzināšanu un ArcFace) un izlaiž InsightFace iebūvētos papildu apstrādes soļus, kas pilnajā sistēmā tiek izpildīti automātiski.

---

## Metodoloģija

**Uzdevums:** 1:N identifikācija ar noraidīšanu. Testa attēlam tiek atrasta atbilstošā persona galerijā, vai arī attēls tiek klasificēts kā neatpazīts.

- **train:** atsauču galerija. Katrai personai tiek saglabāts visu train attēlu iegultņu vidējais vektors.
- **val:** sliekšņa T kalibrēšana katrai sistēmai atsevišķi. Tiek izvēlēta mazākā vērtība, pie kuras validācijas kopā nav nevienas kļūdainas identifikācijas.
- **test:** neatkarīga precizitātes un kļūdu novērtēšana.

Izpildes laiks tiek mērīts no attēla ielādes līdz identifikācijas lēmumam. Atmiņas patēriņš ir Python procesa izmantotā operatīvā atmiņa.

**Aparatūra:** Intel Core i5-1135G7, 8 GB RAM, bez diskrētā GPU, Windows 11.

---

## Prasības

- Windows 10/11
- [Anaconda](https://www.anaconda.com/download) (vai Miniconda)
- ~5 GB brīvas vietas (modeļi un vides)

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

Katras vides izveide aizņem 5–15 minūtes.

### Zināmas problēmas pēc instalācijas

**face_recognition: `No module named 'pkg_resources'`**

Jaunākajās conda versijās `setuptools` dažreiz nav pareizi reģistrēts. Risinājums:

```bat
conda activate env_face_recognition
pip install --force-reinstall setuptools==68.2.2
```

Ja kļūda joprojām parādās, jāizlabo fails
`envs\env_face_recognition\Lib\site-packages\face_recognition_models\__init__.py`, aizstājot tā saturu ar:

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

Lejupielādē modeli un novieto to projekta saknes mapē:
- Adrese: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
- Atarhivē un kopē `shape_predictor_68_face_landmarks.dat` repozitorija saknes mapē.

---

## Datu kopa

Attēli repozitorijā nav iekļauti, jo tie satur personu sejas un ir iegūti no trešo pušu datu kopām ar savām licencēm. Izmantotie avoti:

- [Object Detection: Obama Dataset](https://www.kaggle.com/datasets/jipingsun/object-detection-obama) (Kaggle)
- [Caltech-101](https://www.kaggle.com/datasets/imbikramsaha/caltech-101) (Kaggle)
- [Occluded and Masked Face Dataset](https://data.mendeley.com/datasets/znpyrgbfdr/1) (Mendeley Data)
- [Face Recognition Dataset](https://www.anefian.com/research/face_reco.htm) (A. Nefian)

Mapei `Dataset\` jābūt šādā struktūrā:

```
Dataset\
  person1\
    train\   (atsauces attēli galerijā)
    val\     (sliekšņa kalibrēšanai)
    test\    (gala novērtēšanai)
  person2\
    ...
```

Attēlu nosaukumu formāts: `person_X_AA.jpg`, kur X ir personas numurs un AA ir attēla numurs.

Ja datu kopa atrodas citā vietā, pievieno `--dataset` argumentu:

```bat
python face_recognition/run_face_recognition.py --run 1 --dataset C:\mans_cels\Dataset
```

---

## Eksperimentu palaišana

Katra sistēma tiek palaista 3 reizes, un gala rezultāti ir trīs palaišanu vidējās vērtības:

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

conda activate env_hybrid
python Hybrid/run_hybrid.py --run 1
python Hybrid/run_hybrid.py --run 2
python Hybrid/run_hybrid.py --run 3
```

Pirmā palaišana DeepFace, InsightFace un hibrīdam parasti ir lēnāka, jo modeļi tiek ielādēti atmiņā. Stabiliem mērījumiem ieteicams aizvērt citas programmas.

Rezultāti tiek saglabāti mapē `results\raw\` kā JSON faili.

---

## Grafiku un tabulu ģenerēšana

```bat
conda activate env_analysis
python analysis/analyze_results.py
```

Tikai vienas sistēmas grafiki:

```bat
python analysis/analyze_results.py --system face_recognition
```

Izvade:
- `results\figures\`: grafiki PNG un PDF formātā
- `results\tables\`: CSV tabulas

---

## Projekta struktūra

```
bakalaurs\
  Dataset\                        datu kopa (nav iekļauta repozitorijā)
  face_recognition\
    run_face_recognition.py       dlib CNN + ResNet-34 128D
  DeepFace\
    run_deepface.py               MTCNN + VGG-Face 4096D
  InsightFace\
    run_insightface.py            RetinaFace + ArcFace 512D
  Hybrid\
    run_hybrid.py                 RetinaFace + dlib 68pt + ArcFace 512D
  shared\                         kopīgās palīgfunkcijas
  analysis\                       grafiku un tabulu ģenerēšana
  envs\                           conda vižu YAML faili
  results\
    raw\                          JSON rezultāti katrai palaišanai
    figures\                      grafiki
    tables\                       CSV tabulas
```

---

## Ierobežojumi un turpmākie virzieni

- Neliela datu kopa (15 personas) ar nevienmērīgu attēlu skaitu uz personu.
- Mērījumi veikti tikai uz CPU. Uz GPU sistēmu relatīvā pozīcija ātrumā var mainīties.
- Slieksnis kalibrēts pēc stingra drošības principa, kas palielina noraidījumu skaitu.

Turpmāk būtu vērts izmantot lielāku datu kopu, pievienot citas sistēmas (piemēram, OpenFace 2.0 vai SeetaFace), izmēģināt adaptīvu slieksni un atkārtot eksperimentus uz GPU.

---

## Tiesiskais konteksts

Sejas attēli ir biometriskie dati VDAR (GDPR) izpratnē. Praktiskā ieviešanā jāievēro VDAR, ES Mākslīgā intelekta akts un nacionālais regulējums.

## Izmantotās bibliotēkas

- [face_recognition](https://github.com/ageitgey/face_recognition) (A. Geitgey)
- [DeepFace](https://github.com/serengil/deepface) (S. Serengil)
- [InsightFace](https://github.com/deepinsight/insightface) (J. Guo, J. Deng)
