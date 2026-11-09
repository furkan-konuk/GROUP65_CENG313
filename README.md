# GROUP65\_CENG313 — Fake or Real: The Impostor Hunt in Texts

|||
|-|-|
|**Member**|Yakup Furkan Konuk|
|**Student ID**|22118080089|
|**GitHub**|furkan-konuk|
|**Group ID**|65 (Light, 1 student)|
|**Kaggle competition**|[Fake or Real: The Impostor Hunt in Texts](https://www.kaggle.com/competitions/fake-or-real-the-impostor-hunt)|

## 1\. The competition

The data consists of pairs of LLM-generated summaries of articles from ESO's *The Messenger* journal.
In each pair, one summary was produced by a correctly working LLM (**real**) and the other by a
maliciously modified LLM (**fake**: e.g. hallucinated topics, odd tone, gibberish or non-Latin characters).
The goal is to predict which file of the pair (`1` or `2`) is the real one.

* **Input:** `train/article\\\_XXXX/file\\\_1.txt`, `file\\\_2.txt` (+ the same for `test/`)
* **Labels:** `train.csv` → `id, real\\\_text\\\_id`
* **Metric:** accuracy
* **Submission:** CSV with columns `id, real\\\_text\\\_id`

## 2\. Approach

### Assignment 1 — EDA \& baselines (`assignment1\\\_eda\\\_baseline.ipynb`)

* **EDA:** label balance, example pairs, per-text statistics (length, vocabulary richness,
Latin/non-ASCII character ratios, punctuation, exclamation marks), paired real-vs-fake comparison,
non-Latin character analysis, frequent words, train/test comparison.
* **Baselines** (5-fold stratified CV on train):

  1. Random guess
  2. Majority class
  3. Rule: the text with the higher share of Latin letters is real (idea from the organizers' baseline)
  4. Logistic regression on hand-crafted feature differences `f(text\\\_1) − f(text\\\_2)`
  5. Logistic regression on character n-gram TF-IDF differences
* Models 4–5 use **swap augmentation**: swapping the two texts negates the features and flips the label,
doubling the small training set.

### Assignment 2 — proposed solution (`assignment2\\\_solution.ipynb`)

*To be added.*

## 3\. Results

|Model|5-fold CV accuracy|Kaggle public score|
|-|-|-|
|Random guess|0.505 ± 0.092|–|
|Majority class|0.516 ± 0.021|–|
|Latin-ratio rule|0.611 ± 0.118|–|
|Feature-diff LogReg|0.895 ± 0.067|–|
|TF-IDF-diff LogReg|**0.937 ± 0.021**|*0.87136*|

Dataset: 95 labelled training pairs, 1,068 test pairs. Main EDA findings: fake texts are usually longer,
contain exclamation marks and line breaks, sometimes include Cyrillic/Arabic letters or drift off-topic;
the hardest fakes are near-copies of the real text with only a few words changed.

## 4\. Repository layout

```
GROUP65\\\_CENG313/
├── README.md
├── assignment1\\\_eda\\\_baseline.ipynb
├── assignment2\\\_solution.ipynb      (Assignment 2)
└── src/
    ├── data.py      # loading the article pairs + labels
    ├── models.py    # baseline models + cross-validation
    └── utils.py     # text features, seeding, submission writer
```

## 5\. How to run

1. Join the competition on Kaggle and accept the rules.
2. Kaggle → Settings → API → **Create New Token** → downloads `kaggle.json`.
3. Open `assignment1\\\_eda\\\_baseline.ipynb` in Google Colab
(File → Open notebook → GitHub → paste this repo's URL).
4. Set `REPO\\\_URL` in the first code cell, then **Runtime → Run all**.
When asked, upload `kaggle.json`; the data is downloaded into `data/` automatically.
5. The last section writes `submission.csv`; submit it on Kaggle (max 3 per day).

`data/` is not committed (see `.gitignore`).

## 6\. Progress log

|Date|Work|
|-|-|
|2026-10|Topic selected (group 65, light).|
|2026-11-09|Repository structure, data loader, feature extraction, baseline models.|
|2026-11-09|EDA run on full data, baseline CV results.|
|2026-11-09|First Kaggle submission.|

## 7\. Kaggle submission history

|Date|Model|Public score|
|-|-|-|
|*2026-11-09*|TF-IDF char n-gram LogReg|0.87136|



