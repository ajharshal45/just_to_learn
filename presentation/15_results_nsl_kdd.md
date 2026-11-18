# 15 — NSL-KDD Network Intrusion Notebook: What Each Cell Shows

> Notebook: `notebooks/04_nsl_kdd_intrusion_experiment.ipynb`. The cells are explained in
> [11](11_notebook_walkthrough_cell_by_cell.md); this file gives **this dataset's actual outputs**, cell by cell,
> in plain words. The rare class is **a network attack** (hacking attempt).

---

## Setup and data (cells 1–5)

* **Cell 2:** `DATASET = "nsl_kdd"`. Settings identical to the other notebooks.
* **Cell 4:** NSL-KDD, a classic network-security dataset from the University of New Brunswick (Canada). Each row
  describes one network connection: how long it lasted, how many bytes were sent each way, error rates, how many
  connections went to the same service, and three **category** columns: protocol (`tcp`/`udp`/`icmp`), service
  (`http`, `ftp`, … 70 types) and connection status (`flag`).
  What the loader changed:
  * dropped the `difficulty` column (a note from the dataset's authors, not a network feature), 9 duplicate rows,
    and one column that is always 0 (`num_outbound_cmds`);
  * in the original file **47% of connections are attacks**, which is not rare. To create a rare-event problem we
    **kept a fixed random 5,856 of the 58,621 attacks**, so attacks are 8% of the data. All normal traffic is kept.

  Result: **73,199 rows × 40 features**.
* **Cell 5:** 37 numeric and 3 categorical features. Nothing missing. 4,941 rows have extreme values (traffic data
  is very spiky); they are kept.

## Looking at the data (cells 7–9)

* **Cell 7:** 67,343 normal connections and **5,856 attacks (8.0%)**, imbalance **11.5 : 1**.
* **Cell 8:** The most separating features: **bytes sent by the source (`src_bytes`, KS 0.84)**, bytes sent back
  (`dst_bytes`, 0.81), the share of connections to the same service (0.72), to different services (0.72), and
  `dst_host_srv_count` (0.71). In the histograms, many attacks send **zero bytes** and hit many different services,
  which normal traffic rarely does: attacks are **easy to spot**.
* **Cell 9:** The three connection-error-rate features move together very closely (correlation 0.88–0.93),
  which shows as a dark red block.

## Preparing the data (cell 11)

* Development set: **54,899 connections, 4,392 attacks**. Test set: **18,300 connections, 1,464 attacks**.
* The 3 category columns become 0/1 columns (service alone has 70 types), giving **116 model columns**.

## Making the fake rows (cells 13–24)

* **Cell 13:** Every method turns **4,392 real attacks into 50,507**, adding **46,115 fake attacks**.
  Borderline-SMOTE starts from only **86 DANGER attacks**. SHAP-Filtered SMOTE judges similarity on **23 of the 116
  columns**.
* **Cell 15:** Borderline labels: **4,273 SAFE, 86 DANGER, 33 NOISE**: almost all attacks sit far from normal traffic.
  Neighbour overlap with SMOTE: **SHAP-Filtered 0.51, SHAP-Weighted 0.50**, so SHAP changes about half of every
  attack's neighbours.
* **Cell 18:** Fake row #0 of SMOTE, SHAP-Filtered and SHAP-Weighted all start from attack **row 59752** with
  λ = 0.4389. Partners: row 77608 (SMOTE), 39773 (SHAP-Filtered), 81192 (SHAP-Weighted).
* **Cell 20:** SHAP's top features: **`src_bytes` (0.10), `diff_srv_rate` (0.09), `dst_host_serror_rate` (0.06),
  `same_srv_rate` (0.06)**, then several error rates and two connection-status flags (`flag=SF`, `flag=S0`). The
  importance is spread over many features, unlike Stroke where age dominates.
* **Cell 21:** In the first 8 fake rows of each SHAP method, **50%** of parent pairs would **not** have been
  neighbours in plain SMOTE.
* **Cell 23 (quality):**

  | Method | Normal-traffic neighbours (real attacks: 2.1%) | Realistic spread (KS pass) | Correlations kept | Detectability (0.5 = can't tell) | Category columns stuck between 0 and 1 (per row) |
  |---|---|---|---|---|---|
  | SMOTE | 1.2% | 95% | 0.90 | 0.87 | 0.13 |
  | Borderline-SMOTE | **31.7%** | **61%** | **0.44** | **0.97** | 0.66 |
  | SHAP-Filtered SMOTE | 1.5% | 95% | 0.90 | 0.88 | 0.92 |
  | SHAP-Weighted SMOTE | 1.3% | 95% | 0.90 | 0.88 | 0.84 |

  In plain words: SMOTE and the SHAP methods put fake attacks firmly in attack territory. All of them are fairly
  easy to tell from real attacks (about 0.87): interpolating spiky traffic counts gives values like "312.6 bytes"
  that real connections rarely have. On the number columns alone, the SHAP methods are slightly *more* realistic
  (0.85–0.86 vs 0.89). As on Stroke, SHAP pairing leaves more in-between category values (0.84–0.92 vs 0.13), but
  here it doesn't hurt (see below). Borderline-SMOTE's fakes land in normal traffic and look fake.
* **Cell 24:** All rows saved to `results/nsl_kdd/synthetic_samples/` (46,115 per method; about 17 MB each, so not
  stored in GitHub).

## The fair test (cells 26–29) — 10 rounds, 25 minutes

Average over the 10 rounds (cell 27):

| Model | Method | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|---|
| Logistic Regression | **No Sampling** | **0.98** | 0.93 | **0.95** | 0.976 |
| | SMOTE | 0.80 | 0.96 | 0.88 | 0.976 |
| | SHAP-Filtered / SHAP-Weighted | 0.80 / 0.80 | 0.96 / 0.96 | 0.88 / 0.87 | 0.976 / 0.976 |
| | Borderline-SMOTE | 0.69 | 0.96 | 0.81 | 0.940 |
| Random Forest | No Sampling | 1.00 | 0.90 | 0.95 | 0.994 |
| | **Random Oversampling** | 0.98 | 0.97 | **0.97** | 0.995 |
| | SMOTE | 0.97 | 0.97 | 0.97 | 0.995 |
| | SHAP-Filtered / SHAP-Weighted | 0.97 / 0.97 | 0.97 / 0.97 | 0.97 / 0.97 | 0.995 / 0.995 |
| | Borderline-SMOTE | 0.78 | 0.99 | 0.87 | 0.992 |
| XGBoost | **No Sampling** | **1.00** | 0.99 | **0.99** | **0.999** |
| | SMOTE, both SHAP methods, Random Oversampling | 0.99 | 0.99 | 0.99 | 0.999 |

* **Cell 28 (Graph 1):** the bars almost touch the top. XGBoost is nearly perfect with or without oversampling.
* **Cell 29 (significance), SHAP methods vs SMOTE:** the differences are tiny (all below 0.01). Two are significant,
  and they point in opposite directions:
  * SHAP-Weighted + XGBoost: recall **+0.0015** (about one more attack caught per round), p = 0.016.
  * SHAP-Weighted + Logistic Regression: precision **−0.004** and F1 **−0.002** (p = 0.010 and 0.049).

## The locked test set (cells 31–32) — 1,464 attacks among 18,300 connections

| Random Forest | Attacks caught | Missed | False alarms |
|---|---|---|---|
| No Sampling | 1,346 | 118 | **0** |
| SMOTE | 1,417 | 47 | 18 |
| SHAP-Filtered SMOTE | 1,419 | 45 | 55 |
| SHAP-Weighted SMOTE | 1,419 | 45 | 41 |
| Borderline-SMOTE | 1,451 | **13** | 446 |

* For Random Forest, oversampling catches about 70 more attacks at the cost of a few dozen false alarms; that is a
  good trade.
* XGBoost: about 1,450 of 1,464 caught by every method, with 2–37 false alarms.
* Logistic Regression: oversampling catches about 55 more attacks but adds about 340 false alarms (Borderline-SMOTE:
  about 720 more).

## What the models pay attention to (cells 34–35)

* **Cell 34 (Graph 2):** models lean on `src_bytes`, `dst_host_srv_count`, `dst_bytes` and `count`. Balancing barely
  changes their reasoning (ranking similarity to No Sampling 0.83–0.95).
* **Cell 35:** explanations are very stable for XGBoost (Kendall τ about 0.9) and stable for Random Forest (about 0.8).
  The SHAP methods are as stable as SMOTE for XGBoost and slightly less for Random Forest (0.80–0.81 vs 0.83).

## Conclusion (cells 39–40), in plain words

* **Attacks are easy to detect:** XGBoost with no oversampling is nearly perfect (F1 0.99, PR-AUC 0.999).
* **Oversampling helps Random Forest** (more attacks caught) and **hurts Logistic Regression** (many more false alarms).
* **SHAP-guided SMOTE changed half of the neighbour pairs but made no practical difference** to the results: SMOTE's
  neighbours were already sensible because attacks form tight, well-separated groups.
* **Borderline-SMOTE was clearly the worst**: its fake attacks land in normal traffic and cause hundreds of false alarms.
* SHAP guidance made oversampling about 5× slower (9–11 s vs 1.8 s per round).

**One sentence for the slide:** *"On network attacks, the problem is close to solved, and SHAP guidance changes which
attacks get paired but not the results."*
