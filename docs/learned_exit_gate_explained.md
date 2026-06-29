# Learned Exit Gate — Beginner-Friendly Explanation

A complete walkthrough of the **learned exit gate** experiment added to the cascade
part of this compute-aware ViT project.

**Source files referenced**
- Gate logic: [`src/training/cascade_eval.py`](../src/training/cascade_eval.py) —
  `build_gate_training`, `fit_logistic_gate`, `gate_stage_probs`, `eval_gate`,
  `sweep_gate_thresholds`, `stage_entropy_margin`, `_stage_feature_matrix`
- Orchestration / data split: [`src/train.py`](../src/train.py) — `_run_learned_gate`, `_cache_stage_probs`
- Config: [`configs/cascade.yaml`](../configs/cascade.yaml) · Runner: [`scripts/run_cascade.sh`](../scripts/run_cascade.sh)
- Results: [`checkpoints/cascade_clean_split/metrics.json`](../checkpoints/cascade_clean_split/metrics.json) (`learned_gate_results` section),
  [`results/cascade_clean_split_summary.md`](../results/cascade_clean_split_summary.md)

Every number below is taken directly from those files.

---

## 1. Big picture

**What is a cascade here?** A cascade is a chain of models ordered from cheapest to
most expensive. An image is shown to the cheap model first. If that model is "sure
enough," we accept its answer and **stop** — we never run the expensive models. If
it's not sure, we **pass the image up** to the next, more expensive model.

**The stages** (defined in `run_cascade`, path logic in `cascade_eval.py`):

| Stage | Model | What it is | GFLOPs (one model) |
|---|---|---|---|
| 25 | `static_25` | keeps 25% of tokens | 0.491 |
| 50 | `static_50` | keeps 50% of tokens | 0.687 |
| 75 | `static_75` | keeps 75% of tokens | 0.883 |
| dense | full ViT | keeps all tokens | 1.079 |

(These exact values are in `stage_flops_giga` in the metrics file.)

**What does "exit early" mean?** Accepting the prediction of an early (cheap) stage
and not running the later (expensive) stages for that image.

**Why does exiting early save FLOPs?** FLOPs = floating-point operations = the amount
of compute. A token-pruned model does fewer operations than the full model. If most
images exit at stage 25 (0.491 GFLOPs) instead of running the dense model (1.079
GFLOPs), the *average* cost per image drops.

**The four things, contrasted:**
- **Dense model** — always runs the full network on every image. Most accurate, most
  expensive. (Test: 0.795 acc @ 1.079 GFLOPs.)
- **Static pruning model** — a *single* model that always throws away a fixed fraction
  of tokens for *every* image. Cheaper, slightly less accurate. (e.g. static_75:
  0.7891 @ 0.883.)
- **Threshold-based cascade** — runs the chain and decides exit/continue using a simple
  rule: "is the model's confidence above a number?"
- **Learned-gate cascade** — same chain, but the exit/continue decision is made by a
  *small trained classifier* instead of a fixed confidence rule. **This is the thing
  this document explains.**

---

## 2. Threshold-based cascade (the baseline the gate is compared to)

**How it decides:** at each stage the model outputs class probabilities. Take the
**maximum** probability (the "confidence"). If `confidence >= threshold`, exit;
otherwise continue. Implemented in `eval_threshold_triple`:

```
exit at 25 if  cache['25_conf'] >= t25
else exit at 50 if cache['50_conf'] >= t50
else exit at 75 if cache['75_conf'] >= t75
else use dense
```

**What "0.7" or "0.8" means:** the model must be at least 70% (or 80%) sure of its top
class before we trust it. Higher threshold = stricter = fewer early exits.

**threshold_25 / threshold_50 / threshold_75:** each stage has its *own* threshold.
There is no `threshold_dense` because dense is the last resort — everything that
reaches it just takes its answer.

**What happens to one image:** stage 25 runs; if confident enough, done. Else stage 50;
if confident, done. Else stage 75; if confident, done. Else dense runs and we take its
answer no matter what.

**Why test many combinations?** Each `(t25, t50, t75)` triple gives a different
accuracy/cost trade-off. The code sweeps `[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]`
for each of the three thresholds = **8³ = 512 combinations** (`total_combinations: 512`).

**Why search on validation only, and test once?** If you pick thresholds using the test
set, you've effectively *tuned on the test set*, and your reported test accuracy becomes
optimistic/cheating. So: **choose** thresholds on validation, then **measure** the
chosen setting **one time** on test for an honest number.

---

## 3. Learned-gate motivation (why the supervisor suggested it)

**Limitation of a fixed confidence threshold:** it only looks at *one* number — the top
probability — and uses the *same* cutoff for everyone. The *shape* of the whole
probability distribution (how spread out it is, how close the 2nd-best class is) may
carry extra information about whether the prediction is actually right.

**Why a learned gate might be better:** instead of hand-picking one cutoff on one
feature, let a tiny model *learn* from data how to combine **several** signals into a
smarter "should I trust this prediction?" decision.

**"Exit now / continue":** the gate's job is a yes/no decision at each stage — **exit**
(accept this stage's answer) or **continue** (send the image to the next, costlier stage).

**What the gate learns:** to predict whether the current stage's prediction is **likely
correct**, using features of the prediction. If "likely correct," exit; else continue.

---

## 4. Learned-gate training data

**Which split is used?** Only the **validation** set (5000 images). The test set is
never used to build or tune the gate. Recorded in `gate_train_split`:
> "validation gate-fit subset (3000/5000, split_seed=42); threshold selected on the
> remaining 2000 validation samples"

**How validation is divided** (in `_run_learned_gate`):

```
perm = randperm(5000, seed=42)
n_fit = 0.6 * 5000 = 3000
fit_cache = first 3000 val samples   # used to TRAIN the gate
sel_cache = last  2000 val samples   # used to PICK the gate threshold
```

**"3000 to fit, 2000 to select" means:** 3000 validation images teach the
logistic-regression gate its weights. The other 2000 are held back to choose the gate's
operating threshold — so the threshold isn't chosen on the same data the gate was
trained on.

**Why not use test for gate training?** Same reason as thresholds: anything used to
build or tune the method must not be the data you report final numbers on.

**What is collected per sample per stage?** For each image and each exit stage
(25/50/75), the code stores (in `_cache_stage_probs`) the stage's **softmax
probabilities**, from which it derives four features (next section) plus whether that
stage's prediction was correct (the training label).

---

## 5. Learned-gate features (one by one)

`gate_features: ['max_confidence', 'entropy', 'top1_top2_margin', 'stage_id']`
(computed in `stage_entropy_margin` and `_stage_feature_matrix`).

**max_confidence** — the single biggest probability assigned to any class.
- Computed as `probs.max()`.
- Intuition: "how sure is the model of its best guess?"
- **High = confident.**

**entropy** — how *spread out* the whole probability distribution is.
- Computed as `-sum(p * log(p))`.
- Low entropy = probability concentrated on one class (decisive); high entropy = smeared
  across many classes (confused).
- **Low entropy = confident** (opposite direction to confidence).

**top1_top2_margin** — the gap between the best class and the second-best.
- Computed as `top1_prob - top2_prob`.
- Even if confidence is moderate, a big gap means the model isn't "torn" between two
  classes.
- **High margin = confident.**

**stage_id** — a number saying which stage this is: 0 for stage 25, 1 for stage 50, 2
for stage 75.
- Lets the gate behave differently at different stages (e.g. stricter early, looser late).
- Not a confidence signal — it's context.

So three features say "how trustworthy does this prediction look," and one says "which
stage am I at."

---

## 6. Learned-gate label

`gate_label: "exit=1 if the stage prediction is correct"`. Built in
`build_gate_training` as `(stage_pred == labels)`.

- **Label is 1 (exit) when the predicted class equals the true class; 0 (continue) when
  wrong.**
- **If correct -> exit:** the cheap stage already got the right answer, so stop and save
  compute.
- **If wrong -> continue:** this stage failed, so pay for a stronger stage and hope it
  does better.

**Weakness of this labeling:**
1. **Cost-unaware** — says nothing about *how much* compute continuing costs or whether a
   later stage would even fix the mistake. A sample wrong at stage 25 might *also* be
   wrong at dense; "continue" then just wastes compute.
2. **Noisy/imbalanced** — on easy data most early predictions are correct, so most labels
   are "exit," pushing the gate to exit too often. (Hence `class_weight='balanced'`.)
3. The label is about *this* stage's correctness, not the *globally optimal* exit point.

**Does the gate know the true label at test time?** **No.** True labels exist only during
training (on validation gate-fit data).

**Then how does it decide?** During training it learns "predictions with high confidence
/ low entropy / high margin tend to be the correct ones." At test time it plugs the
*features* into that learned relationship to **estimate the probability the prediction is
correct** — guessing correctness from the prediction's shape, not peeking at the answer.

---

## 7. Logistic-regression gate

`gate_type: "logistic_regression (StandardScaler + balanced)"`, built in
`fit_logistic_gate`.

- **Logistic regression** takes the four features and outputs a probability in [0, 1] —
  the estimated **P(prediction is correct)** = P(exit).
- **Input:** `[max_confidence, entropy, top1_top2_margin, stage_id]`.
- **Output:** one number in [0, 1], the gate probability.
- **StandardScaler:** rescales each feature to mean 0, std 1 before fitting. Needed
  because features live on different scales (confidence in [0,1], entropy in [0,~4.6],
  stage_id in {0,1,2}); scaling stops big-range features from dominating.
- **"balanced":** `class_weight='balanced'` weights the rare class more. Since "exit" is
  common, this stops the gate trivially learning "always exit."
- **Gate probability:** the model's confidence that exiting now is the right call.
- **Gate threshold (0.3 / 0.5 / 0.7):** the cutoff applied to that probability. "Exit if
  gate_probability >= gate_threshold." Low cutoff = exit easily; high = exit rarely.

---

## 8. Learned-gate inference, step by step (one test image)

From `eval_gate` / `gate_stage_probs`:

1. Image enters **stage 25** -> static_25 produces class probabilities.
2. Compute the 4 features (max_conf, entropy, margin, stage_id=0).
3. Logistic gate turns those features into `p_exit` (e.g. 0.82).
4. Compare: `p_exit >= gate_threshold?` If yes -> **exit**, take stage-25's class, done.
5. If no -> **continue** to **stage 50**, repeat with stage_id=1.
6. Still not exited -> **stage 75**, repeat with stage_id=2.
7. Still not exited -> **dense** runs and its answer is taken unconditionally (no gate).

**Toy example (made-up numbers, gate_threshold = 0.5):**
- Easy image at stage 25: `max_conf=0.88`, entropy low, margin high -> gate `p_exit=0.86
  >= 0.5` -> **exit at stage 25** (cost ~0.491 GFLOPs).
- Hard image at stage 25: `max_conf=0.42`, entropy high, margin low -> gate `p_exit=0.20
  < 0.5` -> **continue** to stage 50, and so on.

---

## 9. Learned-gate thresholds

`gate_thresholds: [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]`, swept in `sweep_gate_thresholds`.

**Why still need a threshold after training a gate?** The gate only outputs a
*probability*. You still choose *how high* it must be to actually exit. That choice
slides you along the accuracy/cost trade-off — like the confidence cutoff did before.

- **Low gate threshold (0.3):** exit is "easy" -> more images leave cheaply -> **lower
  FLOPs, but more risk of accepting wrong early answers -> lower accuracy.**
- **High gate threshold (0.9):** exit is "hard" -> most fall through to dense -> **higher
  FLOPs, and (up to a point) higher accuracy.**

Directly visible in the gate's `val_select_results` (on the 2000 gate-select images):

| gate_threshold | acc | avg GFLOPs | dense_rate |
|---|---|---|---|
| 0.3 | 0.7920 | 0.860 | 0.064 |
| 0.5 | 0.8130 | 1.116 | 0.140 |
| 0.7 | 0.8180 | 1.401 | 0.243 |
| 0.9 | 0.7975 | **3.142** | **1.000** |

At 0.9 the gate sends **everything** to dense (`dense_rate=1.0`) and costs **3.142
GFLOPs** — which leads into the cumulative-cost idea below.

---

## 10. Metrics interpretation (the `learned_gate_results` section, field by field)

Descriptive fields:
- **gate_type** — `logistic_regression (StandardScaler + balanced)`.
- **gate_features** — the 4 inputs (section 5).
- **gate_label** — `exit=1 if the stage prediction is correct` (section 6).
- **gate_train_split** — 3000 val images train the gate, 2000 val images pick the
  threshold; seed 42.
- **gate_thresholds** — the 7 cutoffs swept (section 9).
- **val_select_results** — the table above: for each gate threshold, accuracy / FLOPs /
  exit-counts on the 2000 gate-select images. Used to *choose* the operating point.
- **selected** — the chosen operating points, each then evaluated **once** on the 10 000
  test images:
  - **highest_val_acc** — gate threshold with best gate-select accuracy (0.7).
  - **best_under_static75_flops** — best gate-select accuracy among settings under
    static_75's 0.883 GFLOPs (gate threshold 0.3).
  - **test** — the honest final numbers for that setting on test.

Per-result fields, using the **test** record for `best_under_static75_flops`
(gate threshold 0.3) as the worked example:

```
accuracy = 0.7914            avg_flops_giga = 0.873705
exit_25_n = 6976   exit_50_n = 1841   exit_75_n = 536   dense_n = 647   (sum = 10000)
exit_25_rate = 0.6976   exit_50_rate = 0.1841   exit_75_rate = 0.0536   dense_rate = 0.0647
exit_25_acc = 0.8691   exit_50_acc = 0.6817   exit_75_acc = 0.6399   dense_exit_acc = 0.3910
avg_flops_exit_25 = 0.491472    avg_flops_exit_50 = 1.178923
avg_flops_exit_75 = 2.062351    avg_flops_dense_path = 3.141758
```

Meaning of each:
- **accuracy** — overall fraction of the 10 000 test images correct = 0.7914.
- **avg_flops_giga** — average compute per image across the cascade = 0.8737 GFLOPs.
- **exit_25_n / 50 / 75 / dense_n** — how many of the 10 000 images exited at each stage:
  6976 at 25, 1841 at 50, 536 at 75, 647 reached dense.
- **exit_\*_rate / dense_rate** — the same as fractions (0.6976 / 0.1841 / 0.0536 /
  0.0647) — these are the **exit rates**.
- **exit_25_acc / 50 / 75 / dense_exit_acc** — accuracy measured **only on the images that
  exited at that stage**. Images leaving at 25 were 86.9% correct; images reaching dense
  were only 39.1% correct (these are the genuinely hard ones — even the full model gets
  most wrong).
- **avg_flops_exit_25/50/75/dense_path** — the **cumulative** cost of each exit path.
  **Key subtlety:** a sample that exits at stage 50 cost **0.491 + 0.687 = 1.179** GFLOPs
  (`avg_flops_exit_50 = 1.178923`), *not* 0.687 — the cascade already ran stage 25 first.
  A sample reaching dense ran all four models: 0.491 + 0.687 + 0.883 + 1.079 = **3.142**
  (`avg_flops_dense_path`).

That cumulative rule is why high-exit-to-dense settings are so expensive.

---

## 11. Learned gate vs threshold cascade (actual numbers)

**Matched-budget point (near static_75 cost):**

| method | test accuracy | test avg GFLOPs |
|---|---|---|
| threshold cascade `best_under_static75_flops` (0.7, 0.8, 0.6) | **0.7973** | 0.886201 |
| learned gate `best_under_static75_flops` (gate_thr 0.3) | 0.7914 | 0.873705 |

- **Which is better?** The threshold cascade — **+0.0059 (~0.6%) more accurate**.
- **Is the gate cheaper?** Marginally — about **1.4% fewer FLOPs**.
- **Is the accuracy drop worth it?** No. You give up 0.6% accuracy to save ~0.01 GFLOPs.
  The threshold cascade has cheaper points on its own frontier that match the gate's cost
  while keeping higher accuracy.
- **Why "did not outperform"?** To win, the gate would need to be **above and to the
  left** on the accuracy-vs-FLOPs curve (more accurate *and* cheaper). Instead it's
  slightly cheaper *and* slightly less accurate — it sits *on or below* the existing curve.

**High-accuracy point:**

| method | test accuracy | test avg GFLOPs |
|---|---|---|
| threshold cascade `highest_val_acc` (0.95, 0.95, 0.8) | **0.8175** | 1.356347 |
| learned gate `highest_val_acc` (gate_thr 0.7) | 0.8128 | 1.412872 |

Here the gate is **worse on both axes** (lower accuracy *and* more expensive). Reason: at
gate_threshold 0.7 it sends ~24.5% of images all the way to dense (`dense_rate ~ 0.2453`),
and each costs the full 3.142 GFLOPs, inflating the average without enough extra accuracy.

---

## 12. Why the learned gate may have failed (plausible reasons)

- **Max-softmax confidence already captures most of the signal** — the threshold cascade
  keys off confidence, which is also the gate's strongest feature, so the gate has little
  extra to exploit.
- **Logistic regression is very simple** — only a straight (linear) boundary; any useful
  nonlinear pattern is out of reach.
- **The "exit if correct" labels are noisy and cost-blind** — they don't encode the cost
  of continuing or whether a later stage would help.
- **Small data for the gate** — only 3000 to fit, 2000 to select.
- **Features lack image/class content** — only 4 summary statistics; no information about
  *which* class or *what* the image contains.
- **Cumulative cost penalizes "continue" decisions harshly** — over-cautious gating
  (which "balanced" weighting can encourage) gets expensive fast.

(These are reasoned explanations consistent with the code/metrics, not measured ablations.)

---

## 13. Thesis-ready explanation

**Methodology.** In addition to the confidence-threshold cascade, we implemented a learned
exit gate. At each non-final stage a logistic-regression classifier predicts the
probability that the current stage's prediction is correct, from four features of that
stage's softmax output — maximum confidence, predictive entropy, top1-top2 margin, and a
stage identifier. The gate was trained on a 3000-image subset of the validation split
(labels: 1 if the stage prediction is correct, else 0), with feature standardization and
balanced class weighting; its exit-probability threshold was selected on the held-out 2000
validation images. As with the threshold cascade, the official test split was used only
once, to evaluate the selected operating points. FLOPs are accounted cumulatively along
the cascade path.

**Results.** At a budget near the static_75 model, the learned gate achieved 0.7914 test
accuracy at 0.8737 GFLOPs, versus 0.7973 at 0.8862 GFLOPs for the threshold cascade —
about 0.6% lower accuracy for a ~1.4% compute saving. At the high-accuracy operating point
the gate was dominated on both axes (0.8128 @ 1.4129 GFLOPs vs 0.8175 @ 1.3563 GFLOPs).
Across operating points the gate sat on or just below the threshold cascade's
accuracy-FLOPs frontier.

**Conclusion.** A learned logistic-regression exit gate was implemented and evaluated, but
it did not outperform the simpler confidence-threshold cascade. We therefore adopt the
validation-selected threshold cascade as the final method and report the learned gate as
an exploratory negative result. This is itself useful: it indicates that, for this
token-pruned ViT cascade on CIFAR-100, max-softmax confidence already captures most of the
exploitable exit signal, and a simple learned gate over summary statistics adds no benefit.

---

## 14. Visual diagram

```
                         gate sees: [max_conf, entropy, margin, stage_id]
                              |                |                |
  image                      v                v                v
   |
   v
 +--------+  p_exit>=thr? +--------+  p_exit>=thr? +--------+  p_exit>=thr? +--------+
 |stage 25|---- no ------>|stage 50|---- no ------>|stage 75|---- no ------>| dense  |
 | 0.491  |               | +0.687 |               | +0.883 |               | +1.079 |
 +--------+               +--------+               +--------+               +--------+
     | yes                    | yes                    | yes                    |
     v                        v                        v                        v
   EXIT                     EXIT                     EXIT                  forced EXIT
 cost 0.491              cost 1.179                cost 2.062              cost 3.142
 (cumulative GFLOPs along the path - each stage adds to the bill)
```

---

## 15. Final teaching summary (plain words)

- **What it is:** a tiny extra classifier ("gate") bolted onto the cascade that decides,
  at each cheap stage, whether to stop or keep going.
- **What it learns:** from validation data, it learns to guess *"is this stage's
  prediction probably correct?"* using four numbers describing how peaky/decisive the
  model's probabilities are.
- **How it decides:** it outputs a probability; if that probability clears a chosen
  cutoff, the image exits with the current answer, otherwise it moves to the next stage.
- **Why it was tested:** to see if a *learned* decision could beat the plain "is
  confidence above a number?" rule by combining several signals.
- **Why it isn't the final method:** on test it was either slightly worse-and-slightly-
  cheaper (`best_under_static75`: 0.7914 @ 0.8737 vs 0.7973 @ 0.8862) or worse on both
  axes (`highest_val_acc`: 0.8128 @ 1.4129 vs 0.8175 @ 1.3563). It never beat the
  threshold cascade, so the threshold cascade stays as the chosen method and the gate is
  reported honestly as an exploratory negative result.

**Where to look if anything is unclear:** gate logic in
[`src/training/cascade_eval.py`](../src/training/cascade_eval.py)
(`build_gate_training`, `fit_logistic_gate`, `gate_stage_probs`, `eval_gate`,
`sweep_gate_thresholds`, `stage_entropy_margin`), orchestration/data-split in
[`src/train.py`](../src/train.py) (`_run_learned_gate`), and the exact numbers under
`learned_gate_results` in
[`checkpoints/cascade_clean_split/metrics.json`](../checkpoints/cascade_clean_split/metrics.json).
