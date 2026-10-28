# Chameleon: paper ↔ code guide

Erfanian, Jagadish, Asudeh. *Chameleon: Foundation Models for Fairness-aware Multi-modal Data Augmentation to
Enhance Coverage of Minorities.* The paper exists in two versions whose full texts differ, so this guide cites them separately:

* **[arXiv]** arXiv:2402.01071v1, 2 February 2024. <https://arxiv.org/abs/2402.01071v1>.
  The PVLDB paper cites this version as its technical report ([35] there).
* **[PVLDB]** *Proceedings of the VLDB Endowment* 17(11): 3470–3483, 2024.
  [doi:10.14778/3681954.3682014](https://doi.org/10.14778/3681954.3682014).

Every paper reference below names its version, for example *arXiv §6.4.1, Table 4* or *PVLDB §6.3.1, Table 5*.
"Both" means the section has the same number and the same content in both versions. A bare § number with a
link, such as [§4 below](#4-where-the-code-differs-from-the-paper-read-before-comparing-numbers), points to a part of this guide.

### What differs between the versions

Sections 2 to 5 (preliminaries, rejection tests, combination selection, guide selection) keep their numbers.
Section 6 (experiments) is reorganised, and tables and figures are renumbered.

| Topic | arXiv | PVLDB |
|---|---|---|
| §3 title; §3.1 test | "Rejection sampling"; "Data distribution test" (acceptance rate *DDTAR*) | "Rejection test"; "Outlier (novelty) detection test" (*OTAR*). Footnote 3 says it is not rejection sampling and does not preserve the distribution, see PVLDB §8.2 |
| §4 hardness and approximation | Lemma 1 (NP-hard), Theorem 1 (log η ratio), Algorithm 1 (Greedy) printed, proofs in §8 | Theorem 1 (NP-hard), Theorem 2 (log η ratio). Pseudo-code and proofs deferred to the technical report, i.e. arXiv |
| §5.1 random guide | No distribution result | Theorem 3: bound on the KL divergence for the random-guide strategy |
| §5.3 LinUCB pseudo-code | Algorithm 2 | Deferred to the technical report |
| Implementation and hardware (microservices, DALL·E 2 image sizes, crop or encase) | §6.1, §6.2.1 | Deferred to the technical report |
| Total generation spend | 1,962 DALL·E 2 images, $33.6 (§6.2.3) | 3,657 DALL·E 2 images plus 40,656 GPT-4 prompts, $91.9 (§6.1.2) |
| Baselines for the downstream tasks | None | Reweighing and SMOTE (§6.1.3) |
| FERET proof of concept | §6.3. Table 2: 756 images. Table 3, Figure 4. Overall P/R/F1 0.81/0.75/0.78; F1 for Black/Hispanic/Middle Eastern 0.16/0.25/0.00; Black F1-disparity 79% → 27% | §6.2.1. Table 1: 661 images. Table 3 (adds Reweighing and SMOTE), Figure 3. Overall P/R/F1 0.68/0.66/0.67; F1 0.20/0.03/0.00; Black F1-disparity 70% → 21% |
| Same FERET repair cost | 307 queries, 231 accepted (75%), $4.91 ($0.016/image) | Same |
| Other proofs of concept | None | §6.2.2 Animal-10 (Table 4, Figures 4–5, τ = 1200); §6.2.3 UTKFace age-group predictor, Female/Adolescent repair (Figure 6). Table 2: a 16,052-image UTKFace group distribution |
| Task 1: guide strategy × mask level | §6.4.1, Table 4 (sample form: Figure 5) | §6.3.1, Table 5 (same numbers; no sample-form figure) |
| Task 2: combination selection cost | §6.4.2, Figure 6 | §6.3.2, Figure 7 |
| Task 3: NIQE / BRISQUE / NIMA vs humans | §6.4.3, Table 5 (Jaccard 0.127 / 0.068 / 0.068) | Deferred to the technical report; one sentence in the §6 introduction |
| Text augmentation (Emotions, GPT-4, representation rate) | None | §6.4 Extensions, Figure 8 |
| Discussion and limitations | None. §8 is the proofs | §8.1 reliance on existing models, §8.2 distribution preservation |

## 1. What the system does

Given an image dataset with categorical attributes (UTKFace: `age_group`, `gender`, `race`) and a coverage
threshold τ, Chameleon generates the fewest synthetic images needed so that every demographic subgroup
(up to some level) has at least τ examples:

```
            ┌──────────── repeat until no MUPs at the target level ─────────────┐
            ▼                                                                   │
 (1) find MUPs ──► (2) pick combination ──► (3) pick guide image ──► (4) mask ──► (5) inpaint ──► (6) accept/reject
  §2.3              §4                       §5.1-5.3                 §5.4         §2.2           §3 + bandit reward
```

Section numbers in the diagram and in this table are the same in both versions. Algorithms and tables are not.

| Step | Paper | Code |
|---|---|---|
| (1) Maximal Uncovered Patterns (MUPs): the most general subgroups with count < τ. Pattern `1x3` = age_group 1, any gender, race 3 | Both §2.3, ref [5] in both (Asudeh et al., ICDE'19) | [ImageAnalyzer/csv_crud.py:66](../ImageAnalyzer/csv_crud.py:66) runs the prebuilt Java container `merfanian/fairness-lens:0.2.1` (algorithm `greedy` from the ICDE'19 code) via `docker run` on the host's docker socket |
| (2) Combination selection: among the lowest-level MUPs, find the full combination that hits the most of them (tree + bit-vector inverted index) and generate `γ = min gap` images of it | Both §4. Pseudo-code: arXiv Algorithm 1 (not printed in PVLDB). Approximation ratio: arXiv Theorem 1 = PVLDB Theorem 2 | [ImageAnalyzer/main.py:129](../ImageAnalyzer/main.py:129) (`get_best_mup`) and [ImageAnalyzer/greedy_mup_selector.py:66](../ImageAnalyzer/greedy_mup_selector.py:66). Picking the hit MUP with the **largest count** gives the smallest gap γ. The loop in Alg. 1 is spread across UI rounds: the UI calls `/mups/` again once the chosen MUP is satisfied |
| (3a) Random guide | Both §5.1 (PVLDB adds Theorem 3) | `strategy="random"` → [Gateway/services.py](../Gateway/services.py) `get_random_image` |
| (3b) Similar-tuple guide: siblings differing in one attribute (±1 for ordinal attributes), weighted by count, excluding the target combination itself | Both §5.2 | `strategy="similar"` → [Gateway/services.py:23](../Gateway/services.py:23), [Gateway/models.py:80](../Gateway/models.py:80) |
| (3c) Bandit guide: arm = which attribute to change | Both §5.3. Pseudo-code: arXiv Algorithm 2 (not printed in PVLDB) | `strategy="ucb"` → `get_mup_ucb` in [Gateway/services.py](../Gateway/services.py) plus the `UCB/` service: disjoint LinUCB, context = the combination being repaired (one-hot), reward = 1 if the image passes both tests. Was non-contextual UCB1 in the published code, see [§5 below](#5-fixes-made-to-get-it-running) |
| (3d) No guide (baseline) | arXiv §6.2.4 and Table 4; PVLDB §6.1.3 and Table 5 | `strategy="none"`: plain text-to-image |
| (4) Mask: rembg foreground, inverted so the **person** is regenerated and the background kept. *Accurate* = exact silhouette, *moderate* = silhouette + circles of radius 10% of width, *imprecise* = bounding box | Both §5.4, Figure 2 | [MaskGenerator/main.py:19](../MaskGenerator/main.py:19). The inversion is a monkey-patch in `invert_remover.py` (importing it swaps rembg's trimap) |
| (5) Foundation model: prompt = `prompt_prefix + combination words + prompt_suffix`; word order is set by `position` in `config.json` | Both §2.2 | [ImageEditor/main.py](../ImageEditor/main.py) |
| (6a) Quality test: humans flag "unrealistic" images | Both §3.2 | In the UI, you click the unrealistic images, then Submit → [Gateway/main.py:138](../Gateway/main.py:138) appends the accepted ones to the dataset CSV (`is_generated=True`) and sends reward 1/0 to the bandit. The multi-rater t-test was done offline with `Verifier/` and the notebooks in `NIQE/notebooks/`. Its baseline p = 0.86 is the estimated chance that a rater calls a *real* UTKFace image realistic (arXiv §6.4.1, PVLDB §6.3.1) |
| (6b) Distribution test: one-class SVM on image embeddings | Both §3.1 ("data distribution test" in arXiv, "outlier detection test" in PVLDB) | `DataDistributionTester/` (MediaPipe embedder + sklearn `OneClassSVM`, ν = 0.3, RBF by default), called for every generated image from [Gateway/main.py](../Gateway/main.py). Rejected images are greyed out in the UI, never added, and give reward 0. Was disabled in the published code, see [§5 below](#5-fixes-made-to-get-it-running) |

### Experiments and where they live

| Experiment | arXiv | PVLDB | Code / artifacts |
|---|---|---|---|
| Proof of concept: FERET race CNN before/after repair, τ=100. The two versions report different FERET sizes and CNN scores, see the table above | §6.3, Tables 2–3, Figure 4 | §6.2.1, Tables 1 and 3, Figure 3 | `PreProcessor/feret_tools/` (`cnn.ipynb`, `cnn2.ipynb`). FERET itself requires a NIST license request |
| Proof of concept: Animal-10, UTKFace age-group predictor, Reweighing and SMOTE baselines | — | §6.2.2–6.2.3, Table 4, Figures 4–6 | Not in this repo |
| Task 1: guide strategy × mask level, human evaluation | §6.4.1, Table 4 | §6.3.1, Table 5 | UI + `Verifier/` (human-rating app) + `NIQE/notebooks/UTKFace/*.ipynb`, `PreProcessor/mask_level_analysis.ipynb` |
| Task 2: Greedy vs Random vs Min-Gap, number of queries | §6.4.2, Figure 6 | §6.3.2, Figure 7 | `CombinationSelectionAnalyzer/` simulates generation by adding fake rows (no API cost). It also has a Best-Comb strategy, which neither version reports. Committed results are in `l1/`, `l2/`, `mingap_*.csv` |
| Task 3: NIQE / BRISQUE / NIMA vs humans (Jaccard) | §6.4.3, Table 5 | Technical report only | `NIQE/` (`inference_iqa.py`, notebooks) |
| Text augmentation: Emotions, representation rate | — | §6.4, Figure 8 (GPT-4) | `TextAugmentor/` (emotion dataset, GPT-4o) |

## 2. Running it

Prerequisites: Docker Desktop. On Apple Silicon the MUP container runs under amd64 emulation, which works fine.

```bash
cp .env.example .env                                 # set CHAMELEON_DATA_DIR to an ABSOLUTE path, e.g. $PWD/data
cp Gateway/.env-example Gateway/.env
cp ImageAnalyzer/.env-example ImageAnalyzer/.env
cp ImageEditor/.env-example ImageEditor/.env         # add OPENAI_API_KEY
docker pull merfanian/fairness-lens:0.2.1            # MUP finder, started on demand by ImageAnalyzer
```

Get UTKFace (https://susanqq.github.io/UTKFace/, non-commercial research license). The authors' metadata
table (18,978 rows, bundled at `/app/data/images.csv` inside the MUP image) excludes race 4. Neither version gives this
count; PVLDB Table 2 lists a 16,052-image UTKFace distribution by age group and gender. The authors' notebook
dropped images smaller than 256 px, so they likely used the larger *in-the-wild* originals rather than the
200 px aligned crops. Either variant works with the script. Then:

```bash
python3 -m pip install pillow
python3 scripts/prepare_utkface.py --src /path/to/UTKFace --data-dir ./data
docker compose up --build            # UI: http://localhost:3000, API docs: http://localhost:8000/docs
```

The data layout under `CHAMELEON_DATA_DIR`:

```
resources/utkface/<age>_<gender>_<race>_<date>.png   original images
datasets/utkface.csv                                   filename,age_group,gender,race,is_generated
datasets/utkface_<name>_<id>.csv                       sub-datasets made from the UI ("Create dataset")
results/<dataset_id>/{*.png,masks/,base_images/,mups/} generated images + the guide/mask/MUP behind each
```

### UI walkthrough (one iteration of Alg. 1)

1. *Create dataset*: sample N rows of `utkface` into a working copy. The paper's Task 1 used a hand-built
   "challenging subset" with 16 level-3 MUPs at τ=10 (arXiv §6.4.1, PVLDB §6.3.1). Repairs mutate the CSV, so always work on a copy.
2. *Repair dataset* → enter τ → **Find MUPs**. This shows every MUP plus the chosen combination (`pattern`), its prompt, and the count still needed.
3. Choose mask accuracy, guide strategy (LinUCB / Similar / Random / No base image) and batch size → **Repair**.
4. Click the images that look unrealistic → **Submit**. The rest are added to the dataset.
5. **Re-evaluate**: once the MUP reaches τ, the next combination is chosen.

### Same thing with curl

```bash
curl "localhost:8000/v1/datasets/utkface/mups/?threshold=100"            # step 1+2
curl -X POST localhost:8000/v1/datasets/utkface/mups/generate/ -H 'Content-Type: application/json' -d '{
  "pattern":"113", "threshold":100, "frequency":78, "limit":1,
  "strategy":"ucb", "accuracy":"moderate", "prompt":"indian female preschooler",
  "attributes": <the "attributes" array from the /mups/ response> }'
```

### Reproducing Task 2 (combination selection, no API cost; arXiv §6.4.2, PVLDB §6.3.2)

```bash
cp data/datasets/utkface.csv data/datasets/utkface_sim.csv     # the simulator appends fake rows
docker run --rm -p 8010:80 -e IMAGE_ANALYZER_BASE_URL=http://host.docker.internal:8001 \
  -v $PWD/CombinationSelectionAnalyzer:/app/src:ro -v $PWD/data/sim:/out -w /out \
  chameleon-gateway uvicorn main:app --app-dir /app/src --host 0.0.0.0 --port 80
curl -X POST "localhost:8010/v1/simulate/greedy/?dataset_id=utkface_sim&threshold=200&resolve_until_level=2"
# strategies: greedy | random | bestcomb | mingap. Reset utkface_sim.csv between runs.
```

A smoke run on the authors' UTKFace metadata gave: greedy, τ=200, level 2 → **2,148** synthetic tuples
(19 L2 MUPs at start). The paper plots this setting in arXiv Figure 6a and PVLDB Figure 7a without
giving exact totals. The authors' committed run reports 1,558 tuples starting from 15 L2 MUPs, with
Random at 5,384. Their MUP counts don't match either age-group cardinality, so they probably used a
slightly different metadata file. Expect the same ordering between methods, not identical numbers.

### Full repair

1. `docker compose up --build -d`. Rebuilding is needed for the new `ucb` and `data_distribution_tester` images.
2. In the UI, *Create dataset* to get a working copy, then *Repair dataset* with τ (100 in the FERET proof of concept, arXiv §6.3 and PVLDB §6.2.1),
   **LinUCB**, mask **moderate**, and a batch size of 5 to 10.
3. Repeat Repair, review, Submit and Re-evaluate until no MUPs are left. The distribution test rejects roughly
   a third of generated images, so plan for about 1.5 generations per image that gets added.
   The first Repair also embeds the training set (a few minutes, then cached).

Bandit state and DDT embeddings live in `${CHAMELEON_DATA_DIR}/ucb/` and `ddt_cache/`. Delete
`ucb/linucb.pkl` to start the bandit fresh for a new experiment.

### Cost tracking

The paper measures repair cost in foundation-model queries (Task 2 metric, arXiv §6.2.5 and PVLDB §6.1.4). Every query is paid for, including ones whose image
the distribution test or a reviewer later rejects. The Gateway appends one row per query to
`results/<dataset_id>/cost_ledger.jsonl`, failed calls included, and one row per reviewed image on Submit.
Each query row carries the editor's `usage`. For OpenAI that is the model, quality, token counts and `cost_usd`,
priced with the `OPENAI_PRICE_*` rates in `ImageEditor/.env`.

```bash
curl localhost:8000/v1/datasets/<dataset_id>/cost/                                # live, from the Gateway
python3 scripts/cost_report.py --data-dir ./data --dataset <dataset_id> [--json]  # offline
```

Both report queries, failures, DDT rejections, accepted images, **queries per accepted image**, and summed usage
(total `cost_usd` and per accepted image), overall and per pattern. Set `MAX_SPEND_USD` in `Gateway/.env` to stop
generating for a dataset once its ledger reaches that many dollars. A repair request that hits the cap returns
`stopped_reason`.

## 3. Using a new dataset

1. **Table** `datasets/<name>.csv`: column 0 is `filename`, then one integer-coded column per attribute,
   then `is_generated` (`False` for real rows). `<name>` must not contain `_` (the parent dataset is
   the id prefix before the first `_`).
2. **Values must be single digits 0–9.** The Java MUP code reads `value.charAt(0)` and maps it to a
   bit-vector slot `value + Σ previous cardinalities`, so a value ≥ cardinality silently corrupts the counts
   (this happened with UTKFace's 9th age group, see [§5 below](#5-fixes-made-to-get-it-running)).
3. **Images** in `resources/<name>/<filename>`, square PNG (the edit APIs need square images, and arXiv §6.1
   center-crops images with aspect ≤ 1.5 or pads elongated ones with white; PVLDB leaves this to the technical report).
4. **Config** in `ImageAnalyzer/config.json`. Add an entry to `datasets`:
   * `attributes` list order = order of characters in a pattern. Set `pattern_index` equal to that list index.
   * `column_number` = CSV column index of the attribute.
   * `cardinality`, `mapping` (code → words used in the prompt), `position` (word order in the prompt),
     `ordered` (ordinal attributes only use ±1 siblings as similar guides).
   * `prompt_prefix` / `prompt_suffix`: the prompt is `prefix + attribute words (by position) + suffix`. The
     authors left both empty (prompt `"indian female preschooler"`). This repo sets the UTKFace prefix to
     `"a realistic photo of a"` and uses noun age labels, giving e.g. `"a realistic photo of a black female elderly person"`.
5. UTKFace-specific hard-coding to be aware of: `ImageAnalyzer/main.py` `get_all_combinations_status`
   (age_group/gender/race), `add_random_images_to_dataset`, `export_partial_dataset`, and
   `CombinationSelectionAnalyzer/main.py` (the random/min-gap baselines).
6. Also set `UCB/config.json` `num_arms` to your number of attributes.

## 4. Where the code differs from the paper (read before comparing numbers)

* **Foundation model.** The paper used DALL·E 2 `/images/edits` at 256/512 px (arXiv §6.1; $0.016/image in arXiv §6.3 and PVLDB §6.2.1). OpenAI shut
  DALL·E 2 down on 2026-05-12, and `gpt-image-1` on 2026-10-23. `ImageEditor` now uses `OPENAI_IMAGE_MODEL`
  (default `gpt-image-2.5-sunburst`, 1024×1024, `quality=low`, `background=opaque`, `input_fidelity=high`; optional
  parameters a model rejects are dropped and retried). Each call returns and logs its token usage and cost, which go into the cost ledger ([§2](#cost-tracking)). The tests
  below were run with `gpt-image-1`. This changes the paper's results:
  * GPT Image models re-render the whole image and treat the mask only as a hint. In testing, the guide
    photo's background and framing were not kept, so the accurate/moderate/imprecise comparison is not meaningful.
  * OpenAI's moderation blocked an edit of an infant guide image (`moderation_blocked`, output stage). Many
    UTKFace MUPs involve children, and the Gateway retries blocked requests like other errors.
  * Outputs are 1024 px; the Gateway resizes them back to the guide image's size, as the paper did.
  * Cost per image is different. Check current pricing before large runs.
* **Bandit exploration α.** Neither version gives α (§5.3 in both). `UCB/config.json` uses α = 1, the usual choice for
  rewards in [0, 1]. The published config had `alpha: 100`, which the old code never read; with it, LinUCB
  would explore almost uniformly for thousands of pulls.
* **Distribution test kernel.** The paper reports ν = 0.3 with both linear and RBF kernels (arXiv Table 4, PVLDB Table 5). Pick one with
  `DDT_KERNEL` in `Gateway/.env` (default `rbf`), or set `DDT_ENABLED=false` to turn the test off. With ν = 0.3 the
  SVM treats about 30% of its own training images as outliers, so expect a similar share of generated images to be rejected.
* **Quality test** in the UI is one rater with a binary accept. The paper's t-test across raters (both §3.2; α = 0.1/0.4 in arXiv §6.4.1 and PVLDB §6.3.1) was offline.
* The `/examine-dataset` UI page calls `/v1/images/sample/`, which lives in `Verifier/`, not the Gateway.
  That service isn't in `docker-compose.yml` and needs its own CSVs.

## 5. Fixes made to get it running

* `docker-compose.yml`:
  * Hard-coded `/home/mahdi/...` paths replaced by `CHAMELEON_DATA_DIR`.
  * Added a `/results` volume (generated images used to be lost with the container).
  * Added the ImageEditor `.env` and a u2net cache volume.
* `Gateway/.env-example`: fixed the `data_ditribution_tester` hostname typo.
* ImageAnalyzer:
  * Now uses the current docker CLI. Debian's `docker.io` client is rejected by Docker Engine ≥ 29.
  * Starts the MUP image with `--platform linux/amd64`.
* `ImageAnalyzer/config.json`: age_group cardinality **8 → 9**. UTKFace has 9 groups (`PreProcessor/utkface_tools/config.json`), and group 8 was being counted as gender 0.
* Gateway:
  * Accepting an image posted to `/v1/datasets/{filename}/` without the dataset id, so accepted images were never added.
  * Output directories were created with non-recursive `mkdir`.
  * The generation loop retried forever on any error. It now gives up after `MAX_CONSECUTIVE_FAILURES` (default 5).
  * Failed attempts left extra entries in `pulled_arms`, so bandit rewards went to the wrong arms.
* **Aligned with the paper:**
  * Bandit: the published `UCB/` was a non-contextual UCB1. It ignored the combination (the Gateway always sent `0`),
    `num_arms: 2` meant the race arm was never pulled, its mean update was wrong, and its state was saved only on clean
    shutdown. It is now disjoint LinUCB (paper §5.3, both versions) with one bandit per dataset. There is one arm per attribute, and
    arms with no non-empty sibling combination are skipped. State is saved to `${CHAMELEON_DATA_DIR}/ucb/` after every update.
  * Distribution test: it was commented out (`ddt_result = True`), and `mediapipe==0.10.0` has been removed from
    PyPI, so the service couldn't be built. It is now on by default with `mediapipe==0.10.18`, which runs natively on
    arm64. Training embeddings are cached in `${CHAMELEON_DATA_DIR}/ddt_cache/` and the SVM is fitted once
    per training set (a few minutes for all of UTKFace on first use; later runs load the cache).
  * UI: images rejected by the distribution test were counted as added when the remaining count was updated.
* ImageEditor: moved to GPT Image models (base64 responses). It also now starts without an API key.
