# Chameleon: paper ↔ code guide

Paper: Erfanian, Jagadish, Asudeh. *Chameleon: Foundation Models for Fairness-aware Multi-modal Data
Augmentation to Enhance Coverage of Minorities.* PVLDB 17 (2024), [arXiv:2402.01071](https://arxiv.org/abs/2402.01071).

## 1. What the system does

Given an image dataset with categorical attributes (UTKFace: `age_group`, `gender`, `race`) and a coverage
threshold τ, Chameleon generates the fewest synthetic images needed so that every demographic subgroup
(up to some level) has at least τ examples:

```
            ┌──────────── repeat until no MUPs at the target level ─────────────┐
            ▼                                                                   │
 (1) find MUPs ──► (2) pick combination ──► (3) pick guide image ──► (4) mask ──► (5) inpaint ──► (6) accept/reject
  §2.3              §4, Alg. 1               §5.1-5.3                 §5.4         §2.2           §3 + bandit reward
```

| Step | Paper | Code |
|---|---|---|
| (1) Maximal Uncovered Patterns (MUPs): the most general subgroups with count < τ. Pattern `1x3` = age_group 1, any gender, race 3 | §2.3, ref [5] (Asudeh et al., ICDE'19) | [ImageAnalyzer/csv_crud.py:66](../ImageAnalyzer/csv_crud.py:66) runs the prebuilt Java container `merfanian/fairness-lens:0.2.1` (algorithm `greedy` from the ICDE'19 code) via `docker run` on the host's docker socket |
| (2) Combination selection: among the lowest-level MUPs, find the full combination that hits the most of them (tree + bit-vector inverted index) and generate `γ = min gap` images of it | §4, Alg. 1, Thm. 1 | [ImageAnalyzer/main.py:129](../ImageAnalyzer/main.py:129) (`get_best_mup`) and [ImageAnalyzer/greedy_mup_selector.py:66](../ImageAnalyzer/greedy_mup_selector.py:66). Picking the hit MUP with the **largest count** gives the smallest gap γ. The loop in Alg. 1 is spread across UI rounds: the UI calls `/mups/` again once the chosen MUP is satisfied |
| (3a) Random guide | §5.1 | `strategy="random"` → [Gateway/services.py](../Gateway/services.py) `get_random_image` |
| (3b) Similar-tuple guide: siblings differing in one attribute (±1 for ordinal attributes), weighted by count, excluding the target combination itself | §5.2 | `strategy="similar"` → [Gateway/services.py:23](../Gateway/services.py:23), [Gateway/models.py:80](../Gateway/models.py:80) |
| (3c) Bandit guide: arm = which attribute to change | §5.3, Alg. 2 | `strategy="ucb"` → `get_mup_ucb` in [Gateway/services.py](../Gateway/services.py) plus the `UCB/` service: disjoint LinUCB, context = the combination being repaired (one-hot), reward = 1 if the image passes both tests. Was non-contextual UCB1 in the published code, see §5 |
| (3d) No guide (baseline) | Table 4 | `strategy="none"`: plain text-to-image |
| (4) Mask: rembg foreground, inverted so the **person** is regenerated and the background kept. *Accurate* = exact silhouette, *moderate* = silhouette + circles of radius 10% of width, *imprecise* = bounding box | §5.4, Fig. 2 | [MaskGenerator/main.py:19](../MaskGenerator/main.py:19). The inversion is a monkey-patch in `invert_remover.py` (importing it swaps rembg's trimap) |
| (5) Foundation model: prompt = `prompt_prefix + combination words + prompt_suffix`; word order is set by `position` in `config.json` | §2.2 | [ImageEditor/main.py](../ImageEditor/main.py) |
| (6a) Quality test: humans flag "unrealistic" images | §3.2 | In the UI, you click the unrealistic images, then Submit → [Gateway/main.py:138](../Gateway/main.py:138) appends the accepted ones to the dataset CSV (`is_generated=True`) and sends reward 1/0 to the bandit. The multi-rater t-test (p = 0.86) was done offline with `Verifier/` and the notebooks in `NIQE/notebooks/` |
| (6b) Distribution test: one-class SVM on image embeddings | §3.1 | `DataDistributionTester/` (MediaPipe embedder + sklearn `OneClassSVM`, ν = 0.3, RBF by default), called for every generated image from [Gateway/main.py](../Gateway/main.py). Rejected images are greyed out in the UI, never added, and give reward 0. Was disabled in the published code, see §5 |

### Experiments and where they live

| Paper | Code / artifacts |
|---|---|
| §6.3 proof of concept (FERET, τ=100, CNN before/after) | `PreProcessor/feret_tools/` (`cnn.ipynb`, `cnn2.ipynb`). FERET itself requires a NIST license request |
| §6.4.1 Task 1: guide strategy × mask level, human evaluation (Table 4) | UI + `Verifier/` (human-rating app) + `NIQE/notebooks/UTKFace/*.ipynb`, `PreProcessor/mask_level_analysis.ipynb` |
| Task 2: Greedy vs Random vs Min-Gap vs Best-Comb, number of queries | `CombinationSelectionAnalyzer/` simulates generation by adding fake rows (no API cost). Committed results are in `l1/`, `l2/`, `mingap_*.csv` |
| Task 3: NIQE / BRISQUE / NIMA vs humans (Jaccard) | `NIQE/` (`inference_iqa.py`, notebooks) |
| (not in paper) text augmentation | `TextAugmentor/` (emotion dataset, GPT-4o) |

## 2. Running it

Prerequisites: Docker Desktop. On Apple Silicon the MUP container runs under amd64 emulation, which works fine.

```bash
cp .env.example .env                                 # set CHAMELEON_DATA_DIR to an ABSOLUTE path, e.g. $PWD/data
cp Gateway/.env-example Gateway/.env
cp ImageAnalyzer/.env-example ImageAnalyzer/.env
cp ImageEditor/.env-example ImageEditor/.env         # add OPENAI_API_KEY
docker pull merfanian/fairness-lens:0.2.1            # MUP finder, started on demand by ImageAnalyzer
```

Get UTKFace (https://susanqq.github.io/UTKFace/, non-commercial research license). The paper's table
(18,978 rows, bundled at `/app/data/images.csv` inside the MUP image) excludes race 4. The authors' notebook
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
   "challenging subset" with 16 level-3 MUPs at τ=10. Repairs mutate the CSV, so always work on a copy.
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

### Reproducing Task 2 (combination selection, no API cost)

```bash
cp data/datasets/utkface.csv data/datasets/utkface_sim.csv     # the simulator appends fake rows
docker run --rm -p 8010:80 -e IMAGE_ANALYZER_BASE_URL=http://host.docker.internal:8001 \
  -v $PWD/CombinationSelectionAnalyzer:/app/src:ro -v $PWD/data/sim:/out -w /out \
  chameleon-gateway uvicorn main:app --app-dir /app/src --host 0.0.0.0 --port 80
curl -X POST "localhost:8010/v1/simulate/greedy/?dataset_id=utkface_sim&threshold=200&resolve_until_level=2"
# strategies: greedy | random | bestcomb | mingap. Reset utkface_sim.csv between runs.
```

A smoke run on the paper's UTKFace metadata gave: greedy, τ=200, level 2 → **2,148** synthetic tuples
(19 L2 MUPs at start). The authors' committed run reports 1,558 tuples starting from 15 L2 MUPs, with
Random at 5,384. Their MUP counts don't match either age-group cardinality, so they probably used a
slightly different metadata file. Expect the same ordering between methods, not identical numbers.

### Local inpainting backend (recommended instead of OpenAI)

`InpaintServer/` runs SDXL Inpainting (`diffusers/stable-diffusion-xl-1.0-inpainting-0.1`) natively on the
host. Docker on macOS can't reach the Apple GPU, so it runs outside the containers. It serves the same
`/v1/images/edits` API as `ImageEditor`, so switching is one line:

```bash
InpaintServer/run.sh        # first run: Python 3.12 venv via uv (~750MB) + ~6.6GB of weights into InpaintServer/models/
# in Gateway/.env:
IMAGE_EDITOR_BASE_URL=http://host.docker.internal:8006
docker compose up -d gateway
```

Unlike GPT Image models it is true inpainting: only the masked region is regenerated, and the original
pixels outside the mask are pasted back (feathered by `INPAINT_FEATHER` px). Measured on an M5 / 24 GB:
the model loads in ~7 s and takes **~50 s per 512 px image** (30 steps at 1024 px). It costs $0 and has no
moderation step. On test-2's guide and mask the average change outside the mask was 0.0/255, versus 55.5/255 for
`gpt-image-1`. Tunables: `INPAINT_STEPS`, `INPAINT_GUIDANCE`, `INPAINT_STRENGTH`,
`INPAINT_NEGATIVE_PROMPT`, and `INPAINT_MODEL` for another diffusers inpainting checkpoint.
`download_models.sh` uses resumable curl, because `huggingface_hub`'s Xet downloader stalled in testing.

### Full repair with the local backend

1. Start SDXL on the host: `InpaintServer/run.sh` and leave it running.
2. In `Gateway/.env`, set `IMAGE_EDITOR_BASE_URL=http://host.docker.internal:8006`.
3. `docker compose up --build -d`. Rebuilding is needed for the new `ucb` and `data_distribution_tester` images.
4. In the UI, *Create dataset* to get a working copy, then *Repair dataset* with τ (100 in the paper),
   **LinUCB**, mask **moderate**, and a batch size of 5 to 10.
5. Repeat Repair, review, Submit and Re-evaluate until no MUPs are left. Each image takes about 50 s, and the
   distribution test rejects roughly a third, so plan for about 75 s of SDXL time per image that gets added.
   The first Repair also embeds the training set (a few minutes, then cached).

Bandit state and DDT embeddings live in `${CHAMELEON_DATA_DIR}/ucb/` and `ddt_cache/`. Delete
`ucb/linucb.pkl` to start the bandit fresh for a new experiment.

### Cost tracking

The paper measures repair cost in foundation-model queries. Every query is paid for, including ones whose image
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

**Compute with the local backend.** `InpaintServer` returns compute measurements as its `usage`, so the same
ledger and report cover SDXL runs. Each query records:
* `inference_seconds`: time in the SDXL pipeline.
* `cpu_seconds`: process CPU time.
* `peak_device_memory_gb`: Metal driver memory for MPS, or allocated memory for CUDA.
* `peak_rss_gb`.
* `settings`: resolution and steps.

The Gateway's per-query `seconds` adds mask, transfer and distribution-test time on top. `peak_*` fields are reported
as maxima, not sums. For the run as a whole, the server samples every `RESOURCE_SAMPLE_SECONDS` (default 5) into
`InpaintServer/logs/resources_<start>.csv` (busy flag, process CPU %, RSS, device memory, system CPU and memory),
and `curl localhost:8006/resources` reports totals since start, including utilization (the share of time spent generating).

Energy isn't measured, because Apple GPU power needs `sudo powermetrics`. To put SDXL on the same dollar scale as
OpenAI, set `INPAINT_POWER_WATTS` (the machine's draw while generating) and `ELECTRICITY_USD_PER_KWH`
before `run.sh`. Usage then also carries `energy_kwh_est` and `cost_usd`, both labelled as estimates. To measure
the draw once, run `sudo powermetrics --samplers cpu_power,gpu_power -i 5000` during a Repair and read *Combined Power*.

## 3. Using a new dataset

1. **Table** `datasets/<name>.csv`: column 0 is `filename`, then one integer-coded column per attribute,
   then `is_generated` (`False` for real rows). `<name>` must not contain `_` (the parent dataset is
   the id prefix before the first `_`).
2. **Values must be single digits 0–9.** The Java MUP code reads `value.charAt(0)` and maps it to a
   bit-vector slot `value + Σ previous cardinalities`, so a value ≥ cardinality silently corrupts the counts
   (this happened with UTKFace's 9th age group, see §4).
3. **Images** in `resources/<name>/<filename>`, square PNG (the edit APIs need square images, and the paper
   center-crops images with aspect ≤ 1.5 or pads elongated ones with white).
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

* **Foundation model.** (Use the local SDXL backend above to avoid the GPT Image issues below.) The paper used DALL·E 2 `/images/edits` at 256/512 px ($0.016/image). OpenAI shut
  DALL·E 2 down on 2026-05-12, and `gpt-image-1` on 2026-10-23. `ImageEditor` now uses `OPENAI_IMAGE_MODEL`
  (default `gpt-image-2.5-sunburst`, 1024×1024, `quality=low`, `background=opaque`, `input_fidelity=high`; optional
  parameters a model rejects are dropped and retried). Each call returns and logs its token usage and cost, which go into the cost ledger (§2). The tests
  below were run with `gpt-image-1`. This changes the paper's results:
  * GPT Image models re-render the whole image and treat the mask only as a hint. In testing, the guide
    photo's background and framing were not kept, so the accurate/moderate/imprecise comparison is not meaningful.
  * OpenAI's moderation blocked an edit of an infant guide image (`moderation_blocked`, output stage). Many
    UTKFace MUPs involve children, and the Gateway retries blocked requests like other errors.
  * Outputs are 1024 px; the Gateway resizes them back to the guide image's size, as the paper did.
  * Cost per image is different. Check current pricing before large runs.
* **Bandit exploration α.** The paper doesn't give α. `UCB/config.json` uses α = 1, the usual choice for
  rewards in [0, 1]. The published config had `alpha: 100`, which the old code never read; with it, LinUCB
  would explore almost uniformly for thousands of pulls.
* **Distribution test kernel.** The paper reports ν = 0.3 with both linear and RBF kernels. Pick one with
  `DDT_KERNEL` in `Gateway/.env` (default `rbf`), or set `DDT_ENABLED=false` to turn the test off. With ν = 0.3 the
  SVM treats about 30% of its own training images as outliers, so expect a similar share of generated images to be rejected.
* **Quality test** in the UI is one rater with a binary accept. The paper's t-test across raters (α = 0.1/0.4) was offline.
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
    shutdown. It is now disjoint LinUCB (§5.3) with one bandit per dataset. There is one arm per attribute, and
    arms with no non-empty sibling combination are skipped. State is saved to `${CHAMELEON_DATA_DIR}/ucb/` after every update.
  * Distribution test: it was commented out (`ddt_result = True`), and `mediapipe==0.10.0` has been removed from
    PyPI, so the service couldn't be built. It is now on by default with `mediapipe==0.10.18`, which runs natively on
    arm64. Training embeddings are cached in `${CHAMELEON_DATA_DIR}/ddt_cache/` and the SVM is fitted once
    per training set (a few minutes for all of UTKFace on first use; later runs load the cache).
  * UI: images rejected by the distribution test were counted as added when the remaining count was updated.
* ImageEditor: moved to GPT Image models (base64 responses). It also now starts without an API key.
