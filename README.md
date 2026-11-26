# spatial-niche-proteome

**An auditable stress-test framework for pipelines that link pathology foundation-model features to bulk omics — instantiated on DINOv2 whole-slide morphology ↔ CPTAC proteomics.**

The object of this repository is the **association pipeline**, not the encoder. Self-supervised histology encoders (DINOv2, UNI, CONCH) are already published and already robustness-checked; what is not routine is auditing the downstream step that turns 230 morphology features into a list of "significant" proteins. This repository ships that audit as executable code: **five pre-specified orthogonal configuration axes + two null controls + a six-mode silent-failure taxonomy + five machine-executable reporting rules (R1–R5)**. Each axis is a mechanical re-run of a configuration the pipeline already exposes as a command-line switch, so the cost of the audit is a fixed multiplier of the compute you already spend.

The pathology foundation model is the application domain. A pipeline that does *not* expose the five switches below (statistical protocol, covariate set, dictionary source, feature type, outcome platform) cannot be audited with these tools, and the manuscript states that boundary rather than claiming universal applicability.

## What the stress test shows on the shipped cohort

NSCLC discovery panel (CPTAC LUAD+LUSC, n = 203 slides, 2,000 protein targets, positive-direction hits — every number below is a `hits/N` rate, never a raw count):

| Axis | Swap | Hits (positive direction) |
| --- | --- | --- |
| Covariate escalation | no covariates → age + sex → + subtype | 100/2000 → 83/2000 → **3/2000** |
| Dictionary source | CPTAC self-fit (transductive) → TCGA-frozen centroids | 5/2000 → 100/2000 |
| Feature type | spatial niche features → global mean-pooled embedding | 83/2000 → 190/2000 (subtype-adjusted: 3 → 9, intersection exactly {SLPI}) |
| Statistical protocol | mean-of-fold-r t-test → pooled out-of-fold exact test | 40 → 5 (audit snapshot) |
| Outcome platform | protein → RNA → phosphosite | 83/2000 → 55/500 → 0/500 |
| Null control | patient shuffle | 1/2000 |
| Null control | within-slide spatial permutation | 47/2000 (29 of the 83 main hits persist = composition-driven; 54 vanish = arrangement candidates) |

Normalizing any single axis removes up to 97% of the apparent protein hits (100 → 3 tested targets), while the choice of unsupervised dictionary alone swings the count 20-fold — and the patient-shuffle null yields 1/2000, so the attenuation is attributable to confounding and preprocessing fragility, not to the testing machinery. The same pattern is reproduced on PDA (17/755 → 13/820 → 6/1106 across the three pre-registered completeness filters), and the extension panels UCEC (n = 46), GBM (n = 54) and the TCGA WSI–RNA controls KIRC (n = 31) / HNSC (n = 24) are reported in the manuscript's Table 1, with their per-target tables under `results_vitl14_jbi_pancancer/` and `results_vitl14_jbi_r4/`.

## Minimal worked example: run the full five-axis stress test

Everything below runs from this checkout on **CPU only** — the 230-feature matrices for the 203 CPTAC NSCLC cases are included, so no GPU embedding step is needed to reproduce the audit. Only the public CPTAC omics matrices are fetched externally (place them under `data/raw/CPTAC/omics/`).

### Step 0 — pin threads (required for any number you intend to report)

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
FEAT_FROZEN=results_vitl14_jbi/all_features_omics_aligned_frozen.csv   # TCGA-frozen k=10 dictionary
FEAT_TRANS=results_vitl14_jbi/all_features_omics_aligned.csv           # CPTAC self-fit (transductive)
DN="python src/downstream_omics.py"
```

`src/downstream_omics.py` writes one `ALL_<modality>_results.csv` per run into `--out-dir`; each row is one protein/phosphosite/transcript with `n_valid`, `r_pooled`, `q_value`, and the retained `pearson_fold_mean` audit column.

**Cost per cell** (measured on this machine, single CPU thread, CPTAC NSCLC panel): 20 targets × 3 seeds = 155 s wall, ≈2.6 s per target-seed, so run time scales linearly in `--targets-top-k × --seeds` and a full 2,000-target cell is on the order of 3–5 hours. Use `--targets-top-k 200` (≈30 min) to confirm an axis switch behaves as expected, and the full 2,000/500 panels only for numbers you intend to report (R1). Treat the estimate as machine-dependent.

### Axis 1 — statistical protocol (no flag: this is a column check)

The released pipeline reports **pooled out-of-fold** exact Pearson + Benjamini–Hochberg (`r_pooled`, `q_value`). Every CSV also carries the defective alternative (`pearson_fold_mean`, `pearson_std`) that an earlier revision used as its test statistic; it is kept as the audit trail, and R1 forbids reporting from it. The fold-level t-test returned 40 "hits" where the pooled exact test returns 5 on the same features, and half of those 40 did not replicate across seeds. This axis is therefore *read*, not re-run — after you have produced any cell below:

```bash
# works on any output of the current code, e.g. results/demo_cov1/ALL_proteomics_results.csv
python - <<'PY'
import pandas as pd, glob
f = glob.glob("results/demo_cov1/ALL_*_results.csv")[0]   # or a released cell of results_vitl14_jbi_r4st_local/
d = pd.read_csv(f)
print(f, "| targets:", len(d), "| pooled q<0.05:", int((d.q_value < 0.05).sum()))
print("audit columns retained:",
      [c for c in d.columns if "fold" in c or c in ("r_pooled", "q_value", "n_valid")])
PY
```

### Axis 2 — covariate escalation (three runs, same features)

```bash
$DN --features $FEAT_FROZEN --cohort ALL --modalities proteomics --targets-top-k 2000 \
    --seeds 3 --no-permutation --no-cov               --out-dir results/demo_cov0
$DN --features $FEAT_FROZEN --cohort ALL --modalities proteomics --targets-top-k 2000 \
    --seeds 3 --no-permutation                       --out-dir results/demo_cov1
$DN --features $FEAT_FROZEN --cohort ALL --modalities proteomics --targets-top-k 2000 \
    --seeds 3 --no-permutation --adjust-subtype      --out-dir results/demo_cov2
```

Expected (positive-direction rate): 100/2000 → 83/2000 → 3/2000. Read the drop as a rate: the denominators are the 2,000 tested targets, not the sample size.

### Axis 3 — dictionary source (same cases, different unsupervised dictionary)

```bash
$DN --features $FEAT_TRANS --cohort ALL --modalities proteomics --targets-top-k 2000 \
    --seeds 3 --no-permutation --no-cov              --out-dir results/demo_dict_trans
$DN --features $FEAT_FROZEN --cohort ALL --modalities proteomics --targets-top-k 2000 \
    --seeds 3 --no-permutation --no-cov              --out-dir results/demo_dict_frozen
```

Expected: 5/2000 vs 100/2000 — a 20-fold swing from an unsupervised choice made upstream of any association test. To rebuild either dictionary from slides instead of using the shipped matrices, run `python src/run_cptac.py --wsi-dir <WSI_DIR> --step all` (GPU) and `python scripts/_k_scan.py` for the k = 5–15 clustering-quality scan.

### Axis 4 — feature type (spatial niche features vs global mean pooling)

```bash
# global mean-pooled 1024-d DINOv2 embedding, identical CV protocol
python scripts/_baseline_global_pool.py --out results/demo_pooled_cov/ALL_proteomics_results.csv
python scripts/_baseline_global_pool.py --adjust-subtype \
    --out results/demo_pooled_covsub/ALL_proteomics_results.csv
# within-spatial feature-family ablations (same CSV, subset switches)
$DN --features $FEAT_FROZEN --cohort ALL --modalities proteomics --targets-top-k 2000 \
    --seeds 3 --no-permutation --feature-subset no-trans --out-dir results/demo_abund+ripley
$DN --features $FEAT_FROZEN --cohort ALL --modalities proteomics --targets-top-k 2000 \
    --seeds 3 --no-permutation --feature-subset matched-dim --out-dir results/demo_matcheddim
```

Expected: pooled 190/2000 (any-direction 227) vs spatial 83/2000 (87); subtype-adjusted pooled 9 vs spatial 3, intersection exactly {SLPI}. The `matched-dim` subset is the dimensionality control that keeps the spatial-vs-pooled contrast honest.

### Axis 5 — outcome platform

```bash
$DN --features $FEAT_FROZEN --cohort ALL --modalities transcriptomics --targets-top-k 500 \
    --seeds 3 --no-permutation                       --out-dir results/demo_rna_cov
$DN --features $FEAT_FROZEN --cohort ALL --modalities transcriptomics --targets-top-k 500 \
    --seeds 3 --no-permutation --no-cov              --out-dir results/demo_rna_nocov
$DN --features $FEAT_FROZEN --cohort ALL --modalities phosphoproteomics --targets-top-k 500 \
    --seeds 3 --no-permutation                       --out-dir results/demo_phospho_cov
```

Expected: RNA 63/500 (no-cov) → 55/500 (age + sex); phosphosites 0/500 — the platform where arrangement-level signal provably cannot exist at this cohort size, i.e. a boundary condition of the framework rather than a null result to explain away.

### The two null controls (these are what make the axes interpretable)

Both null scripts read the frozen-dictionary matrix from its server-side path,
`data/features/dinov2_vitl14_features_cptac/all_features_frozen_omics_aligned.csv`, which is not
redistributed. Stage the shipped copy there first (same 203 × 239 matrix, different filename):

```bash
mkdir -p data/features/dinov2_vitl14_features_cptac
cp results_vitl14_jbi/all_features_omics_aligned_frozen.csv \
   data/features/dinov2_vitl14_features_cptac/all_features_frozen_omics_aligned.csv

# N1 patient shuffle: permute whole feature rows (case ↔ omics pairing destroyed), metadata untouched.
python scripts/_case_shuffle.py        # → all_features_case_shuffled_omics_aligned.csv (seed 2026)
$DN --features data/features/dinov2_vitl14_features_cptac/all_features_case_shuffled_omics_aligned.csv \
    --cohort ALL --modalities proteomics --targets-top-k 2000 --seeds 3 --no-permutation \
    --out-dir results/demo_null_shuffle

# N2 within-slide spatial permutation: shuffle niche labels *inside* each slide, preserving niche
#   composition and destroying arrangement, then recompute the 230 features.
python scripts/_spatial_shuffle_frozen.py   # → all_features_frozen_shuffled_omics_aligned.csv
$DN --features data/features/dinov2_vitl14_features_cptac/all_features_frozen_shuffled_omics_aligned.csv \
    --cohort ALL --modalities proteomics --targets-top-k 2000 --seeds 3 --no-permutation \
    --out-dir results/demo_null_spatial
```

Expected: shuffle null 1/2000 → empirical false-positive rate 0.05% at the nominal q < 0.05 floor. The spatial permutation returns 47/2000 overall, of which 29 are main-analysis hits that persist (composition-driven) while 54 of the 83 vanish (arrangement candidates) — which is precisely the finding that composition-only signal must not be reported as arrangement-specific (rule R2).

**Caveats on recomputing the nulls.** N1 is self-contained (it only permutes the shipped CSV) and takes about as long as one covariate-axis cell. N2 re-extracts features from per-slide frozen-dictionary niche label maps under `data/processed/dinov2_vitl14_niches_frozen/`, which are large and are not redistributed; regenerate them with `python src/run_cptac.py --wsi-dir <WSI_DIR> --step all` (GPU) or read N2 from the released table in `results_vitl14_jbi_r4/dn_r4_shuffro_cov/` instead. Both released null tables (`results_vitl14_jbi_r4st_local/dn_caseshuffle_cov/`, `results_vitl14_jbi_r4/dn_r4_shuffro_cov/`) can be used without any recomputation — see "Release status" below.

Every command above was executed from this checkout (at `--targets-top-k 5–20`, thread-pinned) to confirm that the flags parse, the shipped matrices load and result tables are written; the hit counts quoted in the expected lines come from the full released runs, not from those smoke runs.

### Reading the output against rules R1–R5

| Rule | Mechanically executable check on the run output |
| --- | --- |
| R1 configuration completeness | A discovery list must ship ≥3 runs: `--no-cov`, fully adjusted, and the patient-shuffle null. Count the `results/demo_*` directories before writing any sentence about "significant proteins". |
| R2 composition readout | Compare each surviving target between `demo_cov2`/`demo_frozen_cov` and `demo_null_spatial`: survives covariates but also survives within-slide permutation ⇒ composition readout, no arrangement claim. |
| R3 arrangement-specific signal | Requires joint survival of `demo_null_spatial` **and** the feature-type contrast (`demo_pooled_cov` at matched n); the classification inherits the permutation configuration's covariate level (here age + sex only). |
| R4 platform/cohort boundary | For `--modalities transcriptomics`, for histologically homogeneous cohorts, or `n_valid < 50`: report hits only as a rate against that cohort's own shuffle null. |
| R5 pipeline integrity | The six silent failure modes of Table 2 are release-blocking checks. Two are observable directly in a run's stdout: the pooled statistic is what `q_value` is computed from (R1), and the covariate non-null rates are echoed per run (`[covariates] 可用协变量 … 非空率 …`, which is how F2 was caught). The other four are *practice* at the launcher layer rather than runtime guards — echo the effective configuration after argument parsing (F3), pin and echo the feature file per task (F4), pin BLAS/OpenMP threads for any reported number (F5; `scripts/_launch_r4st.sh` shows the environment), and diff result vectors across paired runs (F6). Post hoc, however, four of the six are now executed as code against the released tables by `scripts/_kb_rule_check.py`: F1 (C2, pooled-BH provenance of `q_value`), F3/F6 (C6, the covariate switch provably changes counts *and* survivor identity), F5 (C7, all four pinned/unpinned duplicate configurations re-compared); only the run-log items stay outside a table's reach, and the checker records them as not dischargeable rather than passing them. |

## The audit checklist in machine-readable form

`knowledge/pipeline_audit_kb.yaml` is the checklist in data form: the five axes with their switches and the effects observed on the shipped cohorts, the two null controls with the tables that produced their counts, the six failure modes each with a detection predicate, its status (`observed` for F1–F4 and F6, `excluded` for F5) and the rule it links to, the five reporting rules with the check that discharges them and whether that check is implemented in code today, and the five-term classification vocabulary (`composition_readout`, `arrangement_specific`, `platform_or_cohort_artifact`, `low_confidence_small_n`, `excluded_hypothesis`).

It is a controlled vocabulary and rule set, **not** a formal ontology: flat terms, no class hierarchy, no inference, no cross-ontology alignment. The manuscript deliberately does not present it as a knowledge-representation contribution — §8 says the argument does not depend on it — and it is listed in Code availability rather than as a paper claim; its only job is to let the audit be executed and diffed. Every value is transcribed from the manuscript sections named in its `source_sections` field and from the released result tables named in each entry's `evidence` field, so an entry can be re-derived rather than trusted. Consume it to decide what to run before writing an association list:

```bash
python - <<'PY'
import yaml
kb = yaml.safe_load(open("knowledge/pipeline_audit_kb.yaml", encoding="utf-8"))
for a in kb["axes"]:
    print(a["id"], "|", a["question"], "|", ", ".join(a["switches"]))
print("precondition:", kb["scope"]["precondition"]["id"])
print("gates:", ", ".join(r["id"] for r in kb["reporting_rules"]))
print("release-blocking checks:",
      sum(1 for f in kb["failure_modes"] if f["status"] == "observed"), "observed +",
      sum(1 for f in kb["failure_modes"] if f["status"] == "excluded"), "excluded")
PY
```

Or execute every check it declares, against the released per-target tables, as one command:

```bash
python scripts/_kb_rule_check.py     # exit code 0 = no flag; 1 = at least one check failed
```

Which half of the checklist a third party can run without our code is a separate question, and the answer is split by rule rather than uniform: **R1, and R4 wherever a paper publishes its own shuffle arm, are decidable on published tables alone** — the C1–C3 rows of the verdict CSV are exactly that class of check, executed against per-target tables as they appear in a release. **R2–R3 and the F3/F5/F6 result diffs additionally require the switches** (the `--no-cov` / `--null-spatial` / `--feature-type` configurations and the pinned-vs-unpinned paired vectors), so a pipeline that does not expose them is outside this framework's reach. The manuscript states this as limitation (xvii) and the cover letter repeats it; instantiating the re-run half on someone else's implementation is the experiment that would establish transfer, and it is offered as a revision-stage proof of concept, not claimed here.

The checker opens the YAML, reads each rule's `executable_check` / each failure mode's `detection_predicate`, evaluates it as code over the 47 released screens (NSCLC pinned + unpinned, PDA, UCEC/GBM ablations, the k-sensitivity grid, TCGA-KIRC/HNSC RNA), and writes the per-check verdict table to `papers/Journal of Biomedical Informatics/supplementary/S5_checklist_decidability.csv` (the manuscript's Supplementary Data S5; Note N7 narrates it). What the current checkout produces:

```
checklist: 5 axes / 14 declared switch values / 2 nulls / 6 failure predicates / 5 rules / 5 class terms
released per-target screens read: 47 across 6 cohorts (GBM, HNSC, KIRC, NSCLC, PDA, UCEC)
rows emitted: 130   pass: 97   flag: 0   not-dischargeable/absent: 4
```

C1 re-derives **every hit count printed in Table 1 of the manuscript** from the CSVs (47/47); C2 shows the released `q_value` is BH over the pooled out-of-fold association p and that the fold-mean counterfactual cannot generate any released hit set; C3 compares each arm with its own shuffle null; C4/C5 compute the R2/R3 set intersections; C6 proves the covariate switch is not inert; C7 re-runs the pinned/unpinned comparison behind F5; C8 lists the items whose evidence is a run log rather than a column. Rows are labelled `pass` / `flag` / `not_dischargeable`, plus the R4 verdict vocabulary (`signal_above_null`, `at_or_above_own_null`, `no_hits_in_arm`) and the F5 vocabulary (`excluded_not_observed`, `configuration_change_not_threading`) so that an expected noise-floor result is never misread as a failed audit. The script reads result tables and writes only the verdict CSV — it never mutates a released run.

## Full reproduction of the manuscript (paper-round commands)

Submission artifacts are text-first: every file in `papers/Journal of Biomedical Informatics/` is authored in markdown, converted to .docx by `scripts/_md_to_docx_jbi_live.py`, sliced into an upload-ready Main Text by `scripts/_jbi_upload_slice.py`, and rendered to PDF (A4, Times 12pt, figures on their own pages) by `scripts/_jbi_make_pdf.py --with-figures --figures-pdf`. The PDF renderer needs no pandoc, LibreOffice or Word install; it drives headless Chrome/Edge and then reports, per file, the page count, the size, the whitespace-token count of the extracted text (with the counts re-joined across line breaks at existing hyphens) and how many `[Author-side: ...]` placeholders are still visible, so a rendered PDF is a third independent carrier for the word-count check rather than a new source of numbers.

`scripts/_launch_r4st.sh` is the authoritative, thread-pinned launcher for every NSCLC cell of Table 1 (it uses the server-side feature paths; substitute the two shipped CSVs above, or set `FEAT_FROZEN`/`FEAT_TRANS` to `data/features/dinov2_vitl14_features_cptac/…` where the same files live on the analysis server). PDA and the cross-cancer extension panels are `scripts/_pda_downstream.py --nocov` / `--min-cov-avail 0.5` and the `results_vitl14_jbi_pancancer/` drivers; those rounds were run **unpinned** and are labelled as such in Table 1's legend.

## Repository structure

```
spatial-niche-proteome/
├── src/
│   ├── run_cptac.py            # WSI → patches → DINOv2 ViT-L/14 → niches → 230 features
│   ├── features.py             # feature families (abundance, Ripley, interface, CAF,
│   │                           #   diversity, transitions, co-occurrence)
│   ├── downstream_omics.py     # leakage-free nested-CV association protocol + axis switches
│   └── config.py               # encoder/patch/k-means/CV hyperparameters (single source)
├── scripts/
│   ├── _launch_r4st.sh         # thread-pinned launcher for the reported NSCLC round
│   ├── _case_shuffle.py        # null control N1 (patient shuffle)
│   ├── _spatial_shuffle_frozen.py  # null control N2 (within-slide spatial permutation)
│   ├── _baseline_global_pool.py    # feature-type axis: global mean-pooled baseline
│   ├── _pda_downstream.py          # PDA attribution design (three completeness filters)
│   ├── _k_scan.py                  # k = 5–15 clustering-quality scan
│   ├── _jbi_wordcount.py           # verifies the manuscript against JBI word limits
│   ├── _jbi_upload_slice.py        # slices Main Text into upload-ready .docx and counts them
│   ├── _jbi_make_pdf.py            # renders every package .md to PDF with headless Chromium/Edge
│   └── _kb_rule_check.py           # executes the audit checklist over the released tables → S5
├── knowledge/
│   └── pipeline_audit_kb.yaml  # axes, nulls, failure modes + predicates, rules R1–R5 (see below)
├── results_vitl14_jbi/         # shipped 230-feature matrices (transductive + frozen dictionary)
├── results_vitl14_jbi_r4st_local/  # PRIMARY: thread-pinned NSCLC round + patient-shuffle null
├── results_vitl14_jbi_r4/      # unpinned round: PDA panel, spatial-permutation null, survival
├── results_vitl14_jbi_pancancer/   # UCEC/GBM, k-sensitivity, TCGA-KIRC/HNSC cross-platform
├── results_vitl14_jbi_r3/      # attribution audit snapshot (Table 1 protocol-axis row)
├── results_vitl14_jbi_r2/      # superseded round, kept only for the historical r2-vs-r4 contrast
├── figures/                    # Fig 1–5 + Supplementary Fig S1 (PNG review + PDF vector)
├── papers/Journal of Biomedical Informatics/   # submission package (md + docx)
├── requirements.txt
├── LICENSE
└── README.md
```

## Release status of the result tables

The manuscript's Code availability statement promises the per-target tables behind **every cell of Table 1**, the frozen dictionaries, and the paired pinned/unpinned vectors used to settle failure mode F5. As of this commit, only `src/`, `scripts/`, `requirements.txt`, `LICENSE`, `README.md`, `results_vitl14_jbi/` (feature matrices + the superseded r1 round) and `results_vitl14_jbi_r2/` are tracked in Git; `results_vitl14_jbi_r3/`, `results_vitl14_jbi_r4/`, `results_vitl14_jbi_r4st_local/` (the primary thread-pinned round, 3.1 MB), `results_vitl14_jbi_pancancer/` (17 MB) and `figures/` exist in the working copy but are untracked, and `data/` and `papers/` are `.gitignore`d. **Before the Zenodo snapshot, `git add` those five result directories plus `figures/` (~33 MB, 176 CSVs total) and `knowledge/` — otherwise the quickstart's released-table references, the audit-vocabulary artifact cited in §8 of the manuscript, and the availability claim are not satisfiable from the public repository.** `scripts/_launch_r4st.sh` also embeds absolute analysis-server paths and a `screen`-based launcher; keep it as the record of the authoritative commands, but confirm no credentials or hostnames are published with it.

## Requirements

Python 3.11+ (`pip install -r requirements.txt`). GPU/CUDA is needed only for `run_cptac.py` (patch extraction and embedding) and for recomputing the within-slide spatial-permutation features; **every association run in the quickstart above is CPU-only**, and thread pinning is mandatory for any number you report.

## Data sources (public)

| Data | Source |
| --- | --- |
| CPTAC LUAD/LUSC/PDA/UCEC/GBM WSI + proteomics, phosphoproteomics, transcriptomics, clinical | https://proteomics.cancer.gov/data-portal |
| Diagnostic slides via TCIA (PathDB collections for CPTAC and TCGA) | https://www.cancerimagingarchive.net |
| TCGA LUAD/LUSC/KIRC/HNSC slides and RNA-seq (frozen centroids, cross-platform controls) | https://portal.gdc.cancer.gov |

Raw WSIs and full omics matrices are **not** redistributed here (size and redistribution policy); download them from the portals above and place the per-cohort CPTAC tables under `data/raw/CPTAC/omics/`. The aligned per-case feature matrices used by the quickstart **are** included: `results_vitl14_jbi/all_features_omics_aligned.csv` and `all_features_omics_aligned_frozen.csv` (203 cases × 230 features + 9 metadata columns; the two files differ only in which dictionary produced the niches). Per-collection TCIA DOIs and CPTAC/GDC accession numbers are listed in the manuscript's Data availability statement.

## Scope of the claims in this repository

- The framework's methodological validity does not depend on cohort size; the per-cancer hit lists do. UCEC (n = 46), GBM (n = 54), KIRC (n = 31) and HNSC (n = 24) are protocol demonstrations, not biological replications.
- This study validates the framework's methodology, not DINOv2's biology, and it was exercised end-to-end on this pipeline only — not on a third-party published one.
- Protein identities that survive normalization (e.g. SLPI) are test cases for the classification rules, not biomarker claims; computed niches are geometric arrangements of embedding clusters, not pathology-semantic structures (no pathologist reviewed the niche maps).
- Survival analysis is a strict null: morphology adds no prognostic increment under the shared-fold paired protocol (C-index 0.565–0.590 across M0–M4), and nothing here should be read as a weak prognostic gain.
- Cross-cancer non-overlap of hit lists is panel/variance-selection behaviour, not shared biology.
- N varies with configuration because the sample-completeness filters are pre-registered, not tuned post hoc; compare hit **rates**, never raw counts, across configurations.

## Citation

Manuscript submitted to *Briefings in Bioinformatics* as a Problem Solving Protocol ("A five-axis stress-testing protocol for morphology–molecule associations: quantitative audit of compositional confounding, spatial signal and preprocessing fragility across CPTAC cancer cohorts"). A substantially different earlier draft is under consideration at the *Journal of Biomedical Informatics*; this disclosure is stated in the submission cover letter. Citation entry will be added upon publication, and a permanent Zenodo DOI will be minted at acceptance.

```
@software{spatial_niche_proteome_2026,
  title  = {spatial-niche-proteome: an auditable five-axis stress-test framework for
            foundation-model-to-bulk-omics association pipelines},
  author = {Bai, Mingwei and Gu, Zhengmin and Ni, Zhiping and Zhang, Ye and Guo, Lang},
  year   = {2026},
  url    = {https://github.com/WorldBMW/spatial-niche-proteome}
}
```

## License

MIT — see [LICENSE](LICENSE).
