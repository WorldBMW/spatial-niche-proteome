# spatial-niche-proteome

**Interpretable spatial-ecology features from DINOv2 whole-slide embeddings link tumor histology to the druggable proteome in non-small cell lung cancer.**

This repository contains the complete analysis pipeline and result tables for a morphology-to-multi-omics association study on the public CPTAC-LUAD and CPTAC-LUSC cohorts:

- 203 diagnostic H&E whole-slide images are embedded with the **DINOv2 ViT-L/14** foundation model;
- patches are decomposed into **k = 10 histomorphological niches** by K-means (k justified by a k = 5–15 clustering-quality scan: silhouette + Calinski–Harabasz);
- each slide is summarized into **230 named spatial-ecology features** (niche abundance, Ripley's-K clustering, tumor–immune interface, CAF load, niche diversity, niche-transition topology, pairwise co-occurrence);
- features are associated with 2,000 proteins, 500 phosphosites, and 500 transcripts by **ElasticNet regression under leakage-free nested cross-validation** (covariate residualization and standardization strictly inside each training fold), with Benjamini–Hochberg FDR control (q < 0.05);
- sensitivity analyses include a **global-pooling baseline** (mean DINOv2 embedding), **TCGA-frozen K-means centroids**, and a penalized **Cox survival model**;
- significant hits are annotated with **GO/KEGG/Reactome enrichment** (Enrichr, all-tested-proteins background) and **DrugBank** druggability.

## Key results (strict leakage-free protocol)

| Analysis | Targets | Significant (q < 0.05) | of which \|r\| > 0.3 |
| --- | --- | --- | --- |
| Proteomics, pan-tumor | 2,000 | 96 | 48 |
| Proteomics, LUAD | 2,000 | 63 | 63 |
| Proteomics, LUSC | 2,000 | 69 | 69 |
| Phosphoproteomics | 500 | 4 | 4 |
| Transcriptomics | 500 | 39 | 10 |

Best-powered hits: **CTNND2** (r = +0.80, n = 110), **LRATD1** (r = −0.59, n = 157). The global-pooling baseline and the niche features are complementary (intersection 32 proteins; 64 niche-exclusive associations). Every association is reported with its effective sample size; n < 50 is flagged low-confidence.

## Repository structure

```
spatial-niche-proteome/
├── src/
│   ├── run_cptac.py          # WSI pipeline: embed → niche → features → align
│   └── downstream_omics.py   # leakage-free nested-CV association pipeline
├── scripts/                  # analysis utilities (k-scan, baseline, enrichment,
│                             #   DrugBank annotation, figures, statistics)
├── results_vitl14_jbi/       # primary result tables + k-scan + feature matrix
├── results_vitl14_jbi_r2/    # strict re-analysis results (leakage-free) + baseline
├── requirements.txt
├── LICENSE
└── README.md
```

## Requirements

Python 3.11+:

```bash
pip install -r requirements.txt
```

GPU (CUDA) is required only for the embedding step (`run_cptac.py`); all downstream analyses run on CPU.

## Data sources (public)

| Data | Source |
| --- | --- |
| CPTAC-LUAD/LUSC WSI + proteomics/phosphoproteomics/transcriptomics/clinical | https://proteomics.cancer.gov/data-portal |
| TCGA LUAD/LUSC slides (frozen-centroid training) | https://portal.gdc.cancer.gov |

Raw images and omics matrices are **not** redistributed in this repository (size and redistribution policy); download them from the portals above. The aligned per-case feature matrix is provided in `results_vitl14_jbi/all_features_omics_aligned.csv` (203 cases × 236 features + metadata).

## Reproducing the analysis

```bash
# 1. Embed slides and build spatial-ecology features (GPU, hours)
python src/run_cptac.py --wsi-dir <WSI_DIR> --step all

# 2. Morphology → omics association (leakage-free nested CV)
python src/downstream_omics.py \
    --features results_vitl14_jbi/all_features_omics_aligned.csv \
    --cohort ALL --modalities proteomics \
    --targets-top-k 2000 --seeds 3 --no-permutation \
    --out-dir results/dn_ALL_prot2000

# 3. Clustering-quality scan for niche count k
python scripts/_k_scan.py

# 4. Global-pooling baseline
python scripts/_baseline_global_pool.py

# 5. Enrichment + DrugBank annotation of significant hits
python scripts/_run_enrichment.py
python scripts/_drugbank_annot.py
```

Covariates (age, sex, smoking pack-years) are residualized **inside each outer CV fold** in `downstream_omics.py`; tumor stage was unavailable for most cases and is not used. See the manuscript's Supplementary Methods for full detail.

## Citation

Manuscript under review at the *Journal of Biomedical Informatics*. Citation entry will be added upon publication.

```
@software{spatial_niche_proteome_2026,
  title  = {spatial-niche-proteome: interpretable spatial-ecology features linking histology to the proteome},
  author = {Bai, Mingwei and Gu, Zhengmin and Ni, Zhiping and Zhang, Ye and Guo, Lang},
  year   = {2026},
  url    = {https://github.com/WorldBMW/spatial-niche-proteome}
}
```

## License

MIT — see [LICENSE](LICENSE).
