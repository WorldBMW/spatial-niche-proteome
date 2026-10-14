#!/bin/bash
cd /root/autodl-tmp/ai_paper2/experiment2_fm_time
echo "=== STEP 1: niche (frozen KMeans, embed-tag=cptac) ==="
/root/miniconda3/bin/python -u src/run_cptac.py --tag frozen --frozen-kmeans data/processed/kmeans_tcga.pkl --embed-tag cptac --step niche
echo "=== STEP 2: features ==="
/root/miniconda3/bin/python -u src/run_cptac.py --tag frozen --frozen-kmeans data/processed/kmeans_tcga.pkl --embed-tag cptac --step features
echo "=== STEP 3: align ==="
/root/miniconda3/bin/python -u src/run_cptac.py --tag frozen --frozen-kmeans data/processed/kmeans_tcga.pkl --embed-tag cptac --step align
echo "=== ALL DONE ==="
