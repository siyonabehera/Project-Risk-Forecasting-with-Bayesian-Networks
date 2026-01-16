# Simulation-Driven ML for Semi-Dynamic Project-Delay Prediction

This project builds a simulation-driven machine learning framework to predict whether a project will finish late. Instead of relying on static scheduling assumptions or expert-defined fuzzy/causal rules, the approach generates uncertainty empirically using CPM/PERT-based Monte Carlo simulation and trains supervised models on simulation-derived features.

A key goal is **early-stage forecasting**: the pipeline computes Earned Value Management (EVM) signals—**SPI** and **CPI**—at **20% progress**, enabling semi-dynamic prediction that reflects evolving project conditions.

## What this project does

**Pipeline overview**
1. Parse RG30 `.rcp` project networks into task durations + precedence edges
2. Create PERT triplets (o, m, p) around activity durations to model uncertainty
3. Run CPM forward-pass scheduling to compute baseline duration + critical path
4. Run Monte Carlo simulation (e.g., 200 runs) to estimate delay probability
5. Engineer features from structure, uncertainty, and early performance (SPI/CPI @ 20%)
6. Train and evaluate classifiers to predict `label_delay`

## Data + labeling

- Synthetic project networks generated from the RanGen/RG30 benchmark format
- A project is labeled delayed (`label_delay = 1`) if the Monte Carlo probability of finishing after the deadline exceeds a threshold (p_late ≥ 0.5)
- Deadline defined as a small buffer over baseline CPM duration (e.g., 1% buffer)

## Features (examples)

- Structure: number of edges, network density, critical path length, percent critical tasks
- Uncertainty: mean duration, average PERT range (p-o), instability of durations
- Early health: SPI_early, CPI_early computed at 20% progress
- Target: delay vs. no-delay classification

## Models trained

- Logistic Regression (scaled features)
- Naive Bayes
- Random Forest

Performance is compared to analytical baselines, including a simple rule using early SPI.

## Explainability

Tree-based feature importance + SHAP are used to interpret which early signals and structural/uncertainty features most influence delay predictions.

## Files

- `reports/DS340W_Final_Paper.pdf` — full write-up (methods, novelty, results, discussion)
- `reports/DS340W_Final_Slides.pdf` — presentation overview
- `reports/DS340W_Code.pdf` — implementation (parsing, CPM, Monte Carlo, modeling, SHAP)

## How to run (recommended next step)

This repo currently includes the full implementation in `reports/DS340W_Code.pdf`.
If you want this to be runnable end-to-end, the next step is to extract the code into `src/*.py`
and add a single driver script (e.g., `python -m src.train_models`).

## Author

Siyona Behera
