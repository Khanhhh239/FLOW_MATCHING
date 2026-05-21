# Flow Matching: Complete Implementation with Paper-Level Metrics

A comprehensive implementation of Conditional Flow Matching (CFM) with Rectified Flow-oriented improvements, including paper-level metrics and publication-ready visualizations.

## Why This Project Exists

This project was created to make Flow Matching practical, reproducible, and easier to study.
Many examples online focus on visuals only, while papers focus on theory. This repository connects both: clear implementation + quantitative evaluation.

## Project Introduction

The repository provides two implementations:
- `flow_matching.py`: stable baseline for learning and fast experiments.
- `improve_flow_matching.py`: enhanced version with EMA, curvature regularization, time-dependent sigma, and optional reflow.

It also exports JSON metrics and plots so results can be compared and reported consistently.

## Project Goals

- Build a reliable baseline implementation of CFM.
- Evaluate quality using paper-level metrics, not only sample plots.
- Compare baseline and improved approaches with measurable evidence.
- Support learning, experimentation, and research reporting workflows.

## Quick Start

```powershell
conda env create -f environment.yml
conda activate flow_matching

python flow_matching.py --dataset 8gaussians --epochs 500
python improve_flow_matching.py --dataset 8gaussians --epochs 300 --use_reflow --reflow_epochs 200
```

## Documentation

- `THEORY_AND_ANALYSIS.md`: mathematical framework, metric interpretation, and analysis template.
- `INSTRUCTIONS.md`: setup, commands, troubleshooting, and tuning guidance.

## Current Output Files

- `outputs/metrics_8gaussians.json`
- `outputs/improved_metrics_8gaussians.json`
- `outputs/metrics_swiss_roll.json`
- `outputs/improved_metrics_swiss_roll.json`
- `outputs/flow_matching_8gaussians.png`
- `outputs/flow_matching_swiss_roll.png`
- `outputs/comparison_8gaussians.png`
- `outputs/comparison_swiss_roll.png`

## References

1. Lipman et al., Flow Matching for Generative Modeling (2022)
2. Liu et al., Flow Straight and Fast (2022)
3. Villani, Optimal Transport: Old and New (2009)
