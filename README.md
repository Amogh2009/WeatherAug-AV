# WeatherAug-AV

**Curriculum-based weather augmentation for robust object detection in autonomous vehicles**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

WeatherAug-AV investigates whether curriculum-based weather augmentation improves object detection robustness under adverse weather conditions for autonomous driving systems. We train and evaluate YOLOv8s on BDD100K under three conditions: no augmentation (baseline), random weather augmentation, and curriculum-scheduled weather augmentation (WeatherAug).

Our key finding: while both augmentation strategies improve performance on synthetic controlled weather benchmarks, the advantage of curriculum scheduling over random augmentation is not consistently observed on real-world adverse weather images — suggesting that synthetic benchmarks may overestimate real-world robustness gains.

## Results

### Real-World Adverse Weather (BDD100K held-out test set)

| Model | Clean Val mAP50 | Rainy mAP50 | Snowy mAP50 |
|-------|----------------|-------------|-------------|
| Baseline (no aug) | 0.630 | 0.610 | 0.628 |
| Random Augmentation | 0.625 | 0.600 | 0.622 |
| WeatherAug (curriculum) | 0.625 | 0.605 | 0.618 |

### Synthetic Controlled Benchmarks

| Condition | Baseline | Random Aug | WeatherAug |
|-----------|----------|------------|------------|
| Rain Mild | 0.612 | 0.613 | 0.621 |
| Rain Moderate | 0.584 | 0.609 | 0.614 |
| Rain Severe | 0.548 | 0.602 | 0.600 |
| Snow Mild | 0.548 | 0.580 | 0.587 |
| Snow Moderate | 0.550 | 0.575 | 0.582 |
| Snow Severe | 0.552 | 0.579 | 0.588 |

## Dataset

We use [BDD100K](https://bdd-data.berkeley.edu/) — a large-scale diverse driving dataset with weather annotations. Our experimental split:
- **Training:** 66,000 images (excluding held-out weather test set)
- **Held-out real test:** 2,000 rainy + 2,000 snowy images
- **Synthetic test:** 2,000 images per condition (6 conditions)

## Installation

```bash
git clone https://github.com/Amogh2009/WeatherAug-AV.git
cd WeatherAug-AV
pip install -r requirements.txt
```

## Usage

### Prepare BDD100K Labels
```bash
python data/prepare_bdd100k.py --label_dir /path/to/bdd100k/labels/100k/train --output_dir /path/to/yolo_labels/train
```

### Train
```bash
# Baseline
python train.py --config configs/bdd100k_baseline.yaml --name baseline

# WeatherAug (curriculum)
python train.py --config configs/bdd100k_curriculum.yaml --name weatheraug_curriculum

# Random augmentation
python train.py --config configs/bdd100k_random.yaml --name weatheraug_random
```

### Evaluate
```bash
python evaluate.py --model runs/baseline/weights/best.pt --config configs/bdd100k_baseline.yaml
```

## Citation

If you use this code in your research, please cite:

@misc{anonymous2026weatheraugav,
title={WeatherAug: Evaluating Curriculum-Based Weather Augmentation for Adverse Weather Robustness in Autonomous Vehicle Object Detection},
author={Gupta, Amogh},
year={2026}
}

## Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Albumentations](https://github.com/albumentations-team/albumentations)
- [BDD100K Dataset](https://bdd-data.berkeley.edu/)
