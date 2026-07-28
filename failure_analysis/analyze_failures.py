"""
WeatherAug-AV: Failure analysis on real-world rainy test set
Computes per-image recall gaps between baseline and augmented models,
then tests whether brightness, brightness variance, or object count
correlate with the performance gap.
"""
import os
import cv2
import numpy as np
from scipy.stats import pearsonr


def load_yolo_boxes(path):
    boxes = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                boxes.append([float(x) for x in parts])
    return boxes


def compute_recall(gt_boxes, pred_boxes, iou_thresh=0.5):
    # Greedy IoU matching between predicted and ground-truth boxes
    # (implementation matches original notebook: match each GT box
    # to the best unclaimed predicted box above iou_thresh)
    matched = 0
    used_preds = set()
    for gt in gt_boxes:
        best_iou, best_idx = 0, -1
        for i, pred in enumerate(pred_boxes):
            if i in used_preds:
                continue
            iou = compute_iou(gt, pred)
            if iou > best_iou:
                best_iou, best_idx = iou, i
        if best_iou >= iou_thresh:
            matched += 1
            used_preds.add(best_idx)
    return matched / len(gt_boxes) if gt_boxes else 0.0


def compute_iou(box_a, box_b):
    # Standard box IoU on YOLO-format (cx, cy, w, h) boxes
    ax1, ay1 = box_a[1] - box_a[3] / 2, box_a[2] - box_a[4] / 2
    ax2, ay2 = box_a[1] + box_a[3] / 2, box_a[2] + box_a[4] / 2
    bx1, by1 = box_b[1] - box_b[3] / 2, box_b[2] - box_b[4] / 2
    bx2, by2 = box_b[1] + box_b[3] / 2, box_b[2] + box_b[4] / 2

    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def mean_brightness(img_dir, image_stem):
    img = cv2.imread(os.path.join(img_dir, image_stem + ".jpg"))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)


def brightness_std(img_dir, image_stem):
    img = cv2.imread(os.path.join(img_dir, image_stem + ".jpg"))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.std(gray)


def run_failure_analysis(img_dir, gt_dir, pred_dirs):
    """
    pred_dirs: dict like {"baseline": path, "curriculum": path, "random": path}
    """
    image_stems = [f[:-4] for f in os.listdir(gt_dir) if f.endswith(".txt")]

    results = []
    for stem in image_stems:
        gt_boxes = load_yolo_boxes(os.path.join(gt_dir, stem + ".txt"))
        recalls = {}
        for name, d in pred_dirs.items():
            pred_boxes = load_yolo_boxes(os.path.join(d, "labels", stem + ".txt"))
            recalls[name] = compute_recall(gt_boxes, pred_boxes)
        results.append({
            "image": stem,
            "n_gt": len(gt_boxes),
            "baseline": recalls["baseline"],
            "curriculum": recalls["curriculum"],
            "random": recalls["random"],
            "gap": recalls["baseline"] - min(recalls["curriculum"], recalls["random"]),
        })

    # Filter to images with at least 5 GT objects
    filtered = [r for r in results if r["n_gt"] >= 5]

    # Compute brightness stats
    for r in filtered:
        r["brightness"] = mean_brightness(img_dir, r["image"])
        r["brightness_std"] = brightness_std(img_dir, r["image"])

    gaps = [r["gap"] for r in filtered]
    brightnesses = [r["brightness"] for r in filtered]
    stds = [r["brightness_std"] for r in filtered]
    counts = [r["n_gt"] for r in filtered]

    corr_b, pval_b = pearsonr(brightnesses, gaps)
    corr_std, pval_std = pearsonr(stds, gaps)
    corr_n, pval_n = pearsonr(counts, gaps)

    print(f"N evaluable images: {len(filtered)}")
    print(f"Correlation (mean brightness vs. gap): r={corr_b:.3f}, p={pval_b:.4f}")
    print(f"Correlation (brightness std vs. gap): r={corr_std:.3f}, p={pval_std:.4f}")
    print(f"Correlation (object count vs. gap): r={corr_n:.3f}, p={pval_n:.4f}")

    return filtered


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", required=True)
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--baseline_pred_dir", required=True)
    parser.add_argument("--curriculum_pred_dir", required=True)
    parser.add_argument("--random_pred_dir", required=True)
    args = parser.parse_args()

    pred_dirs = {
        "baseline": args.baseline_pred_dir,
        "curriculum": args.curriculum_pred_dir,
        "random": args.random_pred_dir,
    }
    run_failure_analysis(args.img_dir, args.gt_dir, pred_dirs)