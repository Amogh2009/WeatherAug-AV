"""
WeatherAug-AV: BDD100K dataset preparation and label conversion
Converts BDD100K JSON labels to YOLO format txt files
"""
import json
import os
from tqdm import tqdm

CATEGORIES = {
    "car": 0, "truck": 1, "bus": 2, "person": 3,
    "rider": 4, "traffic light": 5, "traffic sign": 6,
    "motorcycle": 7, "bicycle": 8
}

IMG_W, IMG_H = 1280, 720

def convert_bdd_to_yolo(label_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    label_files = [f for f in os.listdir(label_dir) if f.endswith(".json")]
    converted, skipped = 0, 0

    for fname in tqdm(label_files):
        with open(os.path.join(label_dir, fname)) as f:
            sample = json.load(f)

        img_name = sample["name"]
        txt_name = os.path.splitext(img_name)[0] + ".txt"
        lines = []

        for obj in sample["frames"][0]["objects"]:
            cat = obj.get("category")
            if cat not in CATEGORIES or "box2d" not in obj:
                continue
            box = obj["box2d"]
            x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
            cx = ((x1 + x2) / 2) / IMG_W
            cy = ((y1 + y2) / 2) / IMG_H
            w = (x2 - x1) / IMG_W
            h = (y2 - y1) / IMG_H
            if w <= 0 or h <= 0:
                skipped += 1
                continue
            lines.append(f"{CATEGORIES[cat]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        with open(os.path.join(output_dir, txt_name), "w") as f:
            f.write("\n".join(lines))
        converted += 1

    print(f"Converted: {converted} | Skipped: {skipped}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--label_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    convert_bdd_to_yolo(args.label_dir, args.output_dir)
