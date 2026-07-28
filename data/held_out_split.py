"""
WeatherAug-AV: Held-out real-world test set construction
Selects a fixed 2,000-rainy / 2,000-snowy held-out test set from the
BDD100K training pool, using a fixed random seed for reproducibility.
"""
import json
import os
import random

random.seed(42)  # fixed seed for reproducibility

def build_held_out_split(train_img_path, train_label_path):
    rainy_files = []
    snowy_files = []

    train_labels = [f for f in os.listdir(train_label_path) if f.endswith('.json')]

    for fname in train_labels:
        with open(os.path.join(train_label_path, fname)) as f:
            sample = json.load(f)
        weather = sample['attributes']['weather']
        img_name = sample['name'] + '.jpg'
        if weather == 'rainy':
            rainy_files.append(img_name)
        elif weather == 'snowy':
            snowy_files.append(img_name)

    random.shuffle(rainy_files)
    random.shuffle(snowy_files)

    test_rainy = rainy_files[:2000]
    test_snowy = snowy_files[:2000]

    return test_rainy, test_snowy


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_img_dir", required=True)
    parser.add_argument("--train_label_dir", required=True)
    args = parser.parse_args()

    test_rainy, test_snowy = build_held_out_split(args.train_img_dir, args.train_label_dir)

    with open("held_out_test_rainy.txt", "w") as f:
        f.write("\n".join(test_rainy))
    with open("held_out_test_snowy.txt", "w") as f:
        f.write("\n".join(test_snowy))

    print(f"Held-out test rainy: {len(test_rainy)}")
    print(f"Held-out test snowy: {len(test_snowy)}")